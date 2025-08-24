# src/processing/forms_check.py
import cv2
import mediapipe as mp
import numpy as np
from src.processing.utils import calculate_angle, draw_joint_angle, distance_2d
import time
from src.processing.feedback import speak_async, speak
from abc import ABC, abstractmethod


mp_pose = mp.solutions.pose


class BaseExerciseChecker(ABC):
    """Abstract base class for exercise form checkers."""
    
    def __init__(self):
        self.state = self.init_state()
    
    @abstractmethod
    def init_state(self):
        """Initialize exercise-specific state."""
        pass
    
    @abstractmethod
    def check_form(self, image, landmarks, state):
        """Check exercise form and return feedback and rep count."""
        pass
    
    def get_point(self, landmarks, part, side):
        """Get x,y coordinates for a landmark."""
        lm = getattr(mp_pose.PoseLandmark, f"{side.upper()}_{part.upper()}")
        return [landmarks[lm.value].x, landmarks[lm.value].y]
    
    def provide_audio_feedback(self, feedback, state, current_time, delay=1.0, cooldown=5.0):
        """Provide audio feedback with timing controls."""
        is_correct = (len(feedback) == 0)
        if not is_correct:
            if state.get("incorrect_start_time") is None:
                state["incorrect_start_time"] = current_time
            else:
                elapsed = current_time - state["incorrect_start_time"]
                if elapsed > delay:
                    for msg in feedback:
                        time_since_last = current_time - state.get("last_spoken_time", 0)
                        if msg != state.get("last_spoken_msg", "") or time_since_last > cooldown:
                            speak_async(msg)
                            state["last_spoken_msg"] = msg
                            state["last_spoken_time"] = current_time
                            break
        else:
            state["incorrect_start_time"] = None


class LungeChecker(BaseExerciseChecker):
    """Form checker for lunge exercises."""
    
    def init_state(self):
        return {
            "ready": False,
            "direction": None,
            "count": 0,
            "last_message": "",
            "message_timer": 0,
            "incorrect_start_time": None,
            "last_spoken_time": 0,
            "last_spoken_msg": "",
        }
    
    def determine_front_leg(self, landmarks):
        """Determine which leg is in front based on z-depth."""
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        return "left" if left_knee.z < right_knee.z else "right"
    
    def check_form(self, image, landmarks, state):
        """Analyze lunge exercise form."""
        feedback = []
        h, w = image.shape[:2]
        current_time = time.time()

        front_leg = self.determine_front_leg(landmarks)
        back_leg = "right" if front_leg == "left" else "left"

        # Get points for analysis
        hip_f = self.get_point(landmarks, "HIP", front_leg)
        knee_f = self.get_point(landmarks, "KNEE", front_leg)
        ankle_f = self.get_point(landmarks, "ANKLE", front_leg)
        foot_f = self.get_point(landmarks, "FOOT_INDEX", front_leg)
        shoulder_f = self.get_point(landmarks, "SHOULDER", front_leg)
        
        hip_b = self.get_point(landmarks, "HIP", back_leg)
        knee_b = self.get_point(landmarks, "KNEE", back_leg)
        ankle_b = self.get_point(landmarks, "ANKLE", back_leg)

        # Calculate angles
        front_knee_angle = calculate_angle(hip_f, knee_f, ankle_f)
        back_knee_angle = calculate_angle(hip_b, knee_b, ankle_b)
        torso_angle = calculate_angle(shoulder_f, hip_f, knee_f)
        ankle_angle = calculate_angle(knee_f, ankle_f, foot_f)

        # Rep counting logic
        if 85 <= front_knee_angle <= 110:
            if not state.get("ready", False):
                state["ready"] = True
                state["direction"] = "down"
                state["last_message"] = "Lunge down detected"
                state["message_timer"] = current_time
        elif front_knee_angle > 160:
            if state.get("ready") and state.get("direction") == "down":
                state["count"] += 1
                state["ready"] = False
                state["direction"] = "up"
                state["last_message"] = f"Rep #{state['count']} completed"
                state["message_timer"] = current_time

        # Form feedback
        if not (90 <= front_knee_angle <= 110):
            feedback.append(f"{front_leg.title()} knee angle should be 90-110 deg")
        if not (90 <= back_knee_angle <= 100):
            feedback.append(f"{back_leg.title()} knee too straight (90-100 deg ideal)")
        if torso_angle < 165:
            feedback.append("Keep your back straight")
        if ankle_angle < 20 or ankle_angle > 35:
            feedback.append("Ankle angle out of range (20-30 deg ideal)")
        if knee_f[0] > foot_f[0]:
            feedback.append(f"{front_leg.title()} knee passed toes")

        # Draw visual feedback
        draw_joint_angle(image, hip_f, knee_f, ankle_f, front_knee_angle, 90, 110, label=f"{front_leg.title()} Knee:")
        draw_joint_angle(image, hip_b, knee_b, ankle_b, back_knee_angle, 90, 100, label=f"{back_leg.title()} Knee:")
        draw_joint_angle(image, shoulder_f, hip_f, knee_f, torso_angle, 165, 180, label="Torso:")
        draw_joint_angle(image, knee_f, ankle_f, foot_f, ankle_angle, 20, 30, label="Ankle:")

        # Display rep count and messages
        cv2.putText(image, f"Reps: {state['count']}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        y_offset = 100
        for msg in feedback:
            cv2.putText(image, msg, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30

        if current_time - state.get("message_timer", 0) < 3:
            cv2.putText(image, state.get("last_message", ""), (30, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        # Audio feedback
        self.provide_audio_feedback(feedback, state, current_time)
        
        return feedback, 1 if state["direction"] == "up" else 0


class OverheadPressChecker(BaseExerciseChecker):
    """Enhanced form checker for overhead press exercises with comprehensive analysis."""
    
    # Exercise-specific constants
    STARTING_RANGE = (10, 100)     # Starting position elbow angles
    TOP_RANGE = (160, 180)         # Full extension elbow angles
    SYNC_TOLERANCE = 20            # Maximum difference between arms
    READY_FRAMES_REQUIRED = 30     # Frames to hold ready position
    COUNTDOWN_DURATION = 5         # Countdown in seconds
    
    # Spine alignment thresholds
    SPINE_FORWARD_MAX = 10         # Maximum forward lean
    SPINE_BACKWARD_MAX = 15        # Maximum backward arch
    SPINE_TOP_POSITION_MAX = 10    # Stricter at top position
    
    # Ready position requirements
    READY_ELBOW_RANGE = (20, 90)   # Can be very bent, close to shoulders
    READY_WRIST_OFFSET = 10        # Pixels above shoulders
    READY_ELBOW_DISTANCE = 80      # Pixels from shoulders horizontally
    READY_ARM_HEIGHT_DIFF = 30     # Maximum height difference between arms
    
    def init_state(self):
        return {
            # Backward compatibility fields
            "ready": False,
            "exercise_started": False,
            "rep_phase": "bottom",  # Also compatible as "starting"
            "count": 0,
            "last_message": "",
            "message_timer": 0,
            "incorrect_start_time": None,
            "last_spoken_time": 0,
            "last_spoken_msg": "",
            "ready_start_time": None,
            "missing_start_time": None,
            "wrist_below_start": None,
            
            # Enhanced state management
            "exercise_state": "waiting_for_ready",  # waiting_for_ready, countdown, exercising
            "valid_reps": 0,
            "invalid_reps": 0,
            
            # Ready position tracking
            "ready_frames_count": 0,
            "countdown_start_time": None,
            "exercise_start_time": None,
            
            # Form tracking
            "angle_buffer_l": [],              # 5-frame smoothing buffer
            "angle_buffer_r": [],
            "spine_violations": 0,
            "sync_violations": 0,
            
            # Feedback management
            "last_audio_time": 0,
            "last_audio_msg": "",
            
            # Event-based feedback state tracking
            "previous_spine_straight": True,
            "previous_elbows_bent": True,
            "previous_arms_synchronized": True,
            "previous_spine_severity": "good",  # good, caution, warning, critical
            "previous_sync_severity": "good",
            "form_state_changed": False,
            "priority_feedback": [],
            
            # Rep validation tracking
            "rep_full_range": False,
            "rep_sync_maintained": True,
            "rep_spine_ok": True,
            "phase_entry_time": 0,
        }
    
    def all_keypoints_visible(self, landmarks):
        """Check if all required keypoints are visible."""
        required_keypoints = [
            "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
            "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP",
            "LEFT_KNEE", "RIGHT_KNEE", "NOSE"
        ]
        for name in required_keypoints:
            lm = getattr(mp_pose.PoseLandmark, name)
            if landmarks[lm.value].visibility < 0.5:
                return False
        return True
    
    def get_missing_keypoints(self, landmarks):
        """Get list of missing/low visibility keypoints for specific feedback."""
        required_keypoints = {
            "LEFT_SHOULDER": "left shoulder", "RIGHT_SHOULDER": "right shoulder", 
            "LEFT_ELBOW": "left elbow", "RIGHT_ELBOW": "right elbow",
            "LEFT_WRIST": "left wrist", "RIGHT_WRIST": "right wrist", 
            "LEFT_HIP": "left hip", "RIGHT_HIP": "right hip",
            "LEFT_KNEE": "left knee", "RIGHT_KNEE": "right knee", 
            "NOSE": "head/face"
        }
        
        missing = []
        for name, display_name in required_keypoints.items():
            lm = getattr(mp_pose.PoseLandmark, name)
            if landmarks[lm.value].visibility < 0.5:
                missing.append(display_name)
        
        return missing
    
    def get_smoothed_angle(self, angle, buffer, buffer_size=5):
        """Apply 5-frame smoothing to reduce noise."""
        buffer.append(angle)
        if len(buffer) > buffer_size:
            buffer.pop(0)
        return sum(buffer) / len(buffer)
    
    def calculate_spine_alignment(self, landmarks):
        """Calculate spine deviation from vertical."""
        # Get nose point directly since it doesn't follow LEFT/RIGHT pattern
        nose_lm = landmarks[mp_pose.PoseLandmark.NOSE.value]
        nose = [nose_lm.x, nose_lm.y]
        
        shoulder_l = self.get_point(landmarks, "SHOULDER", "LEFT")
        shoulder_r = self.get_point(landmarks, "SHOULDER", "RIGHT")
        hip_l = self.get_point(landmarks, "HIP", "LEFT")
        hip_r = self.get_point(landmarks, "HIP", "RIGHT")
        
        # Calculate center points
        shoulder_center = [(shoulder_l[0] + shoulder_r[0]) / 2, (shoulder_l[1] + shoulder_r[1]) / 2]
        hip_center = [(hip_l[0] + hip_r[0]) / 2, (hip_l[1] + hip_r[1]) / 2]
        
        # Calculate spine angle from vertical
        spine_vector = [shoulder_center[0] - hip_center[0], shoulder_center[1] - hip_center[1]]
        vertical_vector = [0, -1]
        
        # Calculate angle between spine and vertical
        dot_product = spine_vector[0] * vertical_vector[0] + spine_vector[1] * vertical_vector[1]
        spine_magnitude = (spine_vector[0]**2 + spine_vector[1]**2)**0.5
        
        if spine_magnitude == 0:
            return 0
        
        cos_angle = dot_product / spine_magnitude
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
        spine_angle = np.arccos(cos_angle) * 180 / np.pi
        
        # Determine forward/backward lean
        if spine_vector[0] > 0:  # Leaning forward
            return spine_angle
        else:  # Leaning backward
            return -spine_angle
    
    def is_ready_position_enhanced(self, landmarks):
        """Enhanced ready position detection with detailed requirements."""
        # Get all required points
        shoulder_l = self.get_point(landmarks, "SHOULDER", "LEFT")
        shoulder_r = self.get_point(landmarks, "SHOULDER", "RIGHT")
        elbow_l = self.get_point(landmarks, "ELBOW", "LEFT")
        elbow_r = self.get_point(landmarks, "ELBOW", "RIGHT")
        wrist_l = self.get_point(landmarks, "WRIST", "LEFT")
        wrist_r = self.get_point(landmarks, "WRIST", "RIGHT")
        
        # Calculate elbow angles
        elbow_angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
        elbow_angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
        
        # Check elbow angles (can be very bent, close to shoulders)
        elbows_bent = (self.READY_ELBOW_RANGE[0] <= elbow_angle_l <= self.READY_ELBOW_RANGE[1] and
                      self.READY_ELBOW_RANGE[0] <= elbow_angle_r <= self.READY_ELBOW_RANGE[1])
        
        # Check hands above shoulders (TEMPORARILY BYPASSED FOR TESTING)
        # hands_above = (wrist_l[1] < shoulder_l[1] - self.READY_WRIST_OFFSET and
        #               wrist_r[1] < shoulder_r[1] - self.READY_WRIST_OFFSET)
        hands_above = True  # Temporarily set to True to bypass this requirement
        
        # Check elbows close to shoulders horizontally
        elbow_distance_l = abs(elbow_l[0] - shoulder_l[0])
        elbow_distance_r = abs(elbow_r[0] - shoulder_r[0])
        elbows_close = (elbow_distance_l < self.READY_ELBOW_DISTANCE and
                       elbow_distance_r < self.READY_ELBOW_DISTANCE)
        
        # Check spine alignment
        spine_deviation = abs(self.calculate_spine_alignment(landmarks))
        spine_straight = spine_deviation < 15
        
        # Check arm symmetry
        arm_height_diff = abs(wrist_l[1] - wrist_r[1])
        arms_symmetric = arm_height_diff < self.READY_ARM_HEIGHT_DIFF
        
        return {
            "ready": elbows_bent and hands_above and elbows_close and spine_straight and arms_symmetric,
            "elbows_bent": elbows_bent,
            "hands_above": hands_above,
            "elbows_close": elbows_close,
            "spine_straight": spine_straight,
            "arms_symmetric": arms_symmetric,
            "elbow_angle_l": elbow_angle_l,
            "elbow_angle_r": elbow_angle_r,
            "spine_deviation": spine_deviation
        }
    
    def detect_rep_phase(self, angle_l, angle_r, current_phase, is_synchronized):
        """Detect the current phase of the rep using the weaker angle."""
        # Use the minimum angle (weaker arm) for phase detection
        min_angle = min(angle_l, angle_r)
        
        if current_phase == "starting":
            if min_angle > 100 and is_synchronized:
                return "pressing_up"
        elif current_phase == "pressing_up":
            if min_angle >= self.TOP_RANGE[0]:
                return "top_position"
        elif current_phase == "top_position":
            if min_angle < 160:
                return "lowering"
        elif current_phase == "lowering":
            if min_angle <= 100:
                return "starting"
        
        return current_phase
    
    def check_arm_synchronization(self, angle_l, angle_r):
        """Check if arms are moving synchronously."""
        angle_diff = abs(angle_l - angle_r)
        is_synchronized = angle_diff <= self.SYNC_TOLERANCE
        
        feedback = []
        if not is_synchronized:
            if angle_l < angle_r:
                feedback.append("Straighten your left arm more")
            else:
                feedback.append("Straighten your right arm more")
        
        return is_synchronized, feedback
    
    def check_arm_synchronization_enhanced(self, angle_l, angle_r, state):
        """Enhanced arm synchronization checking with detailed feedback."""
        angle_diff = abs(angle_l - angle_r)
        is_synchronized = angle_diff <= self.SYNC_TOLERANCE
        
        feedback = []
        
        if not is_synchronized:
            # More specific feedback based on angle difference severity
            if angle_diff > self.SYNC_TOLERANCE * 2:  # Severe async
                if angle_l < angle_r:
                    feedback.append(f"LEFT ARM LAGGING: Straighten left arm ({angle_diff:.0f} deg behind)")
                else:
                    feedback.append(f"RIGHT ARM LAGGING: Straighten right arm ({angle_diff:.0f} deg behind)")
            else:  # Moderate async
                if angle_l < angle_r:
                    feedback.append("Straighten your left arm more")
                else:
                    feedback.append("Straighten your right arm more")
            
            # Track persistent synchronization issues
            sync_violations = state.get("sync_violations", 0)
            if sync_violations > 10:  # After 10 violations
                feedback.append("Focus on moving both arms together")
        
        return is_synchronized, feedback
    
    def detect_rep_phase_enhanced(self, angle_l, angle_r, current_phase, is_synchronized, phase_duration):
        """Enhanced rep phase detection with timing and synchronization requirements."""
        # Use the minimum angle (weaker arm) for phase detection
        min_angle = min(angle_l, angle_r)
        max_angle = max(angle_l, angle_r)
        
        # Enhanced phase transitions with timing constraints
        if current_phase == "starting":
            # Must be synchronized and above threshold to start pressing
            if min_angle > 100 and is_synchronized and phase_duration > 0.5:
                return "pressing_up"
        elif current_phase == "pressing_up":
            # Must reach top range for both arms
            if min_angle >= self.TOP_RANGE[0] and max_angle >= self.TOP_RANGE[0]:
                return "top_position"
            # Prevent false transitions on slight decreases
            elif min_angle < 90 and phase_duration > 2.0:
                return "starting"  # User gave up mid-press
        elif current_phase == "top_position":
            # Require a significant decrease to start lowering
            if min_angle < 155 and phase_duration > 0.3:
                return "lowering"
        elif current_phase == "lowering":
            # Return to starting position
            if min_angle <= 100 and phase_duration > 0.5:
                return "starting"
        
        return current_phase
    
    def calculate_rep_quality(self, state):
        """Calculate overall rep quality score based on multiple factors."""
        quality_factors = {
            "range_of_motion": 0.4,  # 40% weight - most important
            "synchronization": 0.3,  # 30% weight
            "spine_alignment": 0.3   # 30% weight
        }
        
        scores = {
            "range_of_motion": 1.0 if state.get("rep_full_range", False) else 0.5,
            "synchronization": max(0.0, 1.0 - (state.get("sync_violations", 0) * 0.1)),
            "spine_alignment": max(0.0, 1.0 - (state.get("spine_violations", 0) * 0.1))
        }
        
        # Calculate weighted average
        total_score = sum(scores[factor] * weight for factor, weight in quality_factors.items())
        return min(1.0, max(0.0, total_score))
    
    def display_phase_progress(self, image, state, angle_l, angle_r, current_time):
        """Display enhanced phase indicator with progress visualization."""
        h, w = image.shape[:2]
        
        # Phase colors and names
        phase_info = {
            "starting": {"color": (255, 255, 0), "name": "Starting Position", "target": "Get ready to press"},
            "pressing_up": {"color": (0, 255, 255), "name": "Pressing Up", "target": "Press to full extension"},
            "top_position": {"color": (0, 255, 0), "name": "Top Position", "target": "Hold briefly, then lower"},
            "lowering": {"color": (255, 165, 0), "name": "Lowering", "target": "Lower with control"}
        }
        
        current_phase = state.get("rep_phase", "starting")
        phase_data = phase_info.get(current_phase, phase_info["starting"])
        
        # Main phase display
        cv2.putText(image, f"Phase: {phase_data['name']}", 
                   (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, phase_data["color"], 2)
        cv2.putText(image, phase_data["target"], 
                   (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, phase_data["color"], 1)
        
        # Progress bar for current phase
        min_angle = min(angle_l, angle_r)
        
        if current_phase == "pressing_up":
            progress = (min_angle - 100) / (self.TOP_RANGE[0] - 100)
        elif current_phase == "lowering":
            progress = (self.TOP_RANGE[0] - min_angle) / (self.TOP_RANGE[0] - 100)
        else:
            progress = 1.0  # Full for starting and top position
        
        progress = max(0.0, min(1.0, progress))
        
        # Draw progress bar
        bar_x, bar_y = 30, 150
        bar_width, bar_height = 200, 15
        
        # Background
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        # Progress fill
        fill_width = int(bar_width * progress)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), phase_data["color"], -1)
        # Border
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)
        
        # Progress percentage
        cv2.putText(image, f"{progress:.0%}", (bar_x + bar_width + 10, bar_y + bar_height - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def prioritize_feedback(self, spine_feedback, sync_feedback, range_feedback, form_feedback):
        """Prioritize feedback messages based on safety and importance."""
        priority_feedback = []
        
        # Priority 1: Spine issues (safety critical)
        priority_feedback.extend(spine_feedback)
        
        # Priority 2: Arm synchronization
        priority_feedback.extend(sync_feedback)
        
        # Priority 3: Range of motion
        priority_feedback.extend(range_feedback)
        
        # Priority 4: Form corrections
        priority_feedback.extend(form_feedback)
        
        return priority_feedback[:3]  # Limit to top 3 feedback items
    
    def is_wrist_above_shoulder(self, wrist, shoulder):
        """Check if wrist is above shoulder (backward compatibility)."""
        return wrist[1] < shoulder[1]
    
    def should_provide_audio(self, message, state, current_time, delay=3.0, repeat_delay=5.0):
        """Smart audio timing to avoid overwhelming the user."""
        last_audio_time = state.get("last_audio_time", 0)
        last_audio_msg = state.get("last_audio_msg", "")
        
        # Initial feedback - immediate
        if message != last_audio_msg:
            return True
        
        # Repeat logic - if not corrected after repeat_delay seconds
        if current_time - last_audio_time > repeat_delay:
            return True
        
        return False
    
    def get_spine_severity(self, spine_deviation):
        """Determine spine deviation severity level."""
        if spine_deviation <= 5:
            return "good"
        elif spine_deviation <= 10:
            return "caution"
        elif spine_deviation <= 15:
            return "warning"
        else:
            return "critical"
    
    def get_sync_severity(self, angle_diff):
        """Determine arm synchronization severity level."""
        if angle_diff <= 10:
            return "good"
        elif angle_diff <= 20:
            return "caution"
        else:
            return "warning"
    
    def detect_form_state_changes(self, current_spine_straight, current_elbows_bent, 
                                 current_arms_synchronized, spine_deviation, sync_diff, state):
        """Detect significant changes in form state that warrant feedback."""
        state_changed = False
        
        # Check spine state change
        prev_spine_straight = state.get("previous_spine_straight", True)
        if current_spine_straight != prev_spine_straight:
            state["previous_spine_straight"] = current_spine_straight
            state_changed = True
        
        # Check elbow state change
        prev_elbows_bent = state.get("previous_elbows_bent", True)
        if current_elbows_bent != prev_elbows_bent:
            state["previous_elbows_bent"] = current_elbows_bent
            state_changed = True
        
        # Check arm sync state change
        prev_arms_synchronized = state.get("previous_arms_synchronized", True)
        if current_arms_synchronized != prev_arms_synchronized:
            state["previous_arms_synchronized"] = current_arms_synchronized
            state_changed = True
        
        # Check spine severity change
        current_spine_severity = self.get_spine_severity(spine_deviation)
        prev_spine_severity = state.get("previous_spine_severity", "good")
        if current_spine_severity != prev_spine_severity:
            state["previous_spine_severity"] = current_spine_severity
            state_changed = True
        
        # Check sync severity change
        current_sync_severity = self.get_sync_severity(sync_diff)
        prev_sync_severity = state.get("previous_sync_severity", "good")
        if current_sync_severity != prev_sync_severity:
            state["previous_sync_severity"] = current_sync_severity
            state_changed = True
        
        state["form_state_changed"] = state_changed
        return state_changed
    
    def check_form(self, image, landmarks, state):
        """Enhanced overhead press form analysis with comprehensive feedback and disabled auto-stopping."""
        feedback = []
        reps_count = 0
        h, w = image.shape[:2]
        current_time = time.time()

        # Check for missing keypoints - provide helpful guidance but NEVER stop session
        if not self.all_keypoints_visible(landmarks):
            missing_parts = self.get_missing_keypoints(landmarks)
            if len(missing_parts) == 1:
                feedback.append(f"Adjust position: {missing_parts[0]} not visible")
            elif len(missing_parts) <= 3:
                feedback.append(f"Adjust position: {', '.join(missing_parts)} not visible")
            else:
                feedback.append("Move closer to camera or adjust lighting")
            
            cv2.putText(image, "Adjust your position in camera view", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            cv2.putText(image, "Session continues - fix positioning when ready", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # CRITICAL: Session continues regardless of keypoint detection issues
            # Only provide feedback, never stop the session
            return feedback, 0
        else:
            # Reset any previous warning states when keypoints are restored
            state["missing_start_time"] = None

        # Get keypoints
        shoulder_l = self.get_point(landmarks, "SHOULDER", "LEFT")
        elbow_l = self.get_point(landmarks, "ELBOW", "LEFT")
        wrist_l = self.get_point(landmarks, "WRIST", "LEFT")
        hip_l = self.get_point(landmarks, "HIP", "LEFT")
        
        shoulder_r = self.get_point(landmarks, "SHOULDER", "RIGHT")
        elbow_r = self.get_point(landmarks, "ELBOW", "RIGHT")
        wrist_r = self.get_point(landmarks, "WRIST", "RIGHT")
        hip_r = self.get_point(landmarks, "HIP", "RIGHT")

        # Calculate and smooth angles
        raw_elbow_angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
        raw_elbow_angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
        
        elbow_angle_l = self.get_smoothed_angle(raw_elbow_angle_l, state["angle_buffer_l"])
        elbow_angle_r = self.get_smoothed_angle(raw_elbow_angle_r, state["angle_buffer_r"])
        
        # Calculate spine alignment
        spine_deviation = self.calculate_spine_alignment(landmarks)
        
        # Initialize feedback categories
        spine_feedback = []
        sync_feedback = []
        range_feedback = []
        form_feedback = []
        
        # State machine handling
        exercise_state = state.get("exercise_state", "waiting_for_ready")
        
        if exercise_state == "waiting_for_ready":
            ready_status = self.is_ready_position_enhanced(landmarks)
            
            # Display comprehensive setup instructions
            self.display_setup_instructions(image, ready_status, current_time)
            
            if ready_status["ready"]:
                state["ready_frames_count"] += 1
                frames_remaining = self.READY_FRAMES_REQUIRED - state["ready_frames_count"]
                cv2.putText(image, f"Perfect! Hold position ({frames_remaining} frames)", 
                           (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Provide audio encouragement
                if state["ready_frames_count"] == 1:  # First time achieving ready position
                    if self.should_provide_audio("Excellent form! Hold this position", state, current_time):
                        speak_async("Excellent form! Hold this position")
                        state["last_audio_time"] = current_time
                        state["last_audio_msg"] = "Excellent form! Hold this position"
                
                if state["ready_frames_count"] >= self.READY_FRAMES_REQUIRED:
                    state["exercise_state"] = "countdown"
                    state["countdown_start_time"] = current_time
                    state["ready_frames_count"] = 0
                    speak_async("Perfect! Preparing to start exercise")
            else:
                state["ready_frames_count"] = 0
                
                # Provide detailed setup guidance
                setup_guidance = self.provide_detailed_setup_guidance(ready_status, state, current_time)
                form_feedback.extend(setup_guidance["corrections_needed"])
                
                # Priority feedback for setup
                if not ready_status["spine_straight"]:
                    spine_feedback.append("Stand tall with straight back - engage your core")
                if not ready_status["elbows_bent"]:
                    form_feedback.append("Bend your elbows more - bring weights closer to shoulders")
                # TEMPORARILY DISABLED: hands_above requirement
                # if not ready_status["hands_above"]:
                #     form_feedback.append("Raise your hands above shoulder level")
                if not ready_status["elbows_close"]:
                    form_feedback.append("Position hands directly above your shoulders")
                if not ready_status["arms_symmetric"]:
                    sync_feedback.append("Keep both hands at the same height")
        
        elif exercise_state == "countdown":
            elapsed = current_time - state["countdown_start_time"]
            countdown_remaining = self.COUNTDOWN_DURATION - int(elapsed)
            
            if countdown_remaining > 0:
                cv2.putText(image, f"Starting in {countdown_remaining}", 
                           (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            else:
                state["exercise_state"] = "exercising"
                state["exercise_start_time"] = current_time
                state["rep_phase"] = "starting"
                cv2.putText(image, "GO! Start pressing!", 
                           (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        elif exercise_state == "exercising":
            # Enhanced continuous spine monitoring (priority feedback)
            spine_max = self.SPINE_TOP_POSITION_MAX if state["rep_phase"] == "top_position" else self.SPINE_BACKWARD_MAX
            
            # Event-based spine feedback (only on state changes)
            current_spine_straight = not (spine_deviation > self.SPINE_FORWARD_MAX or spine_deviation < -spine_max)
            
            # Calculate sync difference for event detection
            sync_diff = abs(elbow_angle_l - elbow_angle_r)
            is_synchronized = sync_diff <= 15
            
            # Detect form state changes
            form_changed = self.detect_form_state_changes(
                current_spine_straight, True, is_synchronized, 
                abs(spine_deviation), sync_diff, state
            )
            
            # Only generate spine feedback on state changes or severity escalation
            if form_changed and not current_spine_straight:
                current_severity = self.get_spine_severity(abs(spine_deviation))
                if spine_deviation > self.SPINE_FORWARD_MAX:
                    if current_severity == "critical":
                        spine_feedback.append("CRITICAL: Stand up straight - severe forward lean")
                    elif current_severity == "warning":
                        spine_feedback.append("Don't lean forward - keep chest up")
                elif spine_deviation < -spine_max:
                    if abs(spine_deviation) > 20:
                        spine_feedback.append("CRITICAL: Don't arch back - severe hyperextension")
                    else:
                        spine_feedback.append("Don't arch your back - stay neutral")
                
                state["rep_spine_ok"] = False
                state["spine_violations"] += 1
            else:
                state["rep_spine_ok"] = current_spine_straight
                # Reset spine violation counter when corrected
                if current_spine_straight and state.get("spine_violations", 0) > 0:
                    state["spine_violations"] = 0
            
            # Event-based arm synchronization checking
            # Only generate sync feedback if there was a state change
            if form_changed and not is_synchronized:
                sync_severity = self.get_sync_severity(sync_diff)
                if sync_severity == "warning":
                    sync_feedback.append("Synchronize your arms - significant difference detected")
                elif sync_severity == "caution":
                    sync_feedback.append("Keep arms moving together")
                
                state["rep_sync_maintained"] = False
                state["sync_violations"] += 1
            else:
                state["rep_sync_maintained"] = is_synchronized
                # Reset sync violation counter when corrected
                if is_synchronized and state.get("sync_violations", 0) > 0:
                    state["sync_violations"] = 0
            
            # Enhanced rep phase detection with better thresholds
            old_phase = state["rep_phase"]
            new_phase = self.detect_rep_phase_enhanced(elbow_angle_l, elbow_angle_r, old_phase, is_synchronized, current_time - state.get("phase_entry_time", current_time))
            
            if new_phase != old_phase:
                state["rep_phase"] = new_phase
                state["phase_entry_time"] = current_time
                
                # Phase transition feedback
                if new_phase == "pressing_up":
                    speak_async("Press up steadily")
                elif new_phase == "top_position":
                    speak_async("Good! Now lower slowly")
                
                # Enhanced rep completion validation
                if new_phase == "starting" and old_phase == "lowering":
                    rep_quality_score = self.calculate_rep_quality(state)
                    
                    if rep_quality_score >= 0.8:  # 80% quality threshold
                        state["count"] += 1
                        state["valid_reps"] += 1
                        reps_count = 1
                        state["last_message"] = f"Excellent Rep #{state['count']} (Quality: {rep_quality_score:.1%})"
                        state["message_timer"] = current_time
                        speak_async(f"Excellent rep {state['count']}!")
                    elif rep_quality_score >= 0.6:  # 60% partial credit
                        state["count"] += 1
                        state["valid_reps"] += 1
                        reps_count = 1
                        state["last_message"] = f"Good Rep #{state['count']} (Quality: {rep_quality_score:.1%})"
                        state["message_timer"] = current_time
                        speak_async(f"Good rep {state['count']}, focus on form")
                    else:
                        state["invalid_reps"] += 1
                        state["last_message"] = f"Rep not counted - maintain proper form (Quality: {rep_quality_score:.1%})"
                        state["message_timer"] = current_time
                        speak_async("Rep not counted - maintain proper form")
                    
                    # Reset rep tracking flags
                    state["rep_full_range"] = False
                    state["rep_sync_maintained"] = True
                    state["rep_spine_ok"] = True
            
            # Enhanced range of motion tracking
            min_angle = min(elbow_angle_l, elbow_angle_r)
            max_angle = max(elbow_angle_l, elbow_angle_r)
            
            if min_angle >= self.TOP_RANGE[0]:
                state["rep_full_range"] = True
            
            # Phase-specific feedback
            if state["rep_phase"] == "top_position":
                if min_angle < self.TOP_RANGE[0]:
                    range_feedback.append(f"Press higher - need {self.TOP_RANGE[0] - min_angle:.0f} deg more")
            elif state["rep_phase"] == "pressing_up":
                if max_angle - min_angle > self.SYNC_TOLERANCE:
                    range_feedback.append("Keep arms moving together")
            elif state["rep_phase"] == "lowering":
                if current_time - state.get("phase_entry_time", current_time) > 1.0:  # Slower lowering
                    range_feedback.append("Lower slowly with control")
            
            # Enhanced visual phase indicator with progress
            self.display_phase_progress(image, state, elbow_angle_l, elbow_angle_r, current_time)

        # Prioritize and limit feedback
        priority_feedback = self.prioritize_feedback(spine_feedback, sync_feedback, range_feedback, form_feedback)
        
        # Enhanced visual feedback with improved status panel
        self.draw_enhanced_visuals(image, landmarks, elbow_angle_l, elbow_angle_r, spine_deviation, state)
        
        # Draw the redesigned status panel
        self.draw_improved_status_panel(image, elbow_angle_l, elbow_angle_r, spine_deviation, state)
        
        # Smart audio feedback
        for msg in priority_feedback[:1]:  # Only speak the most important message
            if self.should_provide_audio(msg, state, current_time):
                speak_async(msg)
                state["last_audio_time"] = current_time
                state["last_audio_msg"] = msg
                break
        
        # Display status and feedback
        self.display_status_info(image, state, priority_feedback, current_time)
        
        return priority_feedback, reps_count
    
    def draw_enhanced_visuals(self, image, landmarks, angle_l, angle_r, spine_deviation, state):
        """Draw enhanced visual feedback with comprehensive color coding and form analysis."""
        h, w = image.shape[:2]
        
        # Get keypoints for drawing
        shoulder_l = self.get_point(landmarks, "SHOULDER", "LEFT")
        elbow_l = self.get_point(landmarks, "ELBOW", "LEFT")
        wrist_l = self.get_point(landmarks, "WRIST", "LEFT")
        shoulder_r = self.get_point(landmarks, "SHOULDER", "RIGHT")
        elbow_r = self.get_point(landmarks, "ELBOW", "RIGHT")
        wrist_r = self.get_point(landmarks, "WRIST", "RIGHT")
        
        # Convert to pixel coordinates
        def to_pixels(point):
            return (int(point[0] * w), int(point[1] * h))
        
        shoulder_l_px = to_pixels(shoulder_l)
        elbow_l_px = to_pixels(elbow_l)
        wrist_l_px = to_pixels(wrist_l)
        shoulder_r_px = to_pixels(shoulder_r)
        elbow_r_px = to_pixels(elbow_r)
        wrist_r_px = to_pixels(wrist_r)
        
        # Enhanced color coding based on form and synchronization
        def get_enhanced_angle_color(angle, target_range, is_sync, sync_violations):
            # Base color from angle correctness
            if target_range[0] <= angle <= target_range[1]:
                base_color = (0, 255, 0)  # Green - correct
            elif abs(angle - target_range[0]) < 20 or abs(angle - target_range[1]) < 20:
                base_color = (0, 255, 255)  # Yellow - close
            else:
                base_color = (0, 0, 255)  # Red - incorrect
            
            # Modify color based on synchronization issues
            if not is_sync and sync_violations > 5:
                return (255, 0, 255)  # Magenta - persistent sync issues
            elif not is_sync:
                return (255, 165, 0)  # Orange - sync warning
            
            return base_color
        
        # Determine target range based on exercise state and phase
        exercise_state = state.get("exercise_state", "waiting_for_ready")
        
        if exercise_state == "exercising":
            current_phase = state.get("rep_phase", "starting")
            if current_phase == "top_position":
                target_range = self.TOP_RANGE
            elif current_phase in ["pressing_up", "lowering"]:
                target_range = (100, 180)  # Transition range
            else:
                target_range = self.STARTING_RANGE
        else:
            target_range = self.READY_ELBOW_RANGE
        
        # Check synchronization
        is_synchronized = abs(angle_l - angle_r) <= self.SYNC_TOLERANCE
        sync_violations = state.get("sync_violations", 0)
        
        color_l = get_enhanced_angle_color(angle_l, target_range, is_synchronized, sync_violations)
        color_r = get_enhanced_angle_color(angle_r, target_range, is_synchronized, sync_violations)
        
        # Draw arms with thickness based on form quality
        arm_thickness = 6 if is_synchronized else 4
        
        # Draw left arm with joint circles
        cv2.line(image, shoulder_l_px, elbow_l_px, color_l, arm_thickness)
        cv2.line(image, elbow_l_px, wrist_l_px, color_l, arm_thickness)
        cv2.circle(image, elbow_l_px, 8, color_l, -1)  # Elbow joint
        cv2.circle(image, wrist_l_px, 6, color_l, -1)  # Wrist joint
        
        # Draw right arm with joint circles
        cv2.line(image, shoulder_r_px, elbow_r_px, color_r, arm_thickness)
        cv2.line(image, elbow_r_px, wrist_r_px, color_r, arm_thickness)
        cv2.circle(image, elbow_r_px, 8, color_r, -1)  # Elbow joint
        cv2.circle(image, wrist_r_px, 6, color_r, -1)  # Wrist joint
        
        # Note: Status panel is now drawn separately in draw_improved_status_panel()
    
    def draw_angle_meter(self, image, angle, target_range, position, label):
        """Draw an angle meter with color coding."""
        x, y = position
        width, height = 100, 20
        
        # Background
        cv2.rectangle(image, (x, y), (x + width, y + height), (50, 50, 50), -1)
        
        # Determine color
        if target_range[0] <= angle <= target_range[1]:
            color = (0, 255, 0)  # Green
        elif abs(angle - target_range[0]) < 20 or abs(angle - target_range[1]) < 20:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        # Fill based on angle (normalize to 0-180)
        fill_width = int((angle / 180.0) * width)
        cv2.rectangle(image, (x, y), (x + fill_width, y + height), color, -1)
        
        # Text
        cv2.putText(image, f"{label}: {angle:.0f} deg", (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_enhanced_angle_meter(self, image, angle, target_range, position, label, is_synchronized):
        """Draw an enhanced angle meter with synchronization status."""
        x, y = position
        width, height = 120, 15
        
        # Background
        cv2.rectangle(image, (x, y), (x + width, y + height), (30, 30, 30), -1)
        
        # Determine color with sync consideration
        if target_range[0] <= angle <= target_range[1]:
            color = (0, 255, 0) if is_synchronized else (0, 200, 0)  # Green variants
        elif abs(angle - target_range[0]) < 20 or abs(angle - target_range[1]) < 20:
            color = (0, 255, 255) if is_synchronized else (0, 200, 200)  # Yellow variants
        else:
            color = (0, 0, 255) if is_synchronized else (0, 0, 200)  # Red variants
        
        # Fill based on angle (normalize to 0-180)
        fill_width = int((angle / 180.0) * width)
        cv2.rectangle(image, (x, y), (x + fill_width, y + height), color, -1)
        
        # Target range indicator
        target_start = int((target_range[0] / 180.0) * width)
        target_end = int((target_range[1] / 180.0) * width)
        cv2.rectangle(image, (x + target_start, y - 2), (x + target_end, y + height + 2), (255, 255, 255), 1)
        
        # Sync indicator
        sync_symbol = "OK" if is_synchronized else "X"
        sync_color = (0, 255, 0) if is_synchronized else (0, 0, 255)
        
        # Text with sync status
        cv2.putText(image, f"{label}: {angle:.0f} deg {sync_symbol}", (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, sync_color, 1)
    
    def draw_enhanced_spine_indicator(self, image, spine_deviation, position, state):
        """Draw enhanced spine alignment indicator with violation tracking."""
        x, y = position
        
        # Get spine violation count for severity assessment
        spine_violations = state.get("spine_violations", 0)
        
        # Determine color and size based on spine deviation and violations
        if abs(spine_deviation) <= 5:
            color = (0, 255, 0)  # Green
            radius = 15
            status = "GOOD"
        elif abs(spine_deviation) <= 10:
            color = (0, 255, 255)  # Yellow
            radius = 18
            status = "CAUTION"
        elif abs(spine_deviation) <= 15:
            color = (0, 165, 255)  # Orange
            radius = 20
            status = "WARNING"
        else:
            color = (0, 0, 255)  # Red
            radius = 22
            status = "CRITICAL"
        
        # Pulse effect for violations
        if spine_violations > 0:
            import time
            pulse = int(abs(np.sin(time.time() * 5)) * 10)
            radius += pulse
        
        # Draw spine indicator with border
        cv2.circle(image, (x, y), radius + 2, (255, 255, 255), 2)  # White border
        cv2.circle(image, (x, y), radius, color, -1)
        
        # Status text
        cv2.putText(image, status, (x - 25, y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(image, f"Spine: {spine_deviation:.1f} deg", (x - 40, y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Violation counter
        if spine_violations > 0:
            cv2.putText(image, f"Violations: {spine_violations}", (x - 50, y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def draw_sync_indicator(self, image, angle_l, angle_r, position):
        """Draw arm synchronization status indicator."""
        x, y = position
        
        angle_diff = abs(angle_l - angle_r)
        is_synchronized = angle_diff <= self.SYNC_TOLERANCE
        
        # Color and symbol based on synchronization
        if is_synchronized:
            color = (0, 255, 0)
            symbol = "OK"
            status = "SYNC"
        elif angle_diff <= self.SYNC_TOLERANCE * 1.5:
            color = (0, 255, 255)
            symbol = "~"
            status = "CLOSE"
        else:
            color = (0, 0, 255)
            symbol = "X"
            status = "ASYNC"
        
        # Draw sync status
        cv2.putText(image, f"{symbol} Arms {status} ({angle_diff:.0f} deg)", (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def draw_spine_indicator(self, image, spine_deviation, position):
        """Draw spine alignment indicator (legacy compatibility)."""
        x, y = position
        
        # Determine color based on spine deviation
        if abs(spine_deviation) <= 5:
            color = (0, 255, 0)  # Green
        elif abs(spine_deviation) <= 10:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        # Draw spine indicator
        cv2.circle(image, (x, y), 15, color, -1)
        cv2.putText(image, f"Spine: {spine_deviation:.1f} deg", (x - 40, y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def provide_detailed_setup_guidance(self, ready_status, state, current_time):
        """Provide step-by-step guidance for proper starting position."""
        guidance_messages = {
            "setup_instructions": [
                "Stand with feet shoulder-width apart",
                "Hold weights at shoulder level",
                "Keep elbows bent and close to body", 
                "Maintain straight back with engaged core",
                "Position hands directly above shoulders"
            ],
            "current_status": [],
            "corrections_needed": [],
            "audio_cues": []
        }
        
        # Check each requirement and provide specific feedback
        # TEMPORARILY DISABLED: hands_above requirement
        # if not ready_status["hands_above"]:
        #     guidance_messages["corrections_needed"].append("Raise your hands above shoulder level")
        #     if self.should_provide_audio("Lift the weights up to shoulder height", state, current_time, repeat_delay=8.0):
        #         guidance_messages["audio_cues"].append("Lift the weights up to shoulder height")
        
        if not ready_status["elbows_bent"]:
            guidance_messages["corrections_needed"].append("Bend your elbows more - bring weights closer to shoulders")
            if self.should_provide_audio("Keep your elbows bent and close to your body", state, current_time, repeat_delay=8.0):
                guidance_messages["audio_cues"].append("Keep your elbows bent and close to your body")
        
        if not ready_status["spine_straight"]:
            guidance_messages["corrections_needed"].append("Stand tall with straight back - engage your core")
            if self.should_provide_audio("Straighten your posture and tighten your core muscles", state, current_time, repeat_delay=8.0):
                guidance_messages["audio_cues"].append("Straighten your posture and tighten your core muscles")
        
        if not ready_status["arms_symmetric"]:
            guidance_messages["corrections_needed"].append("Keep both hands at the same height")
            if self.should_provide_audio("Level out both hands at shoulder height", state, current_time, repeat_delay=8.0):
                guidance_messages["audio_cues"].append("Level out both hands at shoulder height")
        
        if not ready_status["elbows_close"]:
            guidance_messages["corrections_needed"].append("Position hands directly above your shoulders")
            if self.should_provide_audio("Move the weights directly over your shoulders", state, current_time, repeat_delay=8.0):
                guidance_messages["audio_cues"].append("Move the weights directly over your shoulders")
        
        # Positive reinforcement when close (temporarily excluding hands_above)
        correct_checks = sum([ready_status["elbows_bent"], ready_status["spine_straight"], ready_status["arms_symmetric"], ready_status["elbows_close"]])
        total_checks = 4  # Reduced from 5 since we're bypassing hands_above
        
        if correct_checks == total_checks:
            guidance_messages["corrections_needed"].append("Perfect starting position! Get ready to begin...")
            if self.should_provide_audio("Excellent form! Preparing to start exercise", state, current_time, repeat_delay=10.0):
                guidance_messages["audio_cues"].append("Excellent form! Preparing to start exercise")
        elif correct_checks >= total_checks - 1:
            guidance_messages["corrections_needed"].append("Almost there! Small adjustment needed")
            if self.should_provide_audio("You're very close! Just minor adjustments needed", state, current_time, repeat_delay=8.0):
                guidance_messages["audio_cues"].append("You're very close! Just minor adjustments needed")
        
        # Provide audio feedback
        for audio_msg in guidance_messages["audio_cues"][:1]:  # Only one audio message at a time
            speak_async(audio_msg)
            state["last_audio_time"] = current_time
            state["last_audio_msg"] = audio_msg
            break
        
        return guidance_messages
    
    def display_setup_instructions(self, image, ready_status, current_time):
        """Display comprehensive setup instructions on the screen."""
        h, w = image.shape[:2]
        
        # Main title
        cv2.putText(image, "OVERHEAD PRESS STARTING POSITION:", (30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Step-by-step checklist with status indicators
        instructions = [
            ("1. Feet shoulder-width apart", True),  # Always assume this is correct
            ("2. Hold weights at shoulder level", True),
            ("3. Keep elbows bent (20-90 degrees)", ready_status["elbows_bent"]),
            ("4. Straight back, core engaged", ready_status["spine_straight"]),
            ("5. Hands above shoulders", ready_status["elbows_close"]),
            ("6. Both arms level", ready_status["arms_symmetric"])
        ]
        
        y_offset = 60
        for instruction, is_correct in instructions:
            # Color coding: green for correct, red for incorrect
            color = (0, 255, 0) if is_correct else (0, 0, 255)
            symbol = "OK" if is_correct else "X"
            
            # Display instruction with status
            cv2.putText(image, f"{symbol} {instruction}", (30, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
        
        # Overall readiness indicator
        all_ready = ready_status["ready"]
        readiness_color = (0, 255, 0) if all_ready else (255, 0, 0)
        readiness_text = "READY TO BEGIN!" if all_ready else "ADJUST POSITION"
        cv2.putText(image, readiness_text, (30, y_offset + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, readiness_color, 2)
    
    def display_status_info(self, image, state, feedback, current_time):
        """Display comprehensive status information."""
        h, w = image.shape[:2]
        
        # Main status display
        exercise_state = state.get("exercise_state", "waiting_for_ready")
        if exercise_state == "exercising":
            cv2.putText(image, f"Valid Reps: {state.get('valid_reps', 0)}", 
                       (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            if state.get("invalid_reps", 0) > 0:
                cv2.putText(image, f"Invalid: {state.get('invalid_reps', 0)}", 
                           (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Feedback messages
        y_offset = 250
        for i, msg in enumerate(feedback[:3]):  # Limit to 3 messages
            color = (0, 0, 255) if i == 0 else (255, 165, 0)  # Red for priority, orange for others
            cv2.putText(image, msg, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
        
        # Temporary success messages
        if current_time - state.get("message_timer", 0) < 3:
            message = state.get("last_message", "")
            if message:
                color = (0, 255, 0) if "Valid" in message else (0, 165, 255)
                cv2.putText(image, message, (30, y_offset + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


    def draw_improved_status_panel(self, image, angle_l, angle_r, spine_deviation, state):
        """Draw an improved status panel with better visibility and layout."""
        h, w = image.shape[:2]
        
        # Panel configuration
        panel_width = 220
        panel_height = 280
        panel_x = w - panel_width - 20
        panel_y = 20
        
        # Draw semi-transparent background panel
        overlay = image.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
        
        # Add border
        cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (200, 200, 200), 2)
        
        # Current Y position for drawing elements
        current_y = panel_y + 20
        
        # 1. SPINE INDICATOR (Larger circle with better visibility)
        spine_center_x = panel_x + 30
        spine_center_y = current_y + 25
        
        # Determine spine status and color
        if abs(spine_deviation) <= 5:
            spine_color = (0, 255, 0)  # Green
            spine_radius = 22  # Increased by 50%
            spine_status = "GOOD"
        elif abs(spine_deviation) <= 10:
            spine_color = (0, 255, 255)  # Yellow
            spine_radius = 25
            spine_status = "CAUTION"
        elif abs(spine_deviation) <= 15:
            spine_color = (0, 165, 255)  # Orange
            spine_radius = 28
            spine_status = "WARNING"
        else:
            spine_color = (0, 0, 255)  # Red
            spine_radius = 30
            spine_status = "CRITICAL"
        
        # Draw spine circle with border for better definition
        cv2.circle(image, (spine_center_x, spine_center_y), spine_radius + 3, (255, 255, 255), 3)  # White border
        cv2.circle(image, (spine_center_x, spine_center_y), spine_radius, spine_color, -1)
        
        # Spine status text (larger font)
        cv2.putText(image, spine_status, (spine_center_x + 40, spine_center_y + 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        current_y += 60
        
        # 2. SPINE DETAILS WITH PROGRESS BAR
        spine_text = f"Spine: {spine_deviation:.1f} deg"
        cv2.putText(image, spine_text, (panel_x + 10, current_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        current_y += 10
        
        # Spine progress bar (smaller height)
        spine_quality = max(0, min(1, 1 - (abs(spine_deviation) / 20)))
        self.draw_compact_progress_bar(image, panel_x + 10, current_y, 180, spine_quality, spine_color)
        
        current_y += 35
        
        # 3. LEFT ELBOW ANGLE
        left_text = f"Left Elbow: {angle_l:.0f} deg"
        cv2.putText(image, left_text, (panel_x + 10, current_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        current_y += 10
        
        # Left elbow progress bar
        exercise_state = state.get("exercise_state", "waiting_for_ready")
        if exercise_state == "exercising":
            target_range = self.TOP_RANGE if state.get("rep_phase") == "top_position" else self.STARTING_RANGE
        else:
            target_range = self.READY_ELBOW_RANGE
        
        left_quality = self.calculate_angle_quality(angle_l, target_range)
        left_color = self.get_quality_color(left_quality)
        self.draw_compact_progress_bar(image, panel_x + 10, current_y, 180, left_quality, left_color)
        
        current_y += 35
        
        # 4. RIGHT ELBOW ANGLE
        right_text = f"Right Elbow: {angle_r:.0f} deg"
        cv2.putText(image, right_text, (panel_x + 10, current_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        current_y += 10
        
        # Right elbow progress bar
        right_quality = self.calculate_angle_quality(angle_r, target_range)
        right_color = self.get_quality_color(right_quality)
        self.draw_compact_progress_bar(image, panel_x + 10, current_y, 180, right_quality, right_color)
        
        current_y += 35
        
        # 5. ARMS SYNCHRONIZATION (Previously missing!)
        angle_diff = abs(angle_l - angle_r)
        is_synchronized = angle_diff <= self.SYNC_TOLERANCE
        
        sync_text = f"Arms SYNC ({angle_diff:.0f} deg)"
        cv2.putText(image, sync_text, (panel_x + 10, current_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        current_y += 10
        
        # Arms sync progress bar (NEW!)
        sync_quality = max(0, min(1, 1 - (angle_diff / (self.SYNC_TOLERANCE * 2))))
        sync_color = (0, 255, 0) if is_synchronized else (0, 0, 255)
        self.draw_compact_progress_bar(image, panel_x + 10, current_y, 180, sync_quality, sync_color)
    
    def draw_compact_progress_bar(self, image, x, y, width, quality, color):
        """Draw a compact progress bar with 15px height and proper styling."""
        bar_height = 12  # Reduced height as requested
        
        # Background
        cv2.rectangle(image, (x, y), (x + width, y + bar_height), (60, 60, 60), -1)
        
        # Progress fill
        fill_width = int(width * quality)
        if fill_width > 0:
            cv2.rectangle(image, (x, y), (x + fill_width, y + bar_height), color, -1)
        
        # Border
        cv2.rectangle(image, (x, y), (x + width, y + bar_height), (150, 150, 150), 1)
        
        # Percentage text
        percentage_text = f"{quality:.0%}"
        cv2.putText(image, percentage_text, (x + width + 8, y + bar_height - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def calculate_angle_quality(self, angle, target_range):
        """Calculate quality score (0-1) for an angle based on target range."""
        if target_range[0] <= angle <= target_range[1]:
            return 1.0  # Perfect
        
        # Calculate distance from target range
        if angle < target_range[0]:
            distance = target_range[0] - angle
        else:
            distance = angle - target_range[1]
        
        # Convert to quality (closer = better)
        max_distance = 50  # Maximum reasonable distance
        quality = max(0, 1 - (distance / max_distance))
        return quality
    
    def get_quality_color(self, quality):
        """Get color based on quality score."""
        if quality >= 0.8:
            return (0, 255, 0)  # Green
        elif quality >= 0.6:
            return (0, 255, 255)  # Yellow
        elif quality >= 0.4:
            return (0, 165, 255)  # Orange
        else:
            return (0, 0, 255)  # Red


class PlankChecker(BaseExerciseChecker):
    """Form checker for plank exercises."""
    
    def init_state(self):
        return {
            "ready": False,
            "count": 0,
            "last_message": "",
            "message_timer": 0,
            "incorrect_start_time": None,
            "last_spoken_time": 0,
            "last_spoken_msg": "",
            "plank_duration_sec": 30,
            "plank_start_time": None,
        }
    
    def check_form(self, image, landmarks, state):
        """Analyze plank form."""
        feedback = []
        h, w = image.shape[:2]
        current_time = time.time()
        duration = state.get("plank_duration_sec", 30)

        # Get keypoints
        shoulder_l = self.get_point(landmarks, "SHOULDER", "LEFT")
        hip_l = self.get_point(landmarks, "HIP", "LEFT")
        ankle_l = self.get_point(landmarks, "ANKLE", "LEFT")
        ear_l = self.get_point(landmarks, "EAR", "LEFT")
        
        shoulder_r = self.get_point(landmarks, "SHOULDER", "RIGHT")
        hip_r = self.get_point(landmarks, "HIP", "RIGHT")
        ankle_r = self.get_point(landmarks, "ANKLE", "RIGHT")
        ear_r = self.get_point(landmarks, "EAR", "RIGHT")

        # Calculate angles
        body_angle_l = calculate_angle(shoulder_l, hip_l, ankle_l)
        body_angle_r = calculate_angle(shoulder_r, hip_r, ankle_r)
        neck_angle_l = calculate_angle(ear_l, shoulder_l, hip_l)
        neck_angle_r = calculate_angle(ear_r, shoulder_r, hip_r)

        # Draw angles
        draw_joint_angle(image, shoulder_l, hip_l, ankle_l, body_angle_l, 165, 180, label="Body (L):")
        draw_joint_angle(image, shoulder_r, hip_r, ankle_r, body_angle_r, 165, 180, label="Body (R):")
        draw_joint_angle(image, ear_l, shoulder_l, hip_l, neck_angle_l, 165, 195, label="Neck (L):")
        draw_joint_angle(image, ear_r, shoulder_r, hip_r, neck_angle_r, 165, 195, label="Neck (R):")

        # Form feedback
        if body_angle_l < 160 or body_angle_r < 160:
            feedback.append("Keep your hips up – don't let them sag")
        if body_angle_l > 185 or body_angle_r > 185:
            feedback.append("Lower your hips – keep a straight line")
        if abs(neck_angle_l - 180) > 15 or abs(neck_angle_r - 180) > 15:
            feedback.append("Keep your neck neutral – look down")

        # Posture check
        is_ready = (
            165 <= body_angle_l <= 185 and
            165 <= body_angle_r <= 185 and
            abs(neck_angle_l - 180) <= 15 and
            abs(neck_angle_r - 180) <= 15
        )

        # State transitions
        if not state.get("ready") and is_ready:
            state["ready"] = True
            state["plank_start_time"] = current_time
            state["last_message"] = "Plank started!"
            state["message_timer"] = current_time
        elif state.get("ready") and not is_ready:
            state["ready"] = False

        # Display status
        y_offset = 60
        cv2.putText(image, f"Plank: {'READY' if state['ready'] else 'NOT READY'}",
                    (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0) if is_ready else (0, 0, 255), 3)
        y_offset += 40

        for msg in feedback:
            cv2.putText(image, msg, (30, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30

        # Timer circle
        if state.get("ready"):
            elapsed = int(current_time - state["plank_start_time"])
            percentage = min(elapsed / duration, 1.0)
            angle = int(360 * percentage)

            if percentage < 0.8:
                color = (0, 255, 0)
            elif percentage < 1.0:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)

            center = (w - 100, 100)
            radius = 50
            thickness = 10
            cv2.ellipse(image, center, (radius, radius), -90, 0, angle, color, thickness)
            cv2.circle(image, center, radius - 15, (0, 0, 0), -1)
            cv2.putText(image, f"{elapsed}s", (center[0] - 20, center[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Temporary message display
        if current_time - state.get("message_timer", 0) < 3:
            cv2.putText(image, state.get("last_message", ""), (30, y_offset + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        # Audio feedback
        self.provide_audio_feedback(feedback, state, current_time)
        
        return feedback, 0

# Exercise checker factory and compatibility functions
_exercise_checkers = {
    "lunge": LungeChecker,
    "overhead_press": OverheadPressChecker,
    "plank": PlankChecker,
}


def get_exercise_checker(exercise_name):
    """Factory function to get the appropriate exercise checker."""
    if exercise_name not in _exercise_checkers:
        raise ValueError(f"Unknown exercise: {exercise_name}")
    return _exercise_checkers[exercise_name]()


def check_form(exercise_name, image, landmarks, state):
    """
    Dispatch the correct form-checking function based on the exercise name.
    This function maintains backward compatibility with the old functional interface.

    Args:
        exercise_name (str): one of ["lunge", "press", "plank"]
        image (np.ndarray): current video frame
        landmarks (list): pose landmarks from MediaPipe
        state (dict): exercise-specific state tracking

    Returns:
        feedback (list of str): textual suggestions/warnings
        reps_count (int): count of completed reps (or 0 for static poses like plank)
    """
    checker = get_exercise_checker(exercise_name)
    return checker.check_form(image, landmarks, state)


def init_state(exercise_name):
    """
    Initializes a per-exercise state dictionary to track posture status,
    rep count, messages, and timing. Each exercise may add custom fields.
    This function maintains backward compatibility with the old functional interface.

    Args:
        exercise_name (str): type of exercise (e.g., "press", "plank")

    Returns:
        dict: initialized state with default values
    """
    checker = get_exercise_checker(exercise_name)
    return checker.init_state()


# Legacy utility functions for backward compatibility
def check_feet_width(landmarks, threshold_ratio=0.2):
    """
    Checks if feet width is approximately the same as shoulder width.
    `threshold_ratio` is the allowed ±% difference from shoulder width.
    Returns (bool_ok, ratio feet_width/shoulder_width).
    """
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    shoulder_width = distance_2d(left_shoulder, right_shoulder)
    feet_width = distance_2d(left_ankle, right_ankle)

    lower_bound = shoulder_width * (1.0 - threshold_ratio)
    upper_bound = shoulder_width * (1.0 + threshold_ratio)
    ok = (feet_width >= lower_bound) and (feet_width <= upper_bound)

    return ok, feet_width / shoulder_width


def check_back_angle(landmarks):
    """Returns angle at the hip formed by (shoulder -> hip -> knee)."""
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

    return calculate_angle(left_shoulder, left_hip, left_knee)


def check_neck_angle(landmarks):
    """Returns neck angle formed by (ear -> shoulder -> hip)."""
    left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

    return calculate_angle(left_ear, left_shoulder, left_hip)


def check_knee_angle(landmarks):
    """Returns the angle at the knee formed by (hip -> knee -> ankle)."""
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

    return calculate_angle(left_hip, left_knee, left_ankle)


def check_hip_angle(landmarks):
    """Returns the angle at the hip formed by (shoulder -> hip -> knee)."""
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

    return calculate_angle(left_shoulder, left_hip, left_knee)


def is_down_position(landmarks):
    """Check if the user is in the bottom of the squat."""
    angle_knee = check_knee_angle(landmarks)
    return angle_knee < 140

