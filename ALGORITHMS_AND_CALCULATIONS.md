# Right Motion - Algorithms and Calculations

## Overview

This document describes all the main algorithms and mathematical calculations used in the Right Motion fitness tracking system for pose analysis, form checking, and exercise evaluation.

---

## Core Mathematical Algorithms

### 1. **Angle Calculation Algorithm**

**Location**: `src/processing/utils.py:6-48`

**Purpose**: Calculate the angle at point 'b' formed by three points a, b, and c.

**Algorithm**:
```python
def calculate_angle(a, b, c):
    # Convert points to numpy arrays (x, y coordinates only)
    a = np.array(a[:2])
    b = np.array(b[:2])
    c = np.array(c[:2])

    # Calculate vectors from point b
    ba = a - b  # Vector from b to a
    bc = c - b  # Vector from b to c

    # Compute dot product and magnitudes
    dot = np.dot(ba, bc)
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)

    # Prevent division by zero
    if mag_ba * mag_bc == 0:
        return 0.0

    # Compute cosine of the angle
    cosine_angle = dot / (mag_ba * mag_bc)

    # Clamp value to valid range [-1, 1] to avoid floating-point errors
    cosine_angle = max(min(cosine_angle, 1.0), -1.0)

    # Return the angle in degrees
    angle = np.degrees(np.arccos(cosine_angle))
    return angle
```

**Mathematical Formula**:
- **Dot Product**: `dot = ba · bc = |ba| × |bc| × cos(θ)`
- **Angle**: `θ = arccos(dot / (|ba| × |bc|))`
- **Convert to degrees**: `angle_degrees = θ × (180/π)`

**Use Cases**:
- Joint angles (knee, elbow, ankle, hip)
- Body alignment angles (spine, neck)
- Form validation thresholds

---

### 2. **2D Distance Calculation**

**Location**: `src/processing/utils.py:81-88`

**Purpose**: Calculate Euclidean distance between two 2D points.

**Algorithm**:
```python
def distance_2d(a, b):
    # Convert to numpy arrays (x, y coordinates only)
    a = np.array(a[:2])
    b = np.array(b[:2])
    
    # Calculate Euclidean distance
    return np.linalg.norm(a - b)
```

**Mathematical Formula**:
- **Distance**: `d = √[(x₂-x₁)² + (y₂-y₁)²]`

**Use Cases**:
- Measuring body part separations
- Checking hand positioning relative to shoulders
- Validating stance width

---

### 3. **Spine Alignment Calculation**

**Location**: `src/processing/forms_check.py:266-300`

**Purpose**: Calculate spine deviation from vertical alignment.

**Algorithm**:
```python
def calculate_spine_alignment(landmarks):
    # Get center points of shoulders and hips
    shoulder_center = [(shoulder_l[0] + shoulder_r[0]) / 2, (shoulder_l[1] + shoulder_r[1]) / 2]
    hip_center = [(hip_l[0] + hip_r[0]) / 2, (hip_l[1] + hip_r[1]) / 2]
    
    # Calculate spine vector
    spine_vector = [shoulder_center[0] - hip_center[0], shoulder_center[1] - hip_center[1]]
    vertical_vector = [0, -1]  # Perfect vertical reference
    
    # Calculate angle between spine and vertical
    dot_product = spine_vector[0] * vertical_vector[0] + spine_vector[1] * vertical_vector[1]
    spine_magnitude = (spine_vector[0]**2 + spine_vector[1]**2)**0.5
    
    if spine_magnitude == 0:
        return 0
    
    cos_angle = dot_product / spine_magnitude
    cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
    spine_angle = np.arccos(cos_angle) * 180 / np.pi
    
    # Determine forward/backward lean direction
    if spine_vector[0] > 0:  # Leaning forward
        return spine_angle
    else:  # Leaning backward
        return -spine_angle
```

**Mathematical Components**:
- **Center Point**: `center = (point1 + point2) / 2`
- **Vector Calculation**: `spine_vector = shoulder_center - hip_center`
- **Dot Product**: `dot = spine_vector · vertical_vector`
- **Angle from Vertical**: `θ = arccos(dot / |spine_vector|)`

**Use Cases**:
- Posture validation in overhead press
- Back straightness in lunges
- Core stability assessment

---

## Exercise-Specific Algorithms

### 4. **Lunge Form Analysis Algorithm**

**Location**: `src/processing/forms_check.py:76-151`

**Key Calculations**:

#### **Front Leg Detection**:
```python
def determine_front_leg(landmarks):
    # Use z-depth to determine which leg is forward
    left_knee_z = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z
    right_knee_z = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z
    return "left" if left_knee_z < right_knee_z else "right"
```

#### **Angle Validations**:
- **Front Knee Angle**: `85° ≤ angle ≤ 110°` (lunge down detection)
- **Front Knee Angle**: `> 160°` (return to standing)
- **Back Knee Angle**: `90° ≤ angle ≤ 100°` (ideal range)
- **Torso Angle**: `≥ 165°` (back straightness)
- **Ankle Angle**: `20° ≤ angle ≤ 35°` (proper foot positioning)

#### **Rep Counting Logic**:
```python
# Rep counting state machine
if 85 <= front_knee_angle <= 110:
    if not state.get("ready", False):
        state["ready"] = True
        state["direction"] = "down"
elif front_knee_angle > 160:
    if state.get("ready") and state.get("direction") == "down":
        state["count"] += 1  # Complete rep
        state["ready"] = False
        state["direction"] = "up"
```

#### **Knee-Over-Toes Detection**:
```python
# Check if knee passes beyond toes
if knee_f[0] > foot_f[0]:  # Compare x-coordinates
    feedback.append(f"{front_leg.title()} knee passed toes")
```

---

### 5. **Overhead Press Analysis Algorithm**

**Location**: `src/processing/forms_check.py:154-1214`

**Key Calculations**:

#### **Angle Smoothing**:
```python
def get_smoothed_angle(angle, buffer, buffer_size=5):
    # 5-frame moving average for noise reduction
    buffer.append(angle)
    if len(buffer) > buffer_size:
        buffer.pop(0)
    return sum(buffer) / len(buffer)
```

#### **Ready Position Detection**:
```python
def is_ready_position_enhanced(landmarks):
    # Multiple criteria for ready position
    criteria = {
        "elbows_bent": 20 <= elbow_angle_l <= 90 and 20 <= elbow_angle_r <= 90,
        "hands_above": wrist_l[1] < shoulder_l[1] - 0.02 and wrist_r[1] < shoulder_r[1] - 0.02,
        "spine_straight": abs(spine_deviation) <= 10,
        "arms_level": abs(wrist_l[1] - wrist_r[1]) <= 0.05,
        "elbows_wide": elbow_shoulder_dist_l >= 0.08 and elbow_shoulder_dist_r >= 0.08
    }
    return criteria
```

#### **Range of Motion Validation**:
- **Starting Position**: `10° ≤ elbow_angle ≤ 100°`
- **Top Position**: `160° ≤ elbow_angle ≤ 180°`
- **Synchronization Tolerance**: `|left_angle - right_angle| ≤ 20°`

#### **Rep Validation Algorithm**:
```python
# Multi-criteria rep validation
rep_valid = (
    state["rep_full_range"] and      # Full range of motion achieved
    state["rep_sync_maintained"] and  # Arms stayed synchronized
    state["rep_spine_ok"]            # Spine alignment maintained
)

if rep_valid:
    state["valid_reps"] += 1
else:
    state["invalid_reps"] += 1
```

---

### 6. **Plank Duration and Stability Algorithm**

**Location**: `src/processing/forms_check.py:1350-1490`

**Key Calculations**:

#### **Body Alignment Angles**:
```python
# Calculate body line angles (shoulder-hip-ankle)
body_angle_l = calculate_angle(shoulder_l, hip_l, ankle_l)
body_angle_r = calculate_angle(shoulder_r, hip_r, ankle_r)

# Calculate neck alignment (ear-shoulder-hip)
neck_angle_l = calculate_angle(ear_l, shoulder_l, hip_l)
neck_angle_r = calculate_angle(ear_r, shoulder_r, hip_r)
```

#### **Posture Validation**:
```python
is_ready = (
    165 <= body_angle_l <= 185 and    # Left side alignment
    165 <= body_angle_r <= 185 and    # Right side alignment
    abs(neck_angle_l - 180) <= 15 and # Left neck neutral
    abs(neck_angle_r - 180) <= 15     # Right neck neutral
)
```

#### **Timer and Progress Calculation**:
```python
# Duration tracking
if state.get("ready"):
    elapsed = int(current_time - state["plank_start_time"])
    percentage = min(elapsed / duration, 1.0)
    angle = int(360 * percentage)  # For circular progress indicator
```

#### **Visual Progress Circle**:
```python
# Draw circular timer
center = (w - 100, 100)
radius = 50
cv2.ellipse(image, center, (radius, radius), -90, 0, angle, color, thickness)
```

---

## Feedback and Audio Algorithms

### 7. **Feedback Throttling Algorithm**

**Location**: `src/processing/forms_check.py:35-53`

**Purpose**: Prevent repetitive audio feedback spam.

**Algorithm**:
```python
def provide_audio_feedback(feedback, state, current_time, delay=1.0, cooldown=5.0):
    is_correct = (len(feedback) == 0)
    
    if not is_correct:
        # Start incorrect timer if not already started
        if state.get("incorrect_start_time") is None:
            state["incorrect_start_time"] = current_time
        else:
            elapsed = current_time - state["incorrect_start_time"]
            
            # Only provide feedback after delay period
            if elapsed > delay:
                for msg in feedback:
                    time_since_last = current_time - state.get("last_spoken_time", 0)
                    
                    # Check if message is different or cooldown period has passed
                    if msg != state.get("last_spoken_msg", "") or time_since_last > cooldown:
                        speak_async(msg)
                        state["last_spoken_msg"] = msg
                        state["last_spoken_time"] = current_time
                        break
    else:
        # Reset incorrect timer when form is correct
        state["incorrect_start_time"] = None
```

**Timing Parameters**:
- **Delay**: 1.0 second before first feedback
- **Cooldown**: 5.0 seconds between identical messages
- **Immediate Reset**: When form becomes correct

---

### 8. **Video Streaming Feedback Throttling**

**Location**: `src/app/video_streamer.py:94-106`

**Algorithm**:
```python
def should_throttle_feedback(user_id, message, current_time, throttle_seconds=2.0):
    if user_id not in self.feedback_throttle:
        self.feedback_throttle[user_id] = {}
    
    user_throttle = self.feedback_throttle[user_id]
    
    # Check if we've seen this exact message recently
    if message in user_throttle:
        time_since_last = current_time - user_throttle[message]
        return time_since_last < throttle_seconds
    
    return False
```

**Purpose**: Prevent duplicate feedback in database during real-time streaming.

---

## Visual Feedback Algorithms

### 9. **Joint Angle Visualization**

**Location**: `src/processing/utils.py:50-79`

**Algorithm**:
```python
def draw_joint_angle(image, a, b, c, angle, min_ok, max_ok, label="", override_color=None):
    # Convert normalized coordinates to pixel coordinates
    h, w = image.shape[:2]
    pt_a = (int(a[0] * w), int(a[1] * h))
    pt_b = (int(b[0] * w), int(b[1] * h))
    pt_c = (int(c[0] * w), int(c[1] * h))

    # Color coding based on angle range
    color = override_color if override_color else (
        (0, 255, 0) if min_ok <= angle <= max_ok else (0, 0, 255)
    )

    # Draw lines and angle text
    cv2.line(image, pt_a, pt_b, color, 4)
    cv2.line(image, pt_b, pt_c, color, 4)
    
    # Add text with background
    text = f"{label}{angle:.1f} deg"
    cv2.putText(image, text, (pt_b[0] + 5, pt_b[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
```

**Color Coding**:
- **Green (0, 255, 0)**: Angle within acceptable range
- **Red (0, 0, 255)**: Angle outside acceptable range

---

### 10. **Coordinate System Conversions**

**Coordinate Transformations**:
```python
# MediaPipe coordinates (normalized 0-1) to pixel coordinates
def normalized_to_pixel(normalized_point, image_width, image_height):
    pixel_x = int(normalized_point[0] * image_width)
    pixel_y = int(normalized_point[1] * image_height)
    return (pixel_x, pixel_y)

# Extract 2D coordinates from 3D landmarks
def get_point(landmarks, part, side):
    lm = getattr(mp_pose.PoseLandmark, f"{side.upper()}_{part.upper()}")
    return [landmarks[lm.value].x, landmarks[lm.value].y]
```

---

## State Machine Algorithms

### 11. **Exercise State Management**

**Overhead Press State Machine**:
```python
# State transitions: waiting_for_ready → countdown → exercising
if exercise_state == "waiting_for_ready":
    if ready_status["ready"]:
        state["ready_frames_count"] += 1
        if state["ready_frames_count"] >= READY_FRAMES_REQUIRED:
            state["exercise_state"] = "countdown"
            state["countdown_start_time"] = current_time

elif exercise_state == "countdown":
    countdown_elapsed = current_time - state["countdown_start_time"]
    if countdown_elapsed >= COUNTDOWN_DURATION:
        state["exercise_state"] = "exercising"
        state["exercise_start_time"] = current_time

elif exercise_state == "exercising":
    # Handle rep counting and form analysis
    process_exercise_phase()
```

### 12. **Rep Counting State Machine**

**Phase-based Rep Detection**:
```python
# Overhead Press: starting → midpoint → top → midpoint → starting
if rep_phase == "starting" and elbow_angle_avg >= 120:
    state["rep_phase"] = "midpoint"
elif rep_phase == "midpoint" and elbow_angle_avg >= 160:
    state["rep_phase"] = "top"
elif rep_phase == "top" and elbow_angle_avg <= 140:
    state["rep_phase"] = "midpoint_return"
elif rep_phase == "midpoint_return" and elbow_angle_avg <= 100:
    state["rep_phase"] = "starting"
    # Complete rep registered
```

---

## Performance Optimization Algorithms

### 13. **Moving Average Smoothing**

**Noise Reduction for Angle Measurements**:
```python
def get_smoothed_angle(angle, buffer, buffer_size=5):
    buffer.append(angle)
    if len(buffer) > buffer_size:
        buffer.pop(0)
    return sum(buffer) / len(buffer)
```

**Benefits**:
- Reduces jitter in angle measurements
- Improves stability of rep detection
- 5-frame window provides good balance of responsiveness and smoothness

---

## Summary of Key Mathematical Constants

### **Angle Thresholds**:

**Lunge Exercise**:
- Front knee (lunge detection): 85° - 110°
- Front knee (standing): > 160°
- Back knee (ideal): 90° - 100°
- Torso alignment: ≥ 165°
- Ankle angle: 20° - 35°

**Overhead Press**:
- Starting position: 10° - 100°
- Top position: 160° - 180°
- Synchronization tolerance: ± 20°
- Spine deviation: ± 10° (ready), ± 15° (exercising)

**Plank**:
- Body alignment: 165° - 185°
- Neck alignment: 180° ± 15°

### **Distance and Position Thresholds**:
- Hand positioning: 0.02 normalized units above shoulders
- Elbow distance: 0.08 normalized units from shoulders
- Arm level difference: ≤ 0.05 normalized units

### **Timing Parameters**:
- Audio feedback delay: 1.0 second
- Audio feedback cooldown: 5.0 seconds
- Video feedback throttle: 2.0 seconds
- Ready position frames: 30 frames
- Countdown duration: 5 seconds

These algorithms work together to provide comprehensive, real-time fitness form analysis with mathematical precision and user-friendly feedback systems.