import cv2
import mediapipe as mp
import numpy as np
import time
from src.database.database_connection import get_connection
import json
from datetime import datetime
import os
from dotenv import load_dotenv

from gtts import gTTS
from playsound import playsound
import os
import uuid
import threading
from threading import Lock

audio_lock = Lock()


def speak_async(text):
    threading.Thread(target=speak, args=(text,), daemon=True).start()


def speak(text):
    if not audio_lock.acquire(blocking=False):
        print(f"[VOICE] Skipping (audio already playing): {text}")
        return

    try:
        print(f"[VOICE] Playing: {text}")
        tts = gTTS(text=text, lang='en')
        filename = f"/tmp/{uuid.uuid4().hex}.mp3"
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
    except Exception as e:
        print("🔴 TTS Error:", e)
    finally:
        audio_lock.release()


###############################
#  1) Setup MediaPipe Pose    #
###############################
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Tracking states
start_position_ready = False
rep_count = 0
direction = None  # "up" or "down"


###############################
#  2) Utility Functions       #
###############################
def calculate_angle(a, b, c):
    """
    Calculate the angle at point 'b' formed by the line segments:
        A--B and B--C
    Accepts a, b, c as lists (or tuples) of [x, y] or [x, y, z].
    Returns angle in degrees.
    """
    a = np.array(a[:2])  # Only use x,y
    b = np.array(b[:2])
    c = np.array(c[:2])

    ba = a - b
    bc = c - b

    # Dot product
    dot = np.dot(ba, bc)
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)

    # Avoid division by zero
    if mag_ba * mag_bc == 0:
        return 0.0

    cosine_angle = dot / (mag_ba * mag_bc)
    # Clamp to avoid floating-point errors out of [-1, 1]
    cosine_angle = max(min(cosine_angle, 1.0), -1.0)

    angle = np.degrees(np.arccos(cosine_angle))
    return angle


def distance_2d(a, b):
    """
    Returns 2D distance between points A and B.
    Each input is [x, y] or [x, y, z].
    """
    a = np.array(a[:2])
    b = np.array(b[:2])
    return np.linalg.norm(a - b)


###############################
#  3) Form Checks             #
###############################
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

    # Check if feet_width is within ± `threshold_ratio` of shoulder_width
    lower_bound = shoulder_width * (1.0 - threshold_ratio)
    upper_bound = shoulder_width * (1.0 + threshold_ratio)
    ok = (feet_width >= lower_bound) and (feet_width <= upper_bound)

    return ok, feet_width / shoulder_width


def check_back_angle(landmarks):
    """
    Returns angle at the hip formed by (shoulder -> hip -> knee).
    Typical "torso tilt" from vertical is ~30–45 degrees forward,
    which translates to an angle of ~135–150 degrees if measuring
    the interior angle at the hip.
    """
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

    angle_left_hip = calculate_angle(left_shoulder, left_hip, left_knee)
    return angle_left_hip


def check_neck_angle(landmarks):
    """
    A rough measure for 'neutral neck' is the angle at the shoulder
    formed by (ear -> shoulder -> hip). If close to 180,
    the neck is relatively straight in line with the torso.
    """
    left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

    angle_shoulder = calculate_angle(left_ear, left_shoulder, left_hip)
    return angle_shoulder


def check_knee_angle(landmarks):
    """
    Returns the angle at the knee formed by (hip -> knee -> ankle).
    """
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

    angle_left_knee = calculate_angle(left_hip, left_knee, left_ankle)
    return angle_left_knee


def check_hip_angle(landmarks):
    """
    Returns the angle at the hip formed by (shoulder -> hip -> knee).
    """
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

    angle_left_hip = calculate_angle(left_shoulder, left_hip, left_knee)
    return angle_left_hip


def is_down_position(landmarks):
    """
    Example heuristic to decide if the user is in the bottom of the squat.
    We'll say if knee angle < 140 degrees => "down."
    """
    angle_knee = check_knee_angle(landmarks)
    return angle_knee < 140


def draw_joint_angle(image, a, b, c, angle, min_ok, max_ok, label="", override_color=None):
    """
    Draws lines (a–b, b–c) and labels the angle at point b.
    Allows override_color for custom arc color (e.g., yellow during transition).
    """
    h, w = image.shape[:2]
    pt_a = (int(a[0] * w), int(a[1] * h))
    pt_b = (int(b[0] * w), int(b[1] * h))
    pt_c = (int(c[0] * w), int(c[1] * h))

    # Determine color
    if override_color is not None:
        color = override_color
    else:
        color = (0, 255, 0) if min_ok <= angle <= max_ok else (0, 0, 255)

    # Draw lines
    cv2.line(image, pt_a, pt_b, color, 4)
    cv2.line(image, pt_b, pt_c, color, 4)

    # Label with angle
    text = f"{label}{angle:.1f}°"
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

    # Bubble background
    cv2.rectangle(image,
                  (pt_b[0], pt_b[1] - text_height - 10),
                  (pt_b[0] + text_width + 10, pt_b[1] + 5),
                  (0, 0, 0),  # black background
                  -1)

    # Draw text
    cv2.putText(image, text, (pt_b[0] + 5, pt_b[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


###############################
#  4) Main Checking Function  #
###############################
def check_lunge_form(image, landmarks, state):
    feedback = []
    h, w = image.shape[:2]
    current_time = time.time()

    # Raw landmark positions
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

    # Decide front leg based on depth
    front_leg = "left" if left_knee.z < right_knee.z else "right"
    back_leg = "right" if front_leg == "left" else "left"

    # Helper
    def get_point(part, side):
        lm = getattr(mp_pose.PoseLandmark, f"{side.upper()}_{part.upper()}")
        return [landmarks[lm.value].x, landmarks[lm.value].y]

    # Points
    hip_f = get_point("HIP", front_leg)
    knee_f = get_point("KNEE", front_leg)
    ankle_f = get_point("ANKLE", front_leg)
    foot_f = get_point("FOOT_INDEX", front_leg)
    shoulder_f = get_point("SHOULDER", front_leg)

    hip_b = get_point("HIP", back_leg)
    knee_b = get_point("KNEE", back_leg)
    ankle_b = get_point("ANKLE", back_leg)

    # Angles
    front_knee_angle = calculate_angle(hip_f, knee_f, ankle_f)
    back_knee_angle = calculate_angle(hip_b, knee_b, ankle_b)
    torso_angle = calculate_angle(shoulder_f, hip_f, knee_f)
    ankle_angle = calculate_angle(knee_f, ankle_f, foot_f)

    # --- Rep Logic ---
    # Consider "bottom" when front knee is bent deeply
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

    # --- Feedback Rules ---
    if not (90 <= front_knee_angle <= 110):
        feedback.append(f"{front_leg.title()} knee angle should be 90°–110°")
    if not (90 <= back_knee_angle <= 100):
        feedback.append(f"{back_leg.title()} knee too straight (90°–100° ideal)")
    if torso_angle < 165:
        feedback.append("Torso leaning forward (keep upright posture)")
    if ankle_angle < 20 or ankle_angle > 35:
        feedback.append("Ankle angle out of range (20°–30° ideal)")
    if knee_f[0] > foot_f[0]:
        feedback.append(f"{front_leg.title()} knee passed toes")

    # --- Drawing angles ---
    draw_joint_angle(image, hip_f, knee_f, ankle_f, front_knee_angle, 90, 110, label=f"{front_leg.title()} Knee:")
    draw_joint_angle(image, hip_b, knee_b, ankle_b, back_knee_angle, 90, 100, label=f"{back_leg.title()} Knee:")
    draw_joint_angle(image, shoulder_f, hip_f, knee_f, torso_angle, 165, 180, label="Torso:")
    draw_joint_angle(image, knee_f, ankle_f, foot_f, ankle_angle, 20, 30, label="Ankle:")

    # --- Draw rep count & messages ---
    cv2.putText(image, f"Reps: {state['count']}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

    y_offset = 100
    for msg in feedback:
        cv2.putText(image, msg, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30

    if current_time - state.get("message_timer", 0) < 3:
        cv2.putText(image, state.get("last_message", ""), (30, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    # --- Audio Feedback ---
    is_correct = (len(feedback) == 0)
    if not is_correct:
        if state.get("incorrect_start_time") is None:
            state["incorrect_start_time"] = current_time
        else:
            elapsed = current_time - state["incorrect_start_time"]
            if elapsed > 1.0:
                for msg in feedback:
                    time_since_last = current_time - state.get("last_spoken_time", 0)
                    cooldown = 5
                    if msg != state.get("last_spoken_msg", "") or time_since_last > cooldown:
                        speak_async(msg)
                        state["last_spoken_msg"] = msg
                        state["last_spoken_time"] = current_time
                        break
    else:
        state["incorrect_start_time"] = None


def check_overhead_press_form(image, landmarks, state):
    """
    Enhanced overhead press form check that ensures the wrist is
    actually above the shoulder to count a rep.
    """
    feedback = []
    h, w = image.shape[:2]
    current_time = time.time()

    # feedback.append("TEST: You should hear this.")
    # if not state.get("alert_given", False):
    #     speak("This is a test voice message")
    #     state["alert_given"] = True

    # Helper to check if wrist is above shoulder
    def is_wrist_above_shoulder(wrist, shoulder):
        # 'Above' in an image means the wrist y is LESS than the shoulder y
        return wrist[1] < shoulder[1]

    # -- LEFT SIDE KEY POINTS
    shoulder_l = [
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    ]
    elbow_l = [
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
    ]
    wrist_l = [
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
    ]
    hip_l = [
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    ]
    knee_l = [
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    ]

    # -- RIGHT SIDE KEY POINTS
    shoulder_r = [
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    ]
    elbow_r = [
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
    ]
    wrist_r = [
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
    ]
    hip_r = [
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
    ]
    knee_r = [
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
    ]

    # -- ANGLES
    elbow_angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
    elbow_angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
    back_angle_l = calculate_angle(shoulder_l, hip_l, knee_l)
    back_angle_r = calculate_angle(shoulder_r, hip_r, knee_r)

    # Back form
    back_ok_l = back_angle_l >= 165
    back_ok_r = back_angle_r >= 165

    # Visual joint lines
    draw_joint_angle(image, shoulder_l, elbow_l, wrist_l, elbow_angle_l, 165, 180, label="Elbow (L): ")
    draw_joint_angle(image, shoulder_r, elbow_r, wrist_r, elbow_angle_r, 165, 180, label="Elbow (R): ")
    draw_joint_angle(image, shoulder_l, hip_l, knee_l, back_angle_l, 165, 180, label="Back (L): ")
    draw_joint_angle(image, shoulder_r, hip_r, knee_r, back_angle_r, 165, 180, label="Back (R): ")

    # Force posture to be considered wrong
    # back_ok_l = False   
    # back_ok_r = False

    # Form feedback collection
    if not back_ok_l or not back_ok_r:
        feedback.append("Arching lower back - Keep core tight")
        print("Feedback collected:", feedback)

    if abs(wrist_l[0] - elbow_l[0]) > 0.05 or abs(wrist_r[0] - elbow_r[0]) > 0.05:
        feedback.append("Wrist not stacked directly above elbow")

    if abs(elbow_l[0] - shoulder_l[0]) > 0.15 or abs(elbow_r[0] - shoulder_r[0]) > 0.15:
        feedback.append("Elbows may be flaring - bring them slightly closer")

    wrist_l_above = is_wrist_above_shoulder(wrist_l, shoulder_l)
    wrist_r_above = is_wrist_above_shoulder(wrist_r, shoulder_r)

    # Phase detection
    def get_phase(angle):
        if 85 <= angle <= 95:
            return "bottom"
        elif angle > 165:
            return "top"
        elif 95 < angle < 165:
            return "transition"
        else:
            return "invalid"

    phase_l = get_phase(elbow_angle_l)
    phase_r = get_phase(elbow_angle_r)

    ready_l = (phase_l == "bottom" and back_ok_l)
    ready_r = (phase_r == "bottom" and back_ok_r)
    extended_l = (phase_l == "top" and wrist_l_above)
    extended_r = (phase_r == "top" and wrist_r_above)

    # Rep logic
    if not state["ready"] and (ready_l or ready_r):
        state["ready"] = True
        state["last_message"] = "Ready position detected"
        state["message_timer"] = current_time

    if state["ready"] and state["direction"] != "up" and (extended_l or extended_r):
        state["direction"] = "up"

    if state["direction"] == "up" and (ready_l or ready_r):
        state["direction"] = "down"
        state["count"] += 1
        state["last_message"] = f"Rep #{state['count']} completed"
        state["message_timer"] = current_time

    # Display rep count
    cv2.putText(image, f"Reps: {state['count']}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

    # Display live angles
    y_offset = 100
    cv2.putText(image, f"Elbow L: {elbow_angle_l:.1f} deg", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    y_offset += 25
    cv2.putText(image, f"Elbow R: {elbow_angle_r:.1f} deg", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    y_offset += 25
    cv2.putText(image, f"Back L : {back_angle_l:.1f} deg", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    y_offset += 25
    cv2.putText(image, f"Back R : {back_angle_r:.1f} deg", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    y_offset += 35

    # Draw feedback messages
    for msg in feedback:
        cv2.putText(image, msg, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30

    # Temporary on-screen message
    if current_time - state.get("message_timer", 0) < 3:
        cv2.putText(image, state.get("last_message", ""), (30, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    # 🔊 Real-time audio feedback after bad form > 1 sec
    is_correct = (len(feedback) == 0)

    if not is_correct:
        if state.get("incorrect_start_time") is None:
            state["incorrect_start_time"] = current_time
        else:
            elapsed = current_time - state["incorrect_start_time"]

            if elapsed > 1.0:
                for msg in feedback:
                    time_since_last = current_time - state.get("last_spoken_time", 0)
                    cooldown = 5  # seconds between repeating the same message

                    if msg != state.get("last_spoken_msg", "") or time_since_last > cooldown:
                        speak_async(msg)
                        state["last_spoken_msg"] = msg
                        state["last_spoken_time"] = current_time
                        break  # Speak only one message per cycle

    else:
        # Reset so next time we can alert again
        state["incorrect_start_time"] = None
        # Keep last_spoken_msg & last_spoken_time so cooldown works


def check_plank_form(image, landmarks, state):
    feedback = []
    h, w = image.shape[:2]
    current_time = time.time()

    DURATION = state.get("plank_duration_sec", 30)  # ברירת מחדל 30 שניות

    def get_point(part, side):
        lm = getattr(mp_pose.PoseLandmark, f"{side.upper()}_{part.upper()}")
        return [landmarks[lm.value].x, landmarks[lm.value].y]

    # נקודות רלוונטיות
    shoulder_l = get_point("SHOULDER", "LEFT")
    hip_l = get_point("HIP", "LEFT")
    ankle_l = get_point("ANKLE", "LEFT")
    ear_l = get_point("EAR", "LEFT")

    shoulder_r = get_point("SHOULDER", "RIGHT")
    hip_r = get_point("HIP", "RIGHT")
    ankle_r = get_point("ANKLE", "RIGHT")
    ear_r = get_point("EAR", "RIGHT")

    # חישוב זוויות
    body_angle_l = calculate_angle(shoulder_l, hip_l, ankle_l)
    body_angle_r = calculate_angle(shoulder_r, hip_r, ankle_r)
    neck_angle_l = calculate_angle(ear_l, shoulder_l, hip_l)
    neck_angle_r = calculate_angle(ear_r, shoulder_r, hip_r)

    # ציור זוויות
    draw_joint_angle(image, shoulder_l, hip_l, ankle_l, body_angle_l, 165, 180, label="Body (L):")
    draw_joint_angle(image, shoulder_r, hip_r, ankle_r, body_angle_r, 165, 180, label="Body (R):")
    draw_joint_angle(image, ear_l, shoulder_l, hip_l, neck_angle_l, 165, 195, label="Neck (L):")
    draw_joint_angle(image, ear_r, shoulder_r, hip_r, neck_angle_r, 165, 195, label="Neck (R):")

    # בדיקות
    if body_angle_l < 160 or body_angle_r < 160:
        feedback.append("Keep your hips up – don't let them sag")
    if body_angle_l > 185 or body_angle_r > 185:
        feedback.append("Lower your hips – keep a straight line")
    if abs(neck_angle_l - 180) > 15 or abs(neck_angle_r - 180) > 15:
        feedback.append("Keep your neck neutral – look down")

    # תנוחה נכונה
    is_ready = (
        165 <= body_angle_l <= 185 and
        165 <= body_angle_r <= 185 and
        abs(neck_angle_l - 180) <= 15 and
        abs(neck_angle_r - 180) <= 15
    )

    # התחלה
    if not state.get("ready") and is_ready:
        state["ready"] = True
        state["plank_start_time"] = current_time
        state["last_message"] = "Plank started!"
        state["message_timer"] = current_time

    elif state.get("ready") and not is_ready:
        state["ready"] = False

    # ציור טקסט
    y_offset = 60
    cv2.putText(image, f"Plank: {'READY' if state['ready'] else 'NOT READY'}", (30, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if is_ready else (0, 0, 255), 3)
    y_offset += 40

    for msg in feedback:
        cv2.putText(image, msg, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30

    # טיימר גרפי
    if state["ready"]:
        elapsed = int(current_time - state["plank_start_time"])
        percentage = min(elapsed / DURATION, 1.0)
        angle = int(360 * percentage)

        # צבעים
        if percentage < 0.8:
            color = (0, 255, 0)       # ירוק
        elif percentage < 1.0:
            color = (0, 165, 255)     # כתום
        else:
            color = (0, 0, 255)       # אדום

        # ציור מעגל (progress circle)
        center = (w - 100, 100)
        radius = 50
        thickness = 10
        cv2.ellipse(image, center, (radius, radius), -90, 0, angle, color, thickness)
        cv2.circle(image, center, radius - 15, (0, 0, 0), -1)
        cv2.putText(image, f"{elapsed}s", (center[0] - 20, center[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # הודעה זמנית
    if current_time - state.get("message_timer", 0) < 3:
        cv2.putText(image, state.get("last_message", ""), (30, y_offset + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    # משוב קולי
    is_correct = len(feedback) == 0
    if not is_correct:
        if state.get("incorrect_start_time") is None:
            state["incorrect_start_time"] = current_time
        else:
            elapsed_err = current_time - state["incorrect_start_time"]
            if elapsed_err > 1.0:
                for msg in feedback:
                    time_since_last = current_time - state.get("last_spoken_time", 0)
                    cooldown = 5
                    if msg != state.get("last_spoken_msg", "") or time_since_last > cooldown:
                        speak_async(msg)
                        state["last_spoken_msg"] = msg
                        state["last_spoken_time"] = current_time
                        break
    else:
        state["incorrect_start_time"] = None


def save_keypoints_to_db(keypoints_data, workout_id, workout_name):
    pass
    # """
    # Inserts keypoints into the 'keypoints' table in the appropriate database (local or cloud).
    # Uses the USE_CLOUD_DB environment variable to determine the target DB.
    # """
    # conn = None
    # cursor = None

    # try:
    #     load_dotenv(override=True)
    #     use_cloud = os.getenv("USE_CLOUD_DB", "false").lower() == "true"  # ✅ convert to boolean

    #     conn = get_connection(use_cloud)  # ✅ now passing actual boolean
    #     cursor = conn.cursor()

    #     keypoints_json = json.dumps(keypoints_data)
    #     current_timestamp = datetime.now()

    #     insert_query = """
    #         INSERT INTO keypoints (workout_id, timestamp, keypoints, workout)
    #         VALUES (%s, %s, %s, %s);
    #     """
    #     cursor.execute(insert_query, (workout_id, current_timestamp, keypoints_json, workout_name))
    #     conn.commit()

    #     db_type = "Cloud" if use_cloud else "Local"
    #     print(f"✅ Keypoints saved successfully to {db_type} DB.")

    # except Exception as e:
    #     print("❌ Error inserting keypoints into the database:", e)
    #     if conn:
    #         conn.rollback()
    # finally:
    #     if cursor:
    #         cursor.close()
    #     if conn:
    #         conn.close()


###############################
#  5) Main Entry Point        #
###############################
def main(exercise_name):
    state = {
        "ready": False,
        "direction": None,
        "count": 0,
        "last_message": "",
        "message_timer": 0,
        "incorrect_start_time": None,
        "alert_given": False,
        "last_spoken_msg": "",
        "last_spoken_time": 0,
        "plank_start_time": 0,
        "plank_duration_sec": 30  # לדוגמה

    }

    cap = cv2.VideoCapture(0)
    workout_name = exercise_name  # 👈 "lunge" / "press" / "plank" by exercise
    with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

        while True:
            success, frame = cap.read()
            if not success:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # choose excercise 
                if workout_name == "lunge":
                    check_lunge_form(frame, landmarks, state)
                elif workout_name == "press":
                    check_overhead_press_form(frame, landmarks, state)
                elif workout_name == "plank":
                    check_plank_form(frame, landmarks, state)

                # save to database
                keypoints_data = [{
                    "x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility
                } for lm in landmarks]
                save_keypoints_to_db(keypoints_data, workout_id=1, workout_name=workout_name)

            cv2.imshow('Pose Tracker', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exercise", type=str, default="plank", help="Exercise to perform (press, lunge, plank)")
    args = parser.parse_args()

    main(exercise_name=args.exercise)
