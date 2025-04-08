import cv2
import mediapipe as mp
import numpy as np
import time
from src.database.database_connection import get_connection
import json
from datetime import datetime
import os
from dotenv import load_dotenv


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
def check_lunge_form(image, landmarks):
    # קדמי
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

    knee_angle = calculate_angle(hip, knee, ankle)
    knee_ok = 80 <= knee_angle <= 100

    cv2.putText(image,
                f"Knee angle: {knee_angle:.1f} => {'OK' if knee_ok else 'NOT OK'}",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if knee_ok else (0, 0, 255), 3)

    # גב זקוף
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    back_angle = calculate_angle(shoulder, hip, knee)
    back_ok = back_angle > 160

    cv2.putText(image,
                f"Back angle: {back_angle:.1f} => {'OK' if back_ok else 'NOT OK'}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if back_ok else (0, 0, 255), 3)


def check_overhead_press_form(image, landmarks, state):
    """
    Enhanced overhead press form check that ensures the wrist is
    actually above the shoulder to count a rep.
    """
    h, w = image.shape[:2]
    current_time = time.time()
    feedback = []

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

    # -- PHASE DETECTION (for overhead press) based on elbow
    def get_phase(elbow_angle):
        if 85 <= elbow_angle <= 95:
            return "bottom"
        elif elbow_angle > 165:
            return "top"
        elif 95 < elbow_angle < 165:
            return "transition"
        else:
            return "invalid"

    phase_l = get_phase(elbow_angle_l)
    phase_r = get_phase(elbow_angle_r)

    # -- ARC COLOR BASED ON PHASE & back posture
    def get_arc_color(phase, back_ok=True):
        if phase in ["top", "bottom"]:
            return (0, 255, 0) if back_ok else (0, 0, 255)
        elif phase == "transition":
            return (0, 255, 255)  # yellowish
        else:
            return (0, 0, 255)    # invalid => red

    back_ok_l = back_angle_l >= 165
    back_ok_r = back_angle_r >= 165

    # -- DRAW JOINT ANGLES
    draw_joint_angle(
        image, shoulder_l, elbow_l, wrist_l, elbow_angle_l, 165, 180,
        label="Elbow (L): ", override_color=get_arc_color(phase_l, back_ok_l)
    )
    draw_joint_angle(
        image, shoulder_r, elbow_r, wrist_r, elbow_angle_r, 165, 180,
        label="Elbow (R): ", override_color=get_arc_color(phase_r, back_ok_r)
    )
    draw_joint_angle(
        image, shoulder_l, hip_l, knee_l, back_angle_l, 165, 180,
        label="Back (L): ", override_color=(0, 255, 0) if back_ok_l else (0, 0, 255)
    )
    draw_joint_angle(
        image, shoulder_r, hip_r, knee_r, back_angle_r, 165, 180,
        label="Back (R): ", override_color=(0, 255, 0) if back_ok_r else (0, 0, 255)
    )

    # -- ADDITIONAL FORM CHECKS
    if not back_ok_l or not back_ok_r:
        feedback.append("Arching lower back - Keep core tight")

    # Check if wrist is stacked over elbow (roughly in same x-range)
    if abs(wrist_l[0] - elbow_l[0]) > 0.05 or abs(wrist_r[0] - elbow_r[0]) > 0.05:
        feedback.append("Wrist not stacked directly above elbow")

    # Check if elbows are not flaring too far from the shoulder
    if abs(elbow_l[0] - shoulder_l[0]) > 0.15 or abs(elbow_r[0] - shoulder_r[0]) > 0.15:
        feedback.append("Elbows may be flaring - bring them slightly closer")

    # -- WRIST-ABOVE-SHOULDER CHECKS
    wrist_l_above_shoulder = is_wrist_above_shoulder(wrist_l, shoulder_l)
    wrist_r_above_shoulder = is_wrist_above_shoulder(wrist_r, shoulder_r)

    # If user tries to be in 'top' phase but wrist is not above shoulder, push a warning
    if phase_l == "top" and not wrist_l_above_shoulder:
        feedback.append("Left wrist not actually above shoulder in top position")
    if phase_r == "top" and not wrist_r_above_shoulder:
        feedback.append("Right wrist not actually above shoulder in top position")

    # -- PHASES + REP LOGIC
    # We only call it a valid "bottom" if the elbow angle is ~90 AND back is OK.
    # We only call it a valid "top" if elbow is extended + wrist is truly above the shoulder.
    ready_l = (phase_l == "bottom" and back_ok_l)
    ready_r = (phase_r == "bottom" and back_ok_r)

    extended_l = (phase_l == "top" and wrist_l_above_shoulder)
    extended_r = (phase_r == "top" and wrist_r_above_shoulder)

    # Mark user "ready" if they go to the bottom position with decent form
    if not state["ready"] and (ready_l or ready_r):
        state["ready"] = True
        state["last_message"] = "Ready position detected"
        state["message_timer"] = current_time

    # Once they're "ready," look for them to press upwards.
    # Only consider we moved "up" if we see a top position with the wrist(s) above the shoulder(s).
    if state["ready"] and state["direction"] != "up" and (extended_l or extended_r):
        state["direction"] = "up"

    # Then once they've gone "up," look for them to return to bottom to complete the rep.
    if state["direction"] == "up" and (ready_l or ready_r):
        state["direction"] = "down"
        state["count"] += 1
        state["last_message"] = f"Rep #{state['count']} completed"
        state["message_timer"] = current_time

    # -- DRAW REPS & FEEDBACK
    cv2.putText(
        image, f"Reps: {state['count']}", (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3
    )

    # Additional textual info (angles, etc.)
    y_offset = 100
    cv2.putText(image, f"Elbow L: {elbow_angle_l:.1f} deg", (30, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 25
    cv2.putText(image, f"Elbow R: {elbow_angle_r:.1f} deg", (30, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 25
    cv2.putText(image, f"Back L : {back_angle_l:.1f} deg", (30, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 25
    cv2.putText(image, f"Back R : {back_angle_r:.1f} deg", (30, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 35

    # Show feedback messages line by line
    for msg in feedback:
        cv2.putText(image, msg, (30, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30

    # Temporary on-screen message for recent events (ready/rep completion)
    if current_time - state.get("message_timer", 0) < 3:
        cv2.putText(
            image, state.get("last_message", ""), (30, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3
        )



def check_plank_form(image, landmarks):
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

    plank_angle = calculate_angle(shoulder, hip, ankle)
    plank_ok = 165 <= plank_angle <= 180

    cv2.putText(image,
                f"Plank Body Line: {plank_angle:.1f} => {'OK' if plank_ok else 'NOT OK'}",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if plank_ok else (0, 0, 255), 3)


def save_keypoints_to_db(keypoints_data, workout_id, workout_name):
    """
    Inserts keypoints into the 'keypoints' table in the appropriate database (local or cloud).
    Uses the USE_CLOUD_DB environment variable to determine the target DB.
    """
    conn = None
    cursor = None

    try:
        load_dotenv(override=True)
        use_cloud = os.getenv("USE_CLOUD_DB", "false").lower() == "true"  # ✅ convert to boolean

        conn = get_connection(use_cloud)  # ✅ now passing actual boolean
        cursor = conn.cursor()

        keypoints_json = json.dumps(keypoints_data)
        current_timestamp = datetime.now()

        insert_query = """
            INSERT INTO keypoints (workout_id, timestamp, keypoints, workout)
            VALUES (%s, %s, %s, %s);
        """
        cursor.execute(insert_query, (workout_id, current_timestamp, keypoints_json, workout_name))
        conn.commit()

        db_type = "Cloud" if use_cloud else "Local"
        print(f"✅ Keypoints saved successfully to {db_type} DB.")

    except Exception as e:
        print("❌ Error inserting keypoints into the database:", e)
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()



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
        "feedback": []
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
                    check_lunge_form(frame, landmarks)
                elif workout_name == "press":
                    check_overhead_press_form(frame, landmarks, state)
                elif workout_name == "plank":
                    check_plank_form(frame, landmarks)

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
    parser.add_argument("--exercise", type=str, default="press", help="Exercise to perform (press, lunge, plank)")
    args = parser.parse_args()

    main(exercise_name=args.exercise)


