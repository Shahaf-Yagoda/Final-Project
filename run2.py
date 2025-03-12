import cv2
import mediapipe as mp
import numpy as np

###############################
#  1) Setup MediaPipe Pose    #
###############################
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


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


###############################
#  4) Main Checking Function  #
###############################
def check_squat_form(image, landmarks):
    # 1) Feet width
    feet_ok, feet_ratio = check_feet_width(landmarks, threshold_ratio=0.2)
    cv2.putText(image,
                f"Feet ratio: {feet_ratio:.2f} => {'OK' if feet_ok else 'NOT OK'}",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if feet_ok else (0, 0, 255), 2)

    # 2) Back angle (expect ~135–150)
    back_angle = check_back_angle(landmarks)
    back_ok = 135 <= back_angle <= 150
    cv2.putText(image,
                f"Back angle: {back_angle:.1f} => {'OK' if back_ok else 'NOT OK'}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if back_ok else (0, 0, 255), 2)

    # 3) Neck angle (170–180)
    neck_angle = check_neck_angle(landmarks)
    neck_ok = 170 <= neck_angle <= 180
    cv2.putText(image,
                f"Neck angle: {neck_angle:.1f} => {'OK' if neck_ok else 'NOT OK'}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if neck_ok else (0, 0, 255), 2)

    # 4) Check knee & hip angles only if "down"
    if is_down_position(landmarks):
        knee_angle = check_knee_angle(landmarks)
        knee_ok = 100 <= knee_angle <= 120  # "ideal range" in the bottom
        cv2.putText(image,
                    f"Knee angle: {knee_angle:.1f} => {'OK' if knee_ok else 'NOT OK'}",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if knee_ok else (0, 0, 255), 2)

        hip_angle = check_hip_angle(landmarks)
        hip_ok = 90 <= hip_angle <= 110  # "ideal range" in the bottom
        cv2.putText(image,
                    f"Hip angle: {hip_angle:.1f} => {'OK' if hip_ok else 'NOT OK'}",
                    (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if hip_ok else (0, 0, 255), 2)
    else:
        cv2.putText(image, "Not in down position", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


###############################
#  5) Main Entry Point        #
###############################
def main():
    cap = cv2.VideoCapture(0)
    # Initialize MediaPipe Pose
    with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame from camera.")
                break

            # Convert the BGR image to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # If landmarks are found, run checks
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Visualize landmarks (optional)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Perform squat form checks and annotate
                check_squat_form(frame, landmarks)

            cv2.imshow('Squat Tracker', frame)

            # Press 'ESC' to exit
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
