import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(point1, point2, point3):
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

# Initialize MediaPipe Pose.
pose = mp_pose.Pose()

# For webcam input:
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect pose.
    results = pose.process(image)

    # Convert back to BGR for OpenCV.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw the pose annotation on the image.
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Extract landmarks.
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates for left shoulder, elbow, and wrist.
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # Get coordinates for right shoulder, hip, and knee.
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

        # Calculate angles for posture correction.
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        back_angle = calculate_angle(right_shoulder, right_hip, right_knee)

        # Display angles on the image.
        cv2.putText(image, f'Left Elbow: {int(left_elbow_angle)} deg',
                    tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, f'Back: {int(back_angle)} deg',
                    tuple(np.multiply(right_hip, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Feedback conditions.
        if back_angle < 160:
            cv2.putText(image, 'Back not straight', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw lines between keypoints for visualization.
        cv2.line(image, tuple(np.multiply(left_shoulder, [640, 480]).astype(int)),
                 tuple(np.multiply(left_elbow, [640, 480]).astype(int)), (255, 0, 0), 2)
        cv2.line(image, tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                 tuple(np.multiply(left_wrist, [640, 480]).astype(int)), (255, 0, 0), 2)

        cv2.line(image, tuple(np.multiply(right_shoulder, [640, 480]).astype(int)),
                 tuple(np.multiply(right_hip, [640, 480]).astype(int)), (255, 0, 0), 2)
        cv2.line(image, tuple(np.multiply(right_hip, [640, 480]).astype(int)),
                 tuple(np.multiply(right_knee, [640, 480]).astype(int)), (255, 0, 0), 2)

    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
