from flask import Flask, Response, request
import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os
from datetime import datetime
from dotenv import load_dotenv
import sys
import os
# ✅ Add both project root and src to sys.path
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
src_path = os.path.join(project_root, "src")

sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

# ✅ Now you can import correctly
from main import (
    calculate_angle, check_overhead_press_form, check_lunge_form,
    check_plank_form, save_keypoints_to_db
)

from database.database_connection import get_connection


###############################
# Flask + Camera Setup
###############################
app = Flask(__name__)
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)


###############################
# Stream Generator
###############################
def generate_frames(exercise="press"):
    # State for rep counting logic
    state = {
        "ready": False,
        "direction": None,
        "count": 0,
        "last_message": "",
        "message_timer": 0,
        "feedback": []
    }

    while True:
        success, frame = camera.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Perform the appropriate form check
            if exercise == "press":
                check_overhead_press_form(frame, landmarks, state)
            elif exercise == "lunge":
                check_lunge_form(frame, landmarks)
            elif exercise == "plank":
                check_plank_form(frame, landmarks)

            # Save keypoints to DB
            keypoints_data = [{
                "x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility
            } for lm in landmarks]

            save_keypoints_to_db(keypoints_data, workout_id=1, workout_name=exercise)

        # Add exit hint
        cv2.putText(frame, "Live Streaming... Close tab to stop", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Encode and yield the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


###############################
# Flask Route
###############################
@app.route('/')
def video_feed():
    exercise = request.args.get("exercise", "press")  # default is "press"
    return Response(generate_frames(exercise),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


###############################
# Run Server
###############################
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
