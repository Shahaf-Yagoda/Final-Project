# gui/app.py
import streamlit as st
import sys
import os
import cv2
import numpy as np
import tempfile
import time
import subprocess
from datetime import datetime
import subprocess
subprocess.Popen(["python", "video_streamer.py"])


# Import custom modules
sys.path.insert(0, '../..')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.database.users.register import register_user
from src.database.users.login import login_user
from main import *  # if you're using logic from main

from main import *
import mediapipe as mp



# Session state to track navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"

def set_page(page_name):
    st.session_state.page = page_name

# Main page
if st.session_state.page == "Home":
    st.title("Welcome to Right Motion")
    st.subheader("Choose an option:")

    col1, col2 , col3= st.columns(3)
    with col1:
        st.button("🔐 Register", on_click=set_page, args=("Register",))
        st.button("📹 Analyze Video", on_click=set_page, args=("Analyze",))
    with col2:
        st.button("🔑 Log In", on_click=set_page, args=("Login",))
        st.button("🧩 TBD", on_click=set_page, args=("TBD",))
    with col3:
        st.button("🏃 Live Exercise", on_click=set_page, args=("LiveExercise",))
        st.button("🏃 TBD", on_click=set_page, args=("TBD",))

# Register Page
elif st.session_state.page == "Register":
    st.title("User Registration")

    name = st.text_input("Name")
    email = st.text_input("Email")
    dob = st.date_input("Date of Birth")
    role = st.selectbox("Role", ["user", "admin", "coach"])
    parent = st.text_input("Parent ID (optional)")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Register"):
        if not username or not password:
            st.error("Username and password are required.")
        else:
            parent_id = int(parent) if parent else None
            result = register_user(name, email, dob, role, parent_id, username, password)

            if isinstance(result, int):
                st.success(f"User registered with ID: {result}")
            else:
                st.error(result)

    st.button("⬅️ Back to Home", on_click=set_page, args=("Home",))


# Analyze Video Page (your original logic)
elif st.session_state.page == "Analyze":
    st.title("Right Motion Video Analyzer")
    st.header("Upload a video to analyze")
    video_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])

    st.header("Start Pose Estimation")
    if st.button("Start"):
        if video_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_file.read())
                video_path = tmp.name

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Error: Unable to open video file")

            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                if results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    # Initialize state
                if "press_state" not in st.session_state:
                    st.session_state.press_state = {
                        "ready": False,
                        "direction": None,
                        "count": 0,
                        "last_message": "",
                        "message_timer": 0,
                        "feedback": []
                    }

                check_overhead_press_form(frame, results.pose_landmarks.landmark, st.session_state.press_state)

                out.write(frame)

            cap.release()
            out.release()
            cv2.destroyAllWindows()

            time.sleep(1)
            converted_output_path = output_path.replace(".mp4", "_converted.mp4")
            subprocess.run(f"ffmpeg -i {output_path} -vcodec libx264 -crf 23 {converted_output_path}", shell=True, check=True)

            st.video(converted_output_path)

    st.button("⬅️ Back to Home", on_click=set_page, args=("Home",))

# Placeholder pages
elif st.session_state.page == "Login":
    st.title("User Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Log In"):
        user_id, error = login_user(username, password)

        if user_id:
            st.success(f"Login successful! Welcome, user #{user_id}.")
        else:
            st.error(error)

    st.button("⬅️ Back to Home", on_click=set_page, args=("Home",))

elif st.session_state.page == "TBD":
    st.title("This feature is coming soon!")
    st.button("⬅️ Back to Home", on_click=set_page, args=("Home",))

    ############## Choose excersice #############
elif st.session_state.page == "LiveExercise":
    st.title("Live Exercise Tracking (Browser View)")
    st.markdown("Make sure `video_streamer.py` is running in the background.")

    exercise = st.selectbox("Choose exercise", ["press", "lunge", "plank"])

    if st.button("Start Live Tracking"):
        st.success(f"Streaming exercise: {exercise}")
        st.markdown("🔴 Live stream below:")

        # Embed the video stream using an iframe
        # Inside LiveExercise page
    url = f"http://localhost:5000?exercise={exercise}"
    st.components.v1.iframe(url, width=1280, height=720)


    st.button("⬅️ Back to Home", on_click=set_page, args=("Home",))

