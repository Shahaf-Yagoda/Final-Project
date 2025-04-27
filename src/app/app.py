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
import mediapipe as mp
import base64
from streamlit_option_menu import option_menu

# Import custom modules
sys.path.insert(0, '../..')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.database.users.register import register_user
from src.database.users.login import login_user
from main import *  # if you're using logic from main



def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

image_path = os.path.join(os.path.dirname(__file__), "../assets/background.jpg")
base64_image = get_base64_image(image_path)

background_css = f"""
<style>
body::before {{
    content: "";
    background-image: url("data:image/jpeg;base64,{base64_image}");
    background-size: cover;
    background-position: center;
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: -1;
    opacity: 0.25;
}}
</style>
"""
st.markdown(background_css, unsafe_allow_html=True)

# Load custom CSS
css_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Set this variable based on your app's current flow/page
is_analyze_video_flow = st.session_state.get('is_analyze_video_flow', False)

# Session state to track navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None

def set_page(page_name):
    st.session_state.page = page_name


# Main page
if st.session_state.page == "Home":
    st.title("Welcome to Right Motion")

    if not st.session_state.logged_in:
        st.subheader("Please log in or register to continue.")
        col1, col2 = st.columns(2)
        with col1:
            st.button("🔐 Register", on_click=set_page, args=("Register",))
        with col2:
            st.button("🔑 Log In", on_click=set_page, args=("Login",))
    else:
        st.success(f"Logged in as {st.session_state.username}")
        st.subheader("Choose an option:")
        col1, col2 , col3 = st.columns(3)
        with col1:
            st.button("📹 Analyze Video", on_click=set_page, args=("Analyze",))
        with col2:
            st.button("🏃 Live Exercise", on_click=set_page, args=("LiveExercise",))
        with col3:
            st.button("🧩 TBD", on_click=set_page, args=("TBD",))
        st.button("🚪 Log Out", on_click=lambda: logout())
    def logout():
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.page = "Home"




# Register Page
elif st.session_state.page == "Register":
    st.title("User Registration")

    email = st.text_input("Email")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Profile fields
    name = st.text_input("Full Name")
    dob = st.date_input("Date of Birth")
    height = st.number_input("Height (cm)", min_value=0)
    weight = st.number_input("Weight (kg)", min_value=0)
    role = st.selectbox("Role", ["user", "coach", "admin"])

    if st.button("Register"):
        if not username or not password or not email:
            st.error("Email, username, and password are required.")
        else:
            profile_data = {
                "name": name,
                "date_of_birth": dob.isoformat(),
                "height_cm": height,
                "weight_kg": weight
            }
            result = register_user(email, username, password, profile_data, role)

            if isinstance(result, int):
                st.success(f"User registered with ID: {result}")
            else:
                st.error(result)

    st.button("⬅️ Back to Home", on_click=set_page, args=("Home",))


# Analyze Video Page (your original logic)
elif st.session_state.page == "Analyze":
    st.session_state.is_analyze_video_flow = True

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

    identifier = st.text_input("Email or Username")
    password = st.text_input("Password", type="password")

    if st.button("Log In"):
        user_id, username, error = login_user(identifier, password)

        if user_id:
            st.session_state.logged_in = True
            st.session_state.user_id = user_id
            st.session_state.username = username
            st.success(f"Login successful! Welcome, {username}.")
            st.session_state.page = "Home"
            st.rerun()
        else:
            st.error(error)

    st.button("⬅️ Back to Home", on_click=set_page, args=("Home",))




elif st.session_state.page == "TBD":
        if not st.session_state.logged_in:
            st.warning("Please log in to access this page.")
            st.button("🔑 Go to Login", on_click=set_page, args=("Login",))
        else:
            st.title("This feature is coming soon!")
            st.button("⬅️ Back to Home", on_click=set_page, args=("Home",))

    ############## Choose excersice #############
elif st.session_state.page == "LiveExercise":
    if not st.session_state.get("logged_in", False):
        st.warning("Please log in to access this page.")
        st.button("🔑 Go to Login", on_click=set_page, args=("Login",))
    else:
        st.title("Live Exercise Analyze")

        # Dropdown with stored selection
        exercise = option_menu(
            menu_title=None,
            options=["lunge", "press", "plank"],
            icons=["1-circle-fill", "2-circle-fill", "3-circle-fill"],
            orientation="horizontal",
        )

        # Start live tracking
        if st.button("Start Live Tracking"):
            st.session_state["start_streaming"] = True
            st.session_state["selected_exercise"] = exercise
            st.session_state["start_time"] = datetime.now().isoformat()
            st.session_state["reps_count"] = 0  # default

        # Display stream if tracking was started
        if st.session_state.get("start_streaming", False):
            selected = st.session_state.get("selected_exercise", "press")
            st.success(f"Streaming exercise: {selected}")

            user_id = st.session_state.get("user_id")
            url = f"http://localhost:5000?exercise={selected}&user_id={user_id}"
            st.components.v1.iframe(url, width=1280, height=720)

            if st.button("🛑 Stop Live Tracking"):
                st.session_state["start_streaming"] = False
                st.session_state["stop_time"] = datetime.now().isoformat()
                try:
                    with open(f"/tmp/reps_{user_id}.txt") as f:
                        reps = int(f.read().strip())
                        os.remove(f"/tmp/reps_{user_id}.txt")
                except:
                    reps = 234  # fallback if file not found

                start = datetime.fromisoformat(st.session_state["start_time"])
                end = datetime.fromisoformat(st.session_state["stop_time"])
                print(f"#########repssssss={reps}")
                #reps = st.session_state.get("reps_count",0)

                save_session_to_db(
                    user_id=user_id,
                    exercise_name=selected,
                    start_time=start,
                    end_time=end,
                    reps_count=reps
                )

                st.success("✅ Session saved to the database.")
        # End of LiveExercise logic
        if st.button("⬅️ Back to Home", key="back_home_live"):
            st.session_state.page = "Home"
            st.session_state["start_streaming"] = False
            st.session_state["selected_exercise"] = None
            for key in ["start_time", "stop_time", "reps_count"]:
                st.session_state.pop(key, None)
            st.rerun()  # 🔄 force rerun so Home page is fresh

