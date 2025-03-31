# app.py
import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import subprocess
import importlib
import sys

sys.path.insert(0, '../..')
from run2 import *  # Import feedback functions from run2.py

# Create a title for the app
st.title("Right motion video analyzer")

# Create a file uploader for the user to upload a video file
st.header("Upload a video to analyze")
video_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])

# Create a button to start Pose Estimation
st.header("Start Pose Estimation")
if st.button("Start"):
    if video_file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            video_path = tmp.name  # Store temp file path

        # Open the video file using OpenCV
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            st.error("Error: Unable to open video file")
            exit()

        # Create a MediaPipe pose estimator
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

        # Create a temporary output video file
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        # Define the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform pose estimation
            results = pose.process(rgb_frame)

            # Draw landmarks
            if results.pose_landmarks:
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Add feedback overlay to the frame
                check_squat_form(frame, results.pose_landmarks.landmark)  # Reuse the feedback function from run2.py

            # Write the frame with feedback to the output video
            out.write(frame)

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Ensure the file is completely written before displaying
        time.sleep(1)

        # Convert the video using FFmpeg to ensure browser compatibility
        converted_output_path = output_path.replace(".mp4", "_converted.mp4")
        ffmpeg_command = f"ffmpeg -i {output_path} -vcodec libx264 -crf 23 {converted_output_path}"
        subprocess.run(ffmpeg_command, shell=True, check=True)

        # Display the processed video in Streamlit
        st.video(converted_output_path)
