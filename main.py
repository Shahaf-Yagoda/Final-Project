import cv2
import mediapipe as mp
import numpy as np
import time
from src.database.database_connection import get_connection
import json
from datetime import datetime
import os
from dotenv import load_dotenv
import streamlit as st
from gtts import gTTS
from playsound import playsound
import os
import uuid
import threading
from threading import Lock
from src.processing.forms_check import *
from src.processing.feedback import speak_async, speak
from src.processing.forms_check import *



###############################
#  1) Setup MediaPipe Pose    #
###############################
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Tracking states
start_position_ready = False
reps_count = 0
direction = None  # "up" or "down"

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


def save_session_to_db(user_id, exercise_name, start_time, end_time, reps_count, video_path=None):
    try:
        print("💾 Saving session to DB...")
        print(f"🔍 user_id={user_id}, exercise_name={exercise_name}, reps={reps_count}")

        conn = get_connection()
        cursor = conn.cursor()

        # Step 1: Get exercise_id
        cursor.execute("SELECT exercise_id FROM Exercise WHERE name = %s", (exercise_name,))
        result = cursor.fetchone()
        if not result:
            print(f"❌ No exercise found with name: {exercise_name}")
            return
        exercise_id = result[0]

        # Step 2: Calculate duration
        duration = int((end_time - start_time).total_seconds())

        # Step 3: Insert session
        insert_query = """
            INSERT INTO Session (
                user_id, exercise_id, start_time, end_time,
                duration_sec, reps_count, feedback_count,
                performance_score, video_path
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            user_id, exercise_id, start_time, end_time,
            duration, st.session_state["reps_count"], 0, 0.0, video_path
        ))

        conn.commit()
        print("✅ Session saved successfully to the database.")

    except Exception as e:
        if conn:
            conn.rollback()
        print("❌ Error while saving session to database:", e)

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()




###############################
#  5) Main Entry Point        #
###############################
def main(exercise_name, user_id):
    print("👤 User ID:", user_id)
    user_id = st.session_state.get("user_id", 1)  # 🔁 fallback for testing
    workout_name = exercise_name

    state = {
        "exercise": {
            workout_name: init_state(workout_name)
        }
    }

    cap = cv2.VideoCapture(0)
    start_time = datetime.now()

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

                exercise_state = state["exercise"][workout_name]
                feedback, reps_count = check_form(workout_name, frame, landmarks, exercise_state)

                st.session_state["reps_count"] = reps_count
                exercise_state["count"] = reps_count

                keypoints_data = [{
                    "x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility
                } for lm in landmarks]
                save_keypoints_to_db(keypoints_data, workout_id=1, workout_name=workout_name)

            cv2.imshow('Pose Tracker', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        end_time = datetime.now()
        print("🕒 Session ended at:", end_time)
        print("⏱️ Duration:", (end_time - start_time).total_seconds(), "seconds")

    #     save_session_to_db(
    #         user_id=user_id,
    #         exercise_name=workout_name,
    #         start_time=start_time,
    #         end_time=end_time,
    #         reps_count=st.session_state["reps_count"]
    # )

        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exercise", type=str, required=True)
    parser.add_argument("--user_id", type=int, required=True)
    args = parser.parse_args()

    main(exercise_name=args.exercise, user_id=args.user_id)