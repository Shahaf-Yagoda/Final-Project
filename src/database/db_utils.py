import json
from datetime import datetime
from .database_connection import get_connection  # Adjust import if necessary

# Utility to get exercise_id from Exercise table by name
def get_exercise_id_by_name(exercise_name):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT exercise_id FROM Exercise WHERE name = %s", (exercise_name,))
            row = cursor.fetchone()
            if row:
                return row[0]
            else:
                raise ValueError(f"Exercise not found: {exercise_name}")
    finally:
        conn.close()

def save_session_to_db(user_id, exercise_name, start_time, end_time, reps_count,
                       video_path=None, feedback_count=None, performance_score=None):
    exercise_id = get_exercise_id_by_name(exercise_name)
    duration_sec = int((datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)).total_seconds())

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO Session (user_id, exercise_id, start_time, end_time, duration_sec, reps_count, 
                                    feedback_count, performance_score, video_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING session_id;
                """,
                (
                    user_id, exercise_id, start_time, end_time, duration_sec, reps_count,
                    feedback_count, performance_score, video_path
                )
            )
            session_id = cursor.fetchone()[0]
            conn.commit()
            return session_id
    finally:
        conn.close()

def save_session_details_to_db(session_id, timestamp, rep_num, keypoints_json,
                               features_json, is_correct, incorrect_duration):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Convert numeric timestamp to datetime if needed
            if isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp)
            
            cursor.execute(
                """
                INSERT INTO SessionDetails (session_id, timestamp, rep_num, keypoints_json, features_json, is_correct, incorrect_duration)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    timestamp,
                    rep_num,
                    keypoints_json if isinstance(keypoints_json, str) else json.dumps(keypoints_json),
                    features_json if isinstance(features_json, str) else json.dumps(features_json),
                    is_correct,
                    incorrect_duration
                )
            )
            conn.commit()
    finally:
        conn.close()

def save_system_feedback_to_db(session_id, timestamp, message):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Convert numeric timestamp to datetime if needed
            if isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp)
            
            cursor.execute(
                """
                INSERT INTO SystemFeedback (session_id, timestamp, message)
                VALUES (%s, %s, %s)
                """,
                (
                    session_id,
                    timestamp,
                    message
                )
            )
            conn.commit()
    finally:
        conn.close()

def save_comment_to_db(session_id, user_id, timestamp, comment):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Convert numeric timestamp to datetime if needed
            if isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp)
            
            cursor.execute(
                """
                INSERT INTO Comment (session_id, user_id, timestamp, comment)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    session_id, user_id, timestamp, comment
                )
            )
            conn.commit()
    finally:
        conn.close()
