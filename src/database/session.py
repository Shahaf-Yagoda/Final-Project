from datetime import datetime
from src.database.database_connection import get_connection
import psycopg2

class Session:
    def __init__(self, session_id=None, user_id=None, exercise_id=None, start_time=None, end_time=None, video_path=None, reps_count=0, feedback_count=0, performance_score=None, duration_sec=None):
        self.session_id = session_id
        self.user_id = user_id
        self.exercise_id = exercise_id
        self.start_time = start_time or datetime.now()
        self.end_time = end_time
        self.video_path = video_path
        self.reps_count = reps_count
        self.feedback_count = feedback_count
        self.performance_score = performance_score
        self.duration_sec = duration_sec

    def save(self):
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO session (user_id, exercise_id, start_time, end_time, duration_sec, video_path, reps_count, feedback_count, performance_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING session_id
                """, (
                    self.user_id,
                    self.exercise_id,
                    self.start_time,
                    self.end_time,
                    self.duration_sec,
                    self.video_path,
                    self.reps_count,
                    self.feedback_count,
                    self.performance_score
                ))
                self.session_id = cur.fetchone()[0]
                conn.commit()
        except psycopg2.Error as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    @classmethod
    def load_by_user(cls, user_id):
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT session_id, user_id, exercise_id, start_time, end_time, duration_sec, video_path, reps_count, feedback_count, performance_score
                    FROM session WHERE user_id = %s ORDER BY start_time DESC
                """, (user_id,))
                sessions = []
                for row in cur.fetchall():
                    sessions.append(cls(*row))
                return sessions
        finally:
            conn.close()

    @classmethod
    def load_by_id(cls, session_id):
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT session_id, user_id, exercise_id, start_time, end_time, duration_sec, video_path, reps_count, feedback_count, performance_score
                    FROM session WHERE session_id = %s
                """, (session_id,))
                row = cur.fetchone()
                if row:
                    return cls(*row)
                else:
                    return None
        finally:
            conn.close()

    def to_dict(self):
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'exercise_id': self.exercise_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_sec': self.duration_sec,
            'video_path': self.video_path,
            'reps_count': self.reps_count,
            'feedback_count': self.feedback_count,
            'performance_score': self.performance_score
        }
