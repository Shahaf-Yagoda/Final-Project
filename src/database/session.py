from datetime import datetime
from src.database.database_connection import get_connection
import psycopg2
from typing import Optional, List, Dict, Any

class Session:
    def __init__(self, session_id=None, workout_id=None, exercise_id=None,
                 session_order=None, start_time=None, end_time=None, duration=None,
                 planned_reps=None, actual_reps=None, session_status='in_progress',
                 video_path=None):
        self.session_id = session_id
        self.workout_id = workout_id
        self.exercise_id = exercise_id
        self.session_order = session_order
        self.start_time = start_time or datetime.now()
        self.end_time = end_time
        self.duration = duration
        self.planned_reps = planned_reps
        self.actual_reps = actual_reps
        self.session_status = session_status
        self.video_path = video_path

    def save(self):
        """Save session"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                if self.session_id is None:
                    # Insert new session
                    cur.execute("""
                        INSERT INTO sessions (workout_id, exercise_id, session_order,
                                           start_time, end_time, duration, planned_reps, actual_reps,
                                           session_status, video_path)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING session_id
                    """, (
                        self.workout_id, self.exercise_id, self.session_order,
                        self.start_time, self.end_time, self.duration, self.planned_reps,
                        self.actual_reps, self.session_status, self.video_path
                    ))
                    self.session_id = cur.fetchone()[0]
                else:
                    # Update existing session
                    cur.execute("""
                        UPDATE sessions SET workout_id = %s, exercise_id = %s,
                               session_order = %s, start_time = %s, end_time = %s, duration = %s,
                               planned_reps = %s, actual_reps = %s, session_status = %s,
                               video_path = %s
                        WHERE session_id = %s
                    """, (
                        self.workout_id, self.exercise_id, self.session_order,
                        self.start_time, self.end_time, self.duration, self.planned_reps,
                        self.actual_reps, self.session_status, self.video_path,
                        self.session_id
                    ))
                
                conn.commit()
                
        except psycopg2.Error as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    @classmethod
    def create(cls, exercise_id: int, workout_id: int = None,
               session_order: int = 1, planned_reps: int = None) -> 'Session':
        """Create a new session"""
        session = cls(
            workout_id=workout_id,
            exercise_id=exercise_id,
            session_order=session_order,
            planned_reps=planned_reps,
            session_status='in_progress'
        )
        session.save()
        return session

    @classmethod
    def load_by_user(cls, user_id: int, limit: int = 10) -> List['Session']:
        """Load sessions by user"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT s.session_id, s.workout_id, s.exercise_id, s.session_order,
                           s.start_time, s.end_time, s.duration, s.planned_reps, s.actual_reps,
                           s.session_status, s.video_path
                    FROM sessions s
                    JOIN workouts w ON s.workout_id = w.workout_id
                    WHERE w.user_id = %s 
                    ORDER BY s.start_time DESC LIMIT %s
                """, (user_id, limit))
                sessions = []
                for row in cur.fetchall():
                    session = cls(
                        session_id=row[0], workout_id=row[1], exercise_id=row[2],
                        session_order=row[3], start_time=row[4], end_time=row[5], duration=row[6],
                        planned_reps=row[7], actual_reps=row[8], session_status=row[9],
                        video_path=row[10]
                    )
                    sessions.append(session)
                return sessions
        finally:
            conn.close()

    @classmethod
    def load_by_id(cls, session_id: int) -> Optional['Session']:
        """Load session by ID"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT session_id, workout_id, exercise_id, session_order,
                           start_time, end_time, duration, planned_reps, actual_reps,
                           session_status, video_path
                    FROM sessions WHERE session_id = %s
                """, (session_id,))
                row = cur.fetchone()
                if row:
                    return cls(
                        session_id=row[0], workout_id=row[1], exercise_id=row[2],
                        session_order=row[3], start_time=row[4], end_time=row[5], duration=row[6],
                        planned_reps=row[7], actual_reps=row[8], session_status=row[9],
                        video_path=row[10]
                    )
                return None
        finally:
            conn.close()

    @classmethod
    def get_by_workout(cls, workout_id: int) -> List['Session']:
        """Get all sessions for a workout"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT session_id, workout_id, exercise_id, session_order,
                           start_time, end_time, duration, planned_reps, actual_reps,
                           session_status, video_path
                    FROM sessions WHERE workout_id = %s ORDER BY session_order, start_time
                """, (workout_id,))
                sessions = []
                for row in cur.fetchall():
                    session = cls(
                        session_id=row[0], workout_id=row[1], exercise_id=row[2],
                        session_order=row[3], start_time=row[4], end_time=row[5], duration=row[6],
                        planned_reps=row[7], actual_reps=row[8], session_status=row[9],
                        video_path=row[10]
                    )
                    sessions.append(session)
                return sessions
        finally:
            conn.close()

    def finish(self, actual_reps: int = None, end_time: datetime = None) -> bool:
        """Mark session as completed"""
        self.end_time = end_time or datetime.now()
        if actual_reps is not None:
            self.actual_reps = actual_reps
        
        if self.start_time and self.end_time:
            self.duration = int((self.end_time - self.start_time).total_seconds())
        
        self.session_status = 'completed'
        self.save()
        return True

    def get_details(self):
        """Get session details"""
        from src.database.session_details import SessionDetails
        return SessionDetails.get_by_session(self.session_id)

    def get_feedback(self):
        """Get system feedback for this session"""
        from src.database.system_feedback import SystemFeedback
        return SystemFeedback.get_by_session(self.session_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return {
            'session_id': self.session_id,
            'workout_id': self.workout_id,
            'exercise_id': self.exercise_id,
            'session_order': self.session_order,
            'start_time': self.start_time.isoformat() if isinstance(self.start_time, datetime) else self.start_time,
            'end_time': self.end_time.isoformat() if isinstance(self.end_time, datetime) else self.end_time,
            'duration': self.duration,
            'planned_reps': self.planned_reps,
            'actual_reps': self.actual_reps,
            'session_status': self.session_status,
            'video_path': self.video_path
        }
