from datetime import datetime, date
from src.database.database_connection import get_connection
import psycopg2
from typing import Optional, List, Dict, Any

class Workout:
    def __init__(self, workout_id=None, user_id=None, workout_date=None, 
                 start_time=None, end_time=None, total_duration=None, 
                 created_at=None, updated_at=None):
        self.workout_id = workout_id
        self.user_id = user_id
        self.workout_date = workout_date or date.today()
        self.start_time = start_time
        self.end_time = end_time
        self.total_duration = total_duration
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()

    @classmethod
    def create(cls, user_id: int, workout_date: date = None, 
               start_time: datetime = None) -> 'Workout':
        """Create a new workout"""
        workout_date = workout_date or date.today()
        start_time = start_time or datetime.now()
        
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                now = datetime.now()
                cur.execute("""
                    INSERT INTO Workout (user_id, workout_date, start_time, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s) RETURNING workout_id
                """, (user_id, workout_date, start_time, now, now))
                workout_id = cur.fetchone()[0]
                conn.commit()
                
                return cls(
                    workout_id=workout_id, user_id=user_id, workout_date=workout_date,
                    start_time=start_time, created_at=now, updated_at=now
                )
        except psycopg2.Error as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    @classmethod
    def get_by_id(cls, workout_id: int) -> Optional['Workout']:
        """Get workout by ID"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT workout_id, user_id, workout_date, start_time, end_time, 
                           total_duration, created_at, updated_at
                    FROM Workout WHERE workout_id = %s
                """, (workout_id,))
                row = cur.fetchone()
                if row:
                    return cls(*row)
                return None
        finally:
            conn.close()

    @classmethod
    def get_by_user(cls, user_id: int, limit: int = 10) -> List['Workout']:
        """Get workouts for a user"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT workout_id, user_id, workout_date, start_time, end_time, 
                           total_duration, created_at, updated_at
                    FROM Workout 
                    WHERE user_id = %s 
                    ORDER BY workout_date DESC, start_time DESC
                    LIMIT %s
                """, (user_id, limit))
                rows = cur.fetchall()
                return [cls(*row) for row in rows]
        finally:
            conn.close()

    def finish(self, end_time: datetime = None) -> bool:
        """Mark workout as finished and calculate duration"""
        end_time = end_time or datetime.now()
        
        if self.start_time:
            duration = int((end_time - self.start_time).total_seconds())
        else:
            duration = 0
        
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE Workout 
                    SET end_time = %s, total_duration = %s, updated_at = %s
                    WHERE workout_id = %s
                """, (end_time, duration, datetime.now(), self.workout_id))
                conn.commit()
                
                self.end_time = end_time
                self.total_duration = duration
                self.updated_at = datetime.now()
                return True
        except psycopg2.Error:
            conn.rollback()
            return False
        finally:
            conn.close()

    def get_sessions(self) -> List:
        """Get all sessions for this workout"""
        from src.database.session import Session
        return Session.get_by_workout(self.workout_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert workout to dictionary"""
        return {
            'workout_id': self.workout_id,
            'user_id': self.user_id,
            'workout_date': self.workout_date.isoformat() if isinstance(self.workout_date, date) else self.workout_date,
            'start_time': self.start_time.isoformat() if isinstance(self.start_time, datetime) else self.start_time,
            'end_time': self.end_time.isoformat() if isinstance(self.end_time, datetime) else self.end_time,
            'total_duration': self.total_duration,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'updated_at': self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at
        }