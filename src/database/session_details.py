from datetime import datetime
from src.database.database_connection import get_connection
import psycopg2
import json
from typing import Optional, List, Dict, Any

class SessionDetails:
    def __init__(self, detail_id=None, session_id=None, rep_number=None, timestamp=None,
                 features_json=None, is_correct_form=None, incorrect_duration=None):
        self.detail_id = detail_id
        self.session_id = session_id
        self.rep_number = rep_number
        self.timestamp = timestamp or datetime.now()
        self.features_json = self._safe_json_load(features_json)
        self.is_correct_form = is_correct_form if is_correct_form is not None else False
        self.incorrect_duration = incorrect_duration or 0.0

    @staticmethod
    def _safe_json_load(val):
        """Safely load JSON data"""
        if val is None or val == '' or val == 'null':
            return None
        if isinstance(val, dict):
            return val
        try:
            return json.loads(val) if isinstance(val, str) else val
        except Exception:
            return val

    @classmethod
    def create(cls, session_id: int, rep_number: int, features_json: Dict[str, Any],
               is_correct_form: bool = False, incorrect_duration: float = 0.0,
               timestamp: datetime = None) -> 'SessionDetails':
        """Create a new session detail record"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                timestamp = timestamp or datetime.now()
                cur.execute("""
                    INSERT INTO SessionDetails (session_id, rep_number, timestamp, features_json, 
                                               is_correct_form, incorrect_duration)
                    VALUES (%s, %s, %s, %s, %s, %s) RETURNING detail_id
                """, (session_id, rep_number, timestamp, json.dumps(features_json),
                      is_correct_form, incorrect_duration))
                detail_id = cur.fetchone()[0]
                conn.commit()
                
                return cls(
                    detail_id=detail_id, session_id=session_id, rep_number=rep_number,
                    timestamp=timestamp, features_json=features_json,
                    is_correct_form=is_correct_form, incorrect_duration=incorrect_duration
                )
        except psycopg2.Error as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    @classmethod
    def get_by_session(cls, session_id: int) -> List['SessionDetails']:
        """Get all details for a session"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT detail_id, session_id, rep_number, timestamp, features_json, 
                           is_correct_form, incorrect_duration
                    FROM SessionDetails 
                    WHERE session_id = %s 
                    ORDER BY timestamp, rep_number
                """, (session_id,))
                rows = cur.fetchall()
                return [cls(*row) for row in rows]
        finally:
            conn.close()

    @classmethod
    def get_by_rep(cls, session_id: int, rep_number: int) -> List['SessionDetails']:
        """Get all details for a specific rep in a session"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT detail_id, session_id, rep_number, timestamp, features_json, 
                           is_correct_form, incorrect_duration
                    FROM SessionDetails 
                    WHERE session_id = %s AND rep_number = %s 
                    ORDER BY timestamp
                """, (session_id, rep_number))
                rows = cur.fetchall()
                return [cls(*row) for row in rows]
        finally:
            conn.close()

    @classmethod
    def get_session_summary(cls, session_id: int) -> Dict[str, Any]:
        """Get summary statistics for a session"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_records,
                        COUNT(DISTINCT rep_number) as total_reps,
                        AVG(CASE WHEN is_correct_form THEN 1.0 ELSE 0.0 END) as form_accuracy,
                        SUM(incorrect_duration) as total_incorrect_duration,
                        MIN(timestamp) as first_timestamp,
                        MAX(timestamp) as last_timestamp
                    FROM SessionDetails 
                    WHERE session_id = %s
                """, (session_id,))
                row = cur.fetchone()
                
                if row and row[0] > 0:  # Check if any records exist
                    return {
                        'total_records': row[0],
                        'total_reps': row[1],
                        'form_accuracy': float(row[2]) if row[2] is not None else 0.0,
                        'total_incorrect_duration': float(row[3]) if row[3] is not None else 0.0,
                        'first_timestamp': row[4],
                        'last_timestamp': row[5],
                        'session_duration': (row[5] - row[4]).total_seconds() if row[4] and row[5] else 0
                    }
                else:
                    return {
                        'total_records': 0,
                        'total_reps': 0,
                        'form_accuracy': 0.0,
                        'total_incorrect_duration': 0.0,
                        'first_timestamp': None,
                        'last_timestamp': None,
                        'session_duration': 0
                    }
        finally:
            conn.close()

    def update_form_status(self, is_correct_form: bool, incorrect_duration: float = None) -> bool:
        """Update the form correctness status"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                if incorrect_duration is not None:
                    cur.execute("""
                        UPDATE SessionDetails 
                        SET is_correct_form = %s, incorrect_duration = %s
                        WHERE detail_id = %s
                    """, (is_correct_form, incorrect_duration, self.detail_id))
                    self.incorrect_duration = incorrect_duration
                else:
                    cur.execute("""
                        UPDATE SessionDetails 
                        SET is_correct_form = %s
                        WHERE detail_id = %s
                    """, (is_correct_form, self.detail_id))
                
                conn.commit()
                self.is_correct_form = is_correct_form
                return True
        except psycopg2.Error:
            conn.rollback()
            return False
        finally:
            conn.close()

    def to_dict(self) -> Dict[str, Any]:
        """Convert session detail to dictionary"""
        return {
            'detail_id': self.detail_id,
            'session_id': self.session_id,
            'rep_number': self.rep_number,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'features_json': self.features_json,
            'is_correct_form': self.is_correct_form,
            'incorrect_duration': self.incorrect_duration
        }