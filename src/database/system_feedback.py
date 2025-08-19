from datetime import datetime
from src.database.database_connection import get_connection
import psycopg2
from typing import Optional, List, Dict, Any

class SystemFeedback:
    def __init__(self, feedback_id=None, session_id=None, timestamp=None, 
                 feedback_type='form_correction', message=None, related_rep=None):
        self.feedback_id = feedback_id
        self.session_id = session_id
        self.timestamp = timestamp or datetime.now()
        self.feedback_type = feedback_type
        self.message = message
        self.related_rep = related_rep

    @classmethod
    def create(cls, session_id: int, message: str, feedback_type: str = 'form_correction',
               related_rep: int = None, timestamp: datetime = None) -> 'SystemFeedback':
        """Create a new system feedback record"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                timestamp = timestamp or datetime.now()
                cur.execute("""
                    INSERT INTO system_feedback (session_id, timestamp, feedback_type, message, related_rep)
                    VALUES (%s, %s, %s, %s, %s) RETURNING feedback_id
                """, (session_id, timestamp, feedback_type, message, related_rep))
                feedback_id = cur.fetchone()[0]
                conn.commit()
                
                return cls(
                    feedback_id=feedback_id, session_id=session_id, timestamp=timestamp,
                    feedback_type=feedback_type, message=message, related_rep=related_rep
                )
        except psycopg2.Error as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    @classmethod
    def get_by_session(cls, session_id: int) -> List['SystemFeedback']:
        """Get all feedback for a session"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT feedback_id, session_id, timestamp, feedback_type, message, related_rep
                    FROM system_feedback 
                    WHERE session_id = %s 
                    ORDER BY timestamp
                """, (session_id,))
                rows = cur.fetchall()
                return [cls(*row) for row in rows]
        finally:
            conn.close()

    @classmethod
    def get_by_type(cls, session_id: int, feedback_type: str) -> List['SystemFeedback']:
        """Get feedback by type for a session"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT feedback_id, session_id, timestamp, feedback_type, message, related_rep
                    FROM system_feedback 
                    WHERE session_id = %s AND feedback_type = %s 
                    ORDER BY timestamp
                """, (session_id, feedback_type))
                rows = cur.fetchall()
                return [cls(*row) for row in rows]
        finally:
            conn.close()

    @classmethod
    def get_by_rep(cls, session_id: int, rep_number: int) -> List['SystemFeedback']:
        """Get feedback for a specific rep"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT feedback_id, session_id, timestamp, feedback_type, message, related_rep
                    FROM system_feedback 
                    WHERE session_id = %s AND related_rep = %s 
                    ORDER BY timestamp
                """, (session_id, rep_number))
                rows = cur.fetchall()
                return [cls(*row) for row in rows]
        finally:
            conn.close()

    @classmethod
    def get_feedback_summary(cls, session_id: int) -> Dict[str, Any]:
        """Get feedback summary for a session"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        feedback_type,
                        COUNT(*) as count,
                        MIN(timestamp) as first_occurrence,
                        MAX(timestamp) as last_occurrence
                    FROM system_feedback 
                    WHERE session_id = %s 
                    GROUP BY feedback_type
                    ORDER BY count DESC
                """, (session_id,))
                rows = cur.fetchall()
                
                summary = {
                    'total_feedback': sum(row[1] for row in rows),
                    'feedback_by_type': {}
                }
                
                for feedback_type, count, first, last in rows:
                    summary['feedback_by_type'][feedback_type] = {
                        'count': count,
                        'first_occurrence': first.isoformat() if first else None,
                        'last_occurrence': last.isoformat() if last else None
                    }
                
                return summary
        finally:
            conn.close()

    @classmethod
    def get_critical_feedback(cls, session_id: int) -> List['SystemFeedback']:
        """Get only critical feedback for a session"""
        return cls.get_by_type(session_id, 'critical')

    @classmethod
    def get_form_corrections(cls, session_id: int) -> List['SystemFeedback']:
        """Get form correction feedback for a session"""
        return cls.get_by_type(session_id, 'form_correction')

    def update_type(self, feedback_type: str) -> bool:
        """Update feedback type"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE system_feedback 
                    SET feedback_type = %s
                    WHERE feedback_id = %s
                """, (feedback_type, self.feedback_id))
                conn.commit()
                self.feedback_type = feedback_type
                return True
        except psycopg2.Error:
            conn.rollback()
            return False
        finally:
            conn.close()

    def to_dict(self) -> Dict[str, Any]:
        """Convert system feedback to dictionary"""
        return {
            'feedback_id': self.feedback_id,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'feedback_type': self.feedback_type,
            'message': self.message,
            'related_rep': self.related_rep
        }