from datetime import datetime
from src.database.database_connection import get_connection
import psycopg2
from typing import Optional, List, Dict, Any

class Comment:
    def __init__(self, comment_id=None, session_id=None, user_id=None, 
                 timestamp=None, comment_text=None):
        self.comment_id = comment_id
        self.session_id = session_id
        self.user_id = user_id
        self.timestamp = timestamp or datetime.now()
        self.comment_text = comment_text

    @classmethod
    def create(cls, session_id: int, user_id: int, comment_text: str,
               timestamp: datetime = None) -> 'Comment':
        """Create a new comment"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                timestamp = timestamp or datetime.now()
                cur.execute("""
                    INSERT INTO Comment (session_id, user_id, timestamp, comment_text)
                    VALUES (%s, %s, %s, %s) RETURNING comment_id
                """, (session_id, user_id, timestamp, comment_text))
                comment_id = cur.fetchone()[0]
                conn.commit()
                
                return cls(
                    comment_id=comment_id, session_id=session_id, user_id=user_id,
                    timestamp=timestamp, comment_text=comment_text
                )
        except psycopg2.Error as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    @classmethod
    def get_by_session(cls, session_id: int) -> List['Comment']:
        """Get all comments for a session"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT c.comment_id, c.session_id, c.user_id, c.timestamp, c.comment_text
                    FROM Comment c
                    WHERE c.session_id = %s 
                    ORDER BY c.timestamp
                """, (session_id,))
                rows = cur.fetchall()
                return [cls(*row) for row in rows]
        finally:
            conn.close()

    @classmethod
    def get_by_user(cls, user_id: int, limit: int = 10) -> List['Comment']:
        """Get comments by a user"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT comment_id, session_id, user_id, timestamp, comment_text
                    FROM Comment 
                    WHERE user_id = %s 
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (user_id, limit))
                rows = cur.fetchall()
                return [cls(*row) for row in rows]
        finally:
            conn.close()

    @classmethod
    def get_with_user_info(cls, session_id: int) -> List[Dict[str, Any]]:
        """Get comments with user information for a session"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT c.comment_id, c.session_id, c.user_id, c.timestamp, c.comment_text,
                           u.username, u.first_name, u.last_name, u.user_type
                    FROM Comment c
                    JOIN users u ON c.user_id = u.user_id
                    WHERE c.session_id = %s 
                    ORDER BY c.timestamp
                """, (session_id,))
                rows = cur.fetchall()
                
                comments = []
                for row in rows:
                    comment = cls(
                        comment_id=row[0], session_id=row[1], user_id=row[2],
                        timestamp=row[3], comment_text=row[4]
                    )
                    comment_dict = comment.to_dict()
                    comment_dict['user_info'] = {
                        'username': row[5],
                        'first_name': row[6],
                        'last_name': row[7],
                        'user_type': row[8],
                        'display_name': f"{row[6]} {row[7]}".strip() if row[6] or row[7] else row[5]
                    }
                    comments.append(comment_dict)
                
                return comments
        finally:
            conn.close()

    @classmethod
    def get_recent_comments(cls, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent comments across all sessions with user and session info"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT c.comment_id, c.session_id, c.user_id, c.timestamp, c.comment_text,
                           u.username, u.first_name, u.last_name, u.user_type,
                           s.exercise_id, e.name as exercise_name
                    FROM Comment c
                    JOIN users u ON c.user_id = u.user_id
                    JOIN Session s ON c.session_id = s.session_id
                    JOIN Exercise e ON s.exercise_id = e.exercise_id
                    ORDER BY c.timestamp DESC
                    LIMIT %s
                """, (limit,))
                rows = cur.fetchall()
                
                comments = []
                for row in rows:
                    comment = cls(
                        comment_id=row[0], session_id=row[1], user_id=row[2],
                        timestamp=row[3], comment_text=row[4]
                    )
                    comment_dict = comment.to_dict()
                    comment_dict['user_info'] = {
                        'username': row[5],
                        'first_name': row[6],
                        'last_name': row[7],
                        'user_type': row[8],
                        'display_name': f"{row[6]} {row[7]}".strip() if row[6] or row[7] else row[5]
                    }
                    comment_dict['session_info'] = {
                        'exercise_id': row[9],
                        'exercise_name': row[10]
                    }
                    comments.append(comment_dict)
                
                return comments
        finally:
            conn.close()

    def update_text(self, comment_text: str) -> bool:
        """Update comment text"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE Comment 
                    SET comment_text = %s
                    WHERE comment_id = %s
                """, (comment_text, self.comment_id))
                conn.commit()
                self.comment_text = comment_text
                return True
        except psycopg2.Error:
            conn.rollback()
            return False
        finally:
            conn.close()

    def delete(self) -> bool:
        """Delete comment"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM Comment 
                    WHERE comment_id = %s
                """, (self.comment_id,))
                conn.commit()
                return True
        except psycopg2.Error:
            conn.rollback()
            return False
        finally:
            conn.close()

    def to_dict(self) -> Dict[str, Any]:
        """Convert comment to dictionary"""
        return {
            'comment_id': self.comment_id,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'comment_text': self.comment_text
        }