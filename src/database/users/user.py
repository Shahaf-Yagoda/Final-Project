from datetime import datetime, date, time
from src.database.database_connection import get_connection
import psycopg2
import bcrypt
import json
from typing import Optional, Dict, Any

class User:
    def __init__(self, user_id=None, email=None, username=None, password=None, 
                 registration_date=None, registration_time=None, profile_data=None, 
                 role=None, created_at=None, updated_at=None):
        self.user_id = user_id
        self.email = email
        self.username = username
        self.password = password
        self.registration_date = registration_date or date.today()
        self.registration_time = registration_time or datetime.now().time()
        self.profile_data = self._safe_json_load(profile_data)
        self.role = role if isinstance(role, str) else 'user'  # Role is now enum string
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()

    @staticmethod
    def _safe_json_load(val):
        if val is None or val == '' or val == 'null':
            return None
        if isinstance(val, dict):
            return val
        try:
            return json.loads(val)
        except Exception:
            return val
    
    @staticmethod
    def _validate_role(role: str) -> str:
        """Validate role is one of the allowed values"""
        valid_roles = ['user', 'coach', 'admin']
        if role not in valid_roles:
            raise ValueError(f"Invalid role: {role}. Must be one of {valid_roles}")
        return role

    @staticmethod
    def hash_password(password):
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    @staticmethod
    def verify_password(plain_pw, hashed_pw):
        return bcrypt.checkpw(plain_pw.encode('utf-8'), hashed_pw.encode('utf-8'))

    @classmethod
    def register(cls, email: str, username: str, password: str, 
                profile_data: Optional[Dict[str, Any]] = None, 
                role: str = 'user') -> 'User':
        """Register a new user with improved schema"""
        # Validate inputs
        role = cls._validate_role(role)
        hashed_pw = cls.hash_password(password)
        
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                now = datetime.now()
                cur.execute("""
                    INSERT INTO "User" (email, username, password, registration_date, 
                                       registration_time, profile_data, role, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING user_id
                """, (email, username, hashed_pw, now.date(), now.time(), 
                      json.dumps(profile_data) if profile_data else None, 
                      role, now, now))
                user_id = cur.fetchone()[0]
                conn.commit()
                return cls(user_id, email, username, hashed_pw, now.date(), 
                          now.time(), profile_data, role, now, now)
        except psycopg2.Error as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    @classmethod
    def authenticate(cls, identifier: str, password: str) -> Optional['User']:
        """Authenticate user with updated schema"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT user_id, email, username, password, registration_date, 
                           registration_time, profile_data, role, created_at, updated_at
                    FROM "User"
                    WHERE email = %s OR username = %s
                """, (identifier, identifier))
                row = cur.fetchone()
                if row and cls.verify_password(password, row[3]):
                    return cls(*row)
                else:
                    return None
        finally:
            conn.close()

    @classmethod
    def get_by_id(cls, user_id: int) -> Optional['User']:
        """Get user by ID with updated schema"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT user_id, email, username, password, registration_date, 
                           registration_time, profile_data, role, created_at, updated_at
                    FROM "User" WHERE user_id = %s
                """, (user_id,))
                row = cur.fetchone()
                if row:
                    return cls(*row)
                else:
                    return None
        finally:
            conn.close()

    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary with updated fields"""
        return {
            'user_id': self.user_id,
            'email': self.email,
            'username': self.username,
            'registration_date': self.registration_date.isoformat() if isinstance(self.registration_date, date) else self.registration_date,
            'registration_time': self.registration_time.isoformat() if isinstance(self.registration_time, time) else self.registration_time,
            'profile_data': self.profile_data,
            'role': self.role,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'updated_at': self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at
        }
    
    def update_profile(self, profile_data: Dict[str, Any]) -> bool:
        """Update user profile data"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE "User" 
                    SET profile_data = %s, updated_at = %s 
                    WHERE user_id = %s
                """, (json.dumps(profile_data), datetime.now(), self.user_id))
                conn.commit()
                self.profile_data = profile_data
                self.updated_at = datetime.now()
                return True
        except psycopg2.Error:
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def change_role(self, new_role: str) -> bool:
        """Change user role"""
        try:
            new_role = self._validate_role(new_role)
            conn = get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE "User" 
                        SET role = %s, updated_at = %s 
                        WHERE user_id = %s
                    """, (new_role, datetime.now(), self.user_id))
                    conn.commit()
                    self.role = new_role
                    self.updated_at = datetime.now()
                    return True
            except psycopg2.Error:
                conn.rollback()
                return False
            finally:
                conn.close()
        except ValueError:
            return False
