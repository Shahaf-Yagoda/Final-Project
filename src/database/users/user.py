from datetime import datetime, date, time
from src.database.database_connection import get_connection
import psycopg2
import bcrypt
import json
from typing import Optional, Dict, Any

class User:
    def __init__(self, user_id=None, email=None, password=None, 
                 first_name=None, last_name=None, profile_data=None,
                 registration_date=None, last_login=None, user_type=None, is_active=None):
        self.user_id = user_id
        self.email = email
        self.password = password
        self.first_name = first_name
        self.last_name = last_name
        self.profile_data = self._safe_json_load(profile_data)
        self.registration_date = registration_date or datetime.now()
        self.last_login = last_login
        self.user_type = user_type if isinstance(user_type, str) else 'user'
        self.is_active = is_active if is_active is not None else True

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
    def register(cls, email: str, password: str, 
                profile_data: Optional[Dict[str, Any]] = None, 
                user_type: str = 'user', first_name: str = None, 
                last_name: str = None) -> 'User':
        """Register a new user"""
        # Validate inputs
        user_type = cls._validate_role(user_type)  # Reuse role validation for user_type
        hashed_pw = cls.hash_password(password)
        
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                now = datetime.now()
                cur.execute("""
                    INSERT INTO users (email, password, first_name, last_name,
                                   profile_data, registration_date, last_login,
                                   user_type, is_active)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING user_id
                """, (email, hashed_pw, first_name, last_name,
                      json.dumps(profile_data) if profile_data else None,
                      now, now, user_type, True))
                user_id = cur.fetchone()[0]
                conn.commit()
                return cls(user_id=user_id, email=email, password=hashed_pw,
                         first_name=first_name, last_name=last_name,
                         profile_data=profile_data, registration_date=now,
                         last_login=now, user_type=user_type, is_active=True)
        except psycopg2.Error as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    @classmethod
    def authenticate(cls, email: str, password: str) -> Optional['User']:
        """Authenticate user"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT user_id, email, password, first_name, last_name,
                           profile_data, registration_date, last_login,
                           user_type, is_active
                    FROM users
                    WHERE email = %s AND is_active = TRUE
                """, (email,))
                row = cur.fetchone()
                if row and cls.verify_password(password, row[2]):
                    # Update last_login
                    now = datetime.now()
                    cur.execute("""
                        UPDATE users SET last_login = %s WHERE user_id = %s
                    """, (now, row[0]))
                    conn.commit()
                    
                    return cls(
                        user_id=row[0], email=row[1], password=row[2],
                        first_name=row[3], last_name=row[4],
                        profile_data=row[5], registration_date=row[6],
                        last_login=now, user_type=row[8], is_active=row[9]
                    )
                else:
                    return None
        finally:
            conn.close()

    @classmethod
    def get_by_id(cls, user_id: int) -> Optional['User']:
        """Get user by ID"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT user_id, email, password, first_name, last_name,
                           profile_data, registration_date, last_login,
                           user_type, is_active
                    FROM users WHERE user_id = %s
                """, (user_id,))
                row = cur.fetchone()
                if row:
                    return cls(
                        user_id=row[0], email=row[1], password=row[2],
                        first_name=row[3], last_name=row[4],
                        profile_data=row[5], registration_date=row[6],
                        last_login=row[7], user_type=row[8], is_active=row[9]
                    )
                else:
                    return None
        finally:
            conn.close()

    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary"""
        return {
            'user_id': self.user_id,
            'email': self.email,
            'password': self.password,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'profile_data': self.profile_data,
            'registration_date': self.registration_date.isoformat() if isinstance(self.registration_date, datetime) else self.registration_date,
            'last_login': self.last_login.isoformat() if isinstance(self.last_login, datetime) else self.last_login,
            'user_type': self.user_type,
            'is_active': self.is_active
        }
    
    def get_full_name(self) -> str:
        """Get user's full name"""
        parts = []
        if self.first_name:
            parts.append(self.first_name)
        if self.last_name:
            parts.append(self.last_name)
        return ' '.join(parts) if parts else self.username
    
    def deactivate(self) -> bool:
        """Deactivate user account"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE users 
                    SET is_active = FALSE, updated_at = %s 
                    WHERE user_id = %s
                """, (datetime.now(), self.user_id))
                conn.commit()
                self.is_active = False
                self.updated_at = datetime.now()
                return True
        except psycopg2.Error:
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def update_profile(self, profile_data: Dict[str, Any]) -> bool:
        """Update user profile data"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE users 
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
                        UPDATE users 
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
