from datetime import datetime
from src.database.database_connection import get_connection
import psycopg2
import bcrypt
import json

class User:
    def __init__(self, user_id=None, email=None, username=None, password=None, registration_date=None, profile_data=None, role=None):
        self.user_id = user_id
        self.email = email
        self.username = username
        self.password = password
        self.registration_date = registration_date or datetime.now()
        self.profile_data = self._safe_json_load(profile_data)
        self.role = self._safe_json_load(role)

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
    def hash_password(password):
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    @staticmethod
    def verify_password(plain_pw, hashed_pw):
        return bcrypt.checkpw(plain_pw.encode('utf-8'), hashed_pw.encode('utf-8'))

    @classmethod
    def register(cls, email, username, password, profile_data=None, role=None):
        hashed_pw = cls.hash_password(password)
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO "User" (email, username, password, registration_date, profile_data, role)
                    VALUES (%s, %s, %s, %s, %s, %s) RETURNING user_id
                """, (email, username, hashed_pw, datetime.now(), json.dumps(profile_data) if profile_data else None, json.dumps(role) if role else None))
                user_id = cur.fetchone()[0]
                conn.commit()
                return cls(user_id, email, username, hashed_pw, datetime.now(), profile_data, role)
        except psycopg2.Error as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    @classmethod
    def authenticate(cls, identifier, password):
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT user_id, email, username, password, registration_date, profile_data, role
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
    def get_by_id(cls, user_id):
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT user_id, email, username, password, registration_date, profile_data, role
                    FROM "User" WHERE user_id = %s
                """, (user_id,))
                row = cur.fetchone()
                if row:
                    return cls(*row)
                else:
                    return None
        finally:
            conn.close()

    def to_dict(self):
        return {
            'user_id': self.user_id,
            'email': self.email,
            'username': self.username,
            'registration_date': self.registration_date,
            'profile_data': self.profile_data,
            'role': self.role
        }
