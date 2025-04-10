import psycopg2
import bcrypt
from datetime import datetime
from src.database.database_connection import get_connection

def verify_password(plain_pw, hashed_pw):
    return bcrypt.checkpw(plain_pw.encode('utf-8'), hashed_pw.encode('utf-8'))

def login_user(identifier, password):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # 1. Get user by email or username
        cursor.execute("""
            SELECT user_id, username, password
            FROM "User"
            WHERE email = %s OR username = %s
        """, (identifier, identifier))
        record = cursor.fetchone()

        if not record:
            return None, None, "User not found"

        user_id, username, stored_hash = record

        # 2. Check password
        if not verify_password(password, stored_hash):
            return None, None, "Incorrect password"

        # 3. Update last login timestamp
        cursor.execute("""
            UPDATE "User"
            SET registration_date = %s
            WHERE user_id = %s
        """, (datetime.now(), user_id))  # Optional: rename this field to `last_login`

        conn.commit()
        return user_id, username, None

    except Exception as e:
        return None, None, f"Database error: {e}"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
