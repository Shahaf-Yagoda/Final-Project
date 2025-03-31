# users/login.py
import psycopg2
import bcrypt
from datetime import datetime
from database.database_connection import get_connection

def verify_password(plain_pw, hashed_pw):
    return bcrypt.checkpw(plain_pw.encode('utf-8'), hashed_pw.encode('utf-8'))

def login_user(username, password):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # 1. Get login info
        cursor.execute("""
            SELECT login.user_id, login.password
            FROM login
            WHERE login.username = %s
        """, (username,))
        record = cursor.fetchone()

        if not record:
            return None, "Username not found"

        user_id, stored_hash = record

        # 2. Check password
        if not verify_password(password, stored_hash):
            return None, "Incorrect password"

        # 3. Update lastlogin
        cursor.execute("""
            UPDATE users
            SET lastlogin = %s
            WHERE user_id = %s
        """, (datetime.now(), user_id))

        conn.commit()
        return user_id, None

    except Exception as e:
        return None, f"Database error: {e}"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
