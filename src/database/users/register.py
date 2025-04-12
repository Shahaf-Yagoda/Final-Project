# users/register.py
import psycopg2
import bcrypt
from src.database.database_connection import get_connection
import json

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def register_user(email, username, password, profile_data=None, role="user"):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Check if email already exists
        cursor.execute('SELECT 1 FROM "User" WHERE email = %s;', (email,))
        if cursor.fetchone():
            return "Error: Email already exists."

        # Check if username already exists
        cursor.execute('SELECT 1 FROM "User" WHERE username = %s;', (username,))
        if cursor.fetchone():
            return "Error: Username already exists."

        # Hash password and convert JSON
        hashed_pw = hash_password(password)
        profile_json = json.dumps(profile_data) if profile_data else None
        role_json = json.dumps(role)

        # Insert into User table
        insert_user = """
        INSERT INTO "User" (email, username, password, registration_date, profile_data, role)
        VALUES (%s, %s, %s, NOW(), %s, %s)
        RETURNING user_id;
        """
        cursor.execute(insert_user, (email, username, hashed_pw, profile_json, role_json))
        user_id = cursor.fetchone()[0]

        conn.commit()
        return user_id

    except psycopg2.IntegrityError as e:
        conn.rollback()
        return f"Integrity error: {e}"
    except Exception as e:
        return f"Database error: {e}"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
