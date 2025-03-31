# users/register.py
import psycopg2
import bcrypt
from database.database_connection import get_connection

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def register_user(name, email, date_of_birth, role=None, parent=None, username=None, password=None):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Insert into users table
        insert_user = """
        INSERT INTO users (name, email, date_of_birth, role, parent)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING user_id;
        """
        cursor.execute(insert_user, (name, email, date_of_birth, role, parent))
        user_id = cursor.fetchone()[0]

        # Insert into login table
        hashed_pw = hash_password(password)
        insert_login = """
        INSERT INTO login (user_id, username, password)
        VALUES (%s, %s, %s);
        """
        cursor.execute(insert_login, (user_id, username, hashed_pw))

        conn.commit()
        return user_id

    except psycopg2.IntegrityError as e:
        conn.rollback()
        return f"Error: Email or Username may already exist. ({e})"
    except Exception as e:
        return f"Database error: {e}"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
