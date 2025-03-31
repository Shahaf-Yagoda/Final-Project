# users/register.py

import psycopg2
from database.database_connection import get_connection

def register_user(name, email, date_of_birth, role=None, parent=None):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        insert_query = """
        INSERT INTO users (name, email, date_of_birth, role, parent)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING user_id;
        """
        cursor.execute(insert_query, (name, email, date_of_birth, role, parent))
        user_id = cursor.fetchone()[0]
        conn.commit()

        return user_id

    except psycopg2.IntegrityError as e:
        conn.rollback()
        return f"Error: Email already exists or invalid data. ({e})"
    except Exception as e:
        return f"Database error: {e}"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
