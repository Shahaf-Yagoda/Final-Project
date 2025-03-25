# database_connection.py

import os
import psycopg2
from dotenv import load_dotenv

def get_connection():
    """
    Returns a psycopg2 connection using environment variables:
      DB_USER, DB_PASSWORD, DB_ADDRESS, DB_NAME
    """
    try:
        # Load variables from .env
        load_dotenv()

        db_user = os.environ.get("DB_USER")
        db_password = os.environ.get("DB_PASSWORD")
        db_address = os.environ.get("DB_ADDRESS")
        db_name = os.environ.get("DB_NAME")

        # Build the Postgres connection string
        connection_string = (
            f"postgresql://{db_user}:{db_password}@{db_address}/{db_name}?sslmode=require"
        )

        conn = psycopg2.connect(connection_string)
        return conn

    except psycopg2.Error as e:
        print("Error connecting to the database:", e)
        # Optionally re-raise the exception or handle it differently
        raise
