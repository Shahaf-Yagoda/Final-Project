import os
import psycopg2
from dotenv import load_dotenv


def get_connection(use_cloud=None):
    """
    Returns a psycopg2 connection to either a cloud or local database
    based on the USE_CLOUD_DB environment variable, or overridden by the use_cloud argument.
    """
    try:
        load_dotenv()

        # Allow override via function argument
        if use_cloud is None:
            use_cloud = os.getenv("USE_CLOUD_DB", "false").lower() == "true"

        if use_cloud:
            db_name = os.getenv("DB_NAME_CLOUD")
            db_user = os.getenv("DB_USER_CLOUD")
            db_password = os.getenv("DB_PASSWORD_CLOUD")
            db_address = os.getenv("DB_ADDRESS_CLOUD")
            sslmode = "require"
        else:
            db_name = os.getenv("DB_NAME_LOCAL")
            db_user = os.getenv("DB_USER_LOCAL")
            db_password = os.getenv("DB_PASSWORD_LOCAL")
            db_address = os.getenv("DB_ADDRESS_LOCAL", "localhost")
            sslmode = "disable"

        connection_string = (
            f"postgresql://{db_user}:{db_password}@{db_address}/{db_name}?sslmode={sslmode}"
        )

        return psycopg2.connect(connection_string)

    except psycopg2.Error as e:
        print("Error connecting to the database:", e)
        raise
