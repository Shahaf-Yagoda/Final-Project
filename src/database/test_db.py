from database_connection import get_connection
import os
from dotenv import load_dotenv

load_dotenv()

def test_connection(use_cloud: bool):
    os.environ["USE_CLOUD_DB"] = "true" if use_cloud else "false"
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        db_type = "Cloud" if use_cloud else "Local"
        print(f"✅ {db_type} DB connected successfully! PostgreSQL version: {version[0]}")
        cursor.close()
        conn.close()
    except Exception as e:
        db_type = "Cloud" if use_cloud else "Local"
        print(f"❌ {db_type} DB connection failed: {e}")

# Test both connections
#test_connection(use_cloud=True)
#test_connection(use_cloud=False)
