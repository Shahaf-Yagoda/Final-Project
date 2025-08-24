#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import subprocess
import psycopg2
from src.database.database_connection import get_connection

def run_sql_command(command):
    """Run a SQL command using psql."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(command)
            conn.commit()
            return True
    except Exception as e:
        print(f"Error executing SQL: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def update_database():
    """Update the database schema to the latest version."""
    print("Starting database update...")

    # 1. Drop triggers and enums first
    print("\n1. Removing old triggers and enums...")
    triggers_sql = """
    DROP TRIGGER IF EXISTS update_users_updated_at ON users;
    DROP TRIGGER IF EXISTS update_workouts_updated_at ON workouts;
    DROP TRIGGER IF EXISTS update_sessions_updated_at ON sessions;
    DROP TYPE IF EXISTS user_role_enum CASCADE;
    DROP TYPE IF EXISTS session_status_enum CASCADE;
    DROP TYPE IF EXISTS feedback_type_enum CASCADE;
    """
    if run_sql_command(triggers_sql):
        print("✓ Successfully removed old triggers and enums")
    else:
        print("⚠️ Failed to remove old triggers and enums")

    # 2. Run migrations in order
    migrations = [
        ("001_improve_user_table.sql", "Improving user table"),
        ("002_implement_comprehensive_schema.sql", "Implementing comprehensive schema"),
        ("003_remove_username_field.sql", "Removing username field"),
        ("004_update_users_fields.sql", "Updating users fields"),
        ("005_update_exercise_fields.sql", "Updating exercise fields"),
        ("006_update_workouts_table.sql", "Updating workouts table"),
        ("007_update_sessions_table.sql", "Updating sessions table"),
        ("008_add_time_columns_to_sessions.sql", "Adding time columns to sessions"),
        ("009_rename_created_at_to_timestamp.sql", "Renaming created_at to timestamp"),
        ("010_update_system_feedback_schema.sql", "Updating system_feedback schema")
    ]

    print("\n2. Running migrations...")
    migrations_dir = os.path.join("src", "database", "migrations")
    
    for migration_file, description in migrations:
        print(f"\nRunning migration: {description}")
        migration_path = os.path.join(migrations_dir, migration_file)
        
        try:
            with open(migration_path, 'r') as f:
                migration_sql = f.read()
                if run_sql_command(migration_sql):
                    print(f"✓ Successfully completed: {description}")
                else:
                    print(f"⚠️ Failed: {description}")
        except FileNotFoundError:
            print(f"⚠️ Migration file not found: {migration_file}")
            continue
        except Exception as e:
            print(f"⚠️ Error in migration {migration_file}: {e}")
            continue

    # 3. Verify final schema
    print("\n3. Verifying final schema...")
    tables = ["users", "workouts", "sessions", "exercises", "system_feedback"]
    
    for table in tables:
        if run_sql_command(f"SELECT * FROM {table} LIMIT 0"):
            print(f"✓ Table {table} exists and is accessible")
        else:
            print(f"⚠️ Table {table} might have issues")

    print("\nDatabase update completed!")
    print("\nIf you see any warnings above, please contact the development team.")
    print("If all checkmarks are green (✓), your database is ready to use!")

if __name__ == "__main__":
    # Add the project root to Python path
    project_root = str(Path(__file__).parent)
    sys.path.insert(0, project_root)
    
    try:
        update_database()
    except Exception as e:
        print(f"\n❌ Error updating database: {e}")
        print("\nPlease make sure:")
        print("1. Your database connection settings are correct in .env")
        print("2. PostgreSQL is running")
        print("3. You have the right permissions")
        sys.exit(1)
