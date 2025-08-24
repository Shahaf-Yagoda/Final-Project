#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.database.database_connection import get_connection

def run_migration():
    """Run the final migration to set the sessions table schema."""
    try:
        conn = get_connection()
        if conn is None:
            print("Failed to connect to database")
            return False

        cursor = conn.cursor()
        
        # Read and execute the migration SQL
        migration_path = os.path.join(os.path.dirname(__file__), 'final_sessions_schema.sql')
        with open(migration_path, 'r') as f:
            migration_sql = f.read()
            
        print("Running final migration to set sessions table schema...")
        cursor.execute(migration_sql)
        conn.commit()
        
        cursor.close()
        conn.close()
        
        print("✓ Migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error running migration: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False

if __name__ == "__main__":
    run_migration()
