#!/usr/bin/env python3
"""
Migration runner for User table improvements
Run this script to apply the database schema changes
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.database.database_connection import get_connection
import psycopg2

def run_migration():
    """Run the User table improvement migration"""
    print("🚀 Starting User table migration...")
    
    # Read the migration SQL file
    migration_file = os.path.join(os.path.dirname(__file__), "001_improve_user_table.sql")
    
    if not os.path.exists(migration_file):
        print("❌ Migration file not found!")
        return False
    
    with open(migration_file, 'r') as f:
        migration_sql = f.read()
    
    conn = get_connection()
    if not conn:
        print("❌ Failed to connect to database!")
        return False
    
    try:
        print("📋 Executing migration SQL...")
        with conn.cursor() as cur:
            # Execute the entire migration as one transaction
            cur.execute(migration_sql)
        
        conn.commit()
        print("✅ Migration completed successfully!")
        
        # Verify the changes
        print("🔍 Verifying schema changes...")
        with conn.cursor() as cur:
            # Check if enum type was created
            cur.execute("SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_role_enum');")
            enum_exists = cur.fetchone()[0]
            
            # Check table structure
            cur.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'User' 
                ORDER BY ordinal_position;
            """)
            columns = cur.fetchall()
            
            print(f"✅ Enum type created: {enum_exists}")
            print("📊 New table structure:")
            for col_name, col_type in columns:
                print(f"   - {col_name}: {col_type}")
        
        return True
        
    except psycopg2.Error as e:
        conn.rollback()
        print(f"❌ Migration failed: {e}")
        return False
    
    except Exception as e:
        conn.rollback()
        print(f"❌ Unexpected error: {e}")
        return False
    
    finally:
        conn.close()

def rollback_migration():
    """Rollback the migration (for development purposes)"""
    print("⚠️  Rolling back User table migration...")
    
    rollback_sql = """
    BEGIN;
    
    -- Drop triggers and functions
    DROP TRIGGER IF EXISTS update_user_updated_at ON "User";
    DROP FUNCTION IF EXISTS update_updated_at_column();
    
    -- Drop indexes
    DROP INDEX IF EXISTS idx_user_registration_date;
    DROP INDEX IF EXISTS idx_user_role;
    DROP INDEX IF EXISTS idx_user_created_at;
    
    -- Add back old columns with temporary names
    ALTER TABLE "User" 
    ADD COLUMN registration_date_old TIMESTAMP,
    ADD COLUMN role_old JSON;
    
    -- Migrate data back
    UPDATE "User" 
    SET 
        registration_date_old = (registration_date::text || ' ' || registration_time::text)::TIMESTAMP,
        role_old = to_json(role::text);
    
    -- Drop new columns
    ALTER TABLE "User" 
    DROP COLUMN IF EXISTS registration_date,
    DROP COLUMN IF EXISTS registration_time,
    DROP COLUMN IF EXISTS role,
    DROP COLUMN IF EXISTS created_at,
    DROP COLUMN IF EXISTS updated_at;
    
    -- Rename old columns back
    ALTER TABLE "User" 
    RENAME COLUMN registration_date_old TO registration_date;
    ALTER TABLE "User" 
    RENAME COLUMN role_old TO role;
    
    -- Convert profile_data back to JSON
    ALTER TABLE "User" 
    ALTER COLUMN profile_data TYPE JSON USING profile_data::JSON;
    
    -- Drop enum type
    DROP TYPE IF EXISTS user_role_enum;
    
    COMMIT;
    """
    
    conn = get_connection()
    if not conn:
        print("❌ Failed to connect to database!")
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute(rollback_sql)
        conn.commit()
        print("✅ Rollback completed successfully!")
        return True
        
    except psycopg2.Error as e:
        conn.rollback()
        print(f"❌ Rollback failed: {e}")
        return False
    
    finally:
        conn.close()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--rollback":
        rollback_migration()
    else:
        run_migration()