#!/usr/bin/env python3
"""
Migration runner for comprehensive fitness tracking schema
Run this script to apply the database schema changes
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.database.database_connection import get_connection
import psycopg2

def run_migration():
    """Run the comprehensive schema migration"""
    print("🚀 Starting comprehensive schema migration...")
    
    # Read the migration SQL file
    migration_file = os.path.join(os.path.dirname(__file__), "002_implement_comprehensive_schema.sql")
    
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
            # Check if all tables exist
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """)
            tables = [row[0] for row in cur.fetchall()]
            
            # Check if enum types were created
            cur.execute("""
                SELECT typname 
                FROM pg_type 
                WHERE typcategory = 'E' 
                ORDER BY typname;
            """)
            enums = [row[0] for row in cur.fetchall()]
            
            print("📊 Database tables:")
            for table in tables:
                print(f"   ✅ {table}")
                
            print("📋 Enum types:")
            for enum in enums:
                print(f"   ✅ {enum}")
        
        return True
        
    except psycopg2.Error as e:
        conn.rollback()
        print(f"❌ Migration failed: {e}")
        print("💡 This might be due to existing schema elements. Checking what exists...")
        
        # Check what already exists
        try:
            with conn.cursor() as cur:
                # Check existing enums
                cur.execute("SELECT typname FROM pg_type WHERE typcategory = 'E';")
                existing_enums = [row[0] for row in cur.fetchall()]
                if existing_enums:
                    print(f"📋 Existing enums: {', '.join(existing_enums)}")
                
                # Check existing tables
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_type = 'BASE TABLE';
                """)
                existing_tables = [row[0] for row in cur.fetchall()]
                print(f"📊 Existing tables: {', '.join(existing_tables)}")
                
        except Exception as check_error:
            print(f"❌ Could not check existing schema: {check_error}")
        
        return False
    
    except Exception as e:
        conn.rollback()
        print(f"❌ Unexpected error: {e}")
        return False
    
    finally:
        conn.close()

if __name__ == "__main__":
    run_migration()