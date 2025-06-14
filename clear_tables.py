#!/usr/bin/env python3
"""
Clear Session, SessionDetails and SystemFeedback tables
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.database.database_connection import get_connection

def clear_tables():
    print("🧹 Clearing Session, SessionDetails and SystemFeedback tables...")
    
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Get counts before clearing
            cur.execute("SELECT COUNT(*) FROM SessionDetails;")
            sessiondetails_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM SystemFeedback;")
            systemfeedback_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM Session;")
            session_count = cur.fetchone()[0]
            
            print(f"📊 Found {session_count} Session records")
            print(f"📊 Found {sessiondetails_count} SessionDetails records")
            print(f"📊 Found {systemfeedback_count} SystemFeedback records")
            
            # Clear the tables (order matters due to foreign keys)
            # Clear child tables first
            cur.execute("DELETE FROM SessionDetails;")
            sessiondetails_deleted = cur.rowcount
            
            cur.execute("DELETE FROM SystemFeedback;")
            systemfeedback_deleted = cur.rowcount
            
            # Clear parent table last
            cur.execute("DELETE FROM Session;")
            session_deleted = cur.rowcount
            
            # Reset sequences to start from 1
            cur.execute("ALTER SEQUENCE session_session_id_seq RESTART WITH 1;")
            cur.execute("ALTER SEQUENCE sessiondetails_detail_id_seq RESTART WITH 1;")
            cur.execute("ALTER SEQUENCE systemfeedback_feedback_id_seq RESTART WITH 1;")
            
        conn.commit()
        print(f"✅ Cleared {session_deleted} Session records")
        print(f"✅ Cleared {sessiondetails_deleted} SessionDetails records")
        print(f"✅ Cleared {systemfeedback_deleted} SystemFeedback records")
        print("✅ Reset ID sequences to start from 1")
        print("🎉 All session tables cleared successfully!")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error clearing tables: {e}")
        return False
        
    finally:
        conn.close()
    
    return True

if __name__ == "__main__":
    clear_tables()