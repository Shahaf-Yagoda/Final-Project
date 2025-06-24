#!/usr/bin/env python3
"""
Simple Database Setup Script for Right Motion Fitness App

This script creates the database tables schema for PostgreSQL.
Usage: python setup_database.py
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from database.database_connection import get_connection


def setup_database():
    """Create all database tables and initial data."""
    
    # Complete schema SQL
    sql_commands = [
        # Create ENUM types
        "CREATE TYPE user_role_enum AS ENUM ('admin', 'premium', 'basic');",
        "CREATE TYPE session_status_enum AS ENUM ('active', 'completed', 'paused', 'cancelled');",
        "CREATE TYPE feedback_type_enum AS ENUM ('form_correction', 'motivation', 'progress', 'warning', 'achievement');",
        
        # Create tables
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id SERIAL PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            username VARCHAR(100) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            registration_date DATE DEFAULT CURRENT_DATE,
            registration_time TIME DEFAULT CURRENT_TIME,
            profile_data JSONB DEFAULT '{}',
            user_type user_role_enum DEFAULT 'basic',
            first_name VARCHAR(100),
            last_name VARCHAR(100),
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        
        """
        CREATE TABLE IF NOT EXISTS exercises (
            exercise_id SERIAL PRIMARY KEY,
            exercise_name VARCHAR(255) UNIQUE NOT NULL,
            category VARCHAR(100),
            target_muscles JSONB DEFAULT '[]',
            instructions TEXT,
            difficulty_level INTEGER CHECK (difficulty_level BETWEEN 1 AND 5),
            equipment_needed TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        
        """
        CREATE TABLE IF NOT EXISTS workouts (
            workout_id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
            workout_name VARCHAR(255) NOT NULL,
            description TEXT,
            target_muscles JSONB DEFAULT '[]',
            difficulty_level INTEGER CHECK (difficulty_level BETWEEN 1 AND 5),
            estimated_duration INTEGER,
            is_template BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
            workout_id INTEGER REFERENCES workouts(workout_id) ON DELETE SET NULL,
            exercise_id INTEGER REFERENCES exercises(exercise_id) ON DELETE CASCADE,
            session_date DATE DEFAULT CURRENT_DATE,
            session_time TIME DEFAULT CURRENT_TIME,
            duration INTEGER,
            reps INTEGER DEFAULT 0,
            sets INTEGER DEFAULT 1,
            calories_burned DECIMAL(6,2),
            session_status session_status_enum DEFAULT 'active',
            video_path VARCHAR(500),
            notes TEXT,
            performance_score DECIMAL(5,2),
            form_accuracy DECIMAL(5,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        
        """
        CREATE TABLE IF NOT EXISTS session_details (
            detail_id SERIAL PRIMARY KEY,
            session_id INTEGER REFERENCES sessions(session_id) ON DELETE CASCADE,
            rep_number INTEGER,
            timestamp_in_session DECIMAL(10,3),
            pose_keypoints JSONB,
            form_features JSONB,
            form_score DECIMAL(5,2),
            is_correct_form BOOLEAN,
            feedback_message TEXT,
            angles JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        
        """
        CREATE TABLE IF NOT EXISTS system_feedback (
            feedback_id SERIAL PRIMARY KEY,
            session_id INTEGER REFERENCES sessions(session_id) ON DELETE CASCADE,
            user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
            feedback_type feedback_type_enum NOT NULL,
            message TEXT NOT NULL,
            severity_level INTEGER CHECK (severity_level BETWEEN 1 AND 5),
            is_automated BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        
        """
        CREATE TABLE IF NOT EXISTS comments (
            comment_id SERIAL PRIMARY KEY,
            session_id INTEGER REFERENCES sessions(session_id) ON DELETE CASCADE,
            user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
            comment_text TEXT NOT NULL,
            is_public BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        
        # Create indexes
        "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);",
        "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);",
        "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);",
        "CREATE INDEX IF NOT EXISTS idx_sessions_exercise_id ON sessions(exercise_id);",
        "CREATE INDEX IF NOT EXISTS idx_session_details_session_id ON session_details(session_id);",
        
        # Create update trigger function
        """
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        """,
        
        # Create triggers
        "CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();",
        "CREATE TRIGGER update_workouts_updated_at BEFORE UPDATE ON workouts FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();",
        "CREATE TRIGGER update_exercises_updated_at BEFORE UPDATE ON exercises FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();",
        "CREATE TRIGGER update_sessions_updated_at BEFORE UPDATE ON sessions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();",
        "CREATE TRIGGER update_comments_updated_at BEFORE UPDATE ON comments FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();",
        
        # Insert default exercises
        """
        INSERT INTO exercises (exercise_name, category, target_muscles, instructions, difficulty_level, equipment_needed) 
        VALUES 
        ('lunge', 'Lower Body', '["quadriceps", "glutes", "hamstrings", "calves"]', 
         'Stand with feet hip-width apart. Step forward with one leg, lowering hips until both knees are bent at 90 degrees. Push back to starting position.', 
         2, 'None'),
        ('overhead_press', 'Upper Body', '["shoulders", "triceps", "upper_chest"]', 
         'Stand with feet shoulder-width apart. Hold weights at shoulder level. Press weights overhead until arms are fully extended. Lower back to starting position.', 
         3, 'Dumbbells or Barbell'),
        ('plank', 'Core', '["core", "shoulders", "glutes"]', 
         'Start in push-up position. Lower to forearms, keeping body in straight line from head to heels. Hold position.', 
         2, 'None')
        ON CONFLICT (exercise_name) DO NOTHING;
        """
    ]
    
    try:
        conn = get_connection()
        if not conn:
            print("❌ Failed to connect to database")
            return False
            
        cursor = conn.cursor()
        
        print("🚀 Setting up database schema...")
        
        for i, sql in enumerate(sql_commands):
            try:
                cursor.execute(sql)
                print(f"✓ Executed command {i+1}/{len(sql_commands)}")
            except Exception as e:
                # Skip if type/table already exists
                if "already exists" in str(e).lower():
                    print(f"⚠️  Command {i+1} skipped (already exists)")
                    continue
                else:
                    raise e
        
        conn.commit()
        print("✅ Database schema created successfully!")
        
        # Verify tables
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        print(f"\n📊 Created {len(tables)} tables:")
        for table in tables:
            print(f"   • {table[0]}")
            
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False


if __name__ == "__main__":
    print("Right Motion Fitness App - Database Setup")
    print("=" * 45)
    
    success = setup_database()
    
    if success:
        print("\n🎉 Database setup completed successfully!")
        print("You can now run the application.")
    else:
        print("\n💥 Database setup failed!")
        sys.exit(1)