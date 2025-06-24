#!/usr/bin/env python3
"""
Database Schema Builder for Right Motion Fitness App

This script creates the complete PostgreSQL database schema for the fitness tracking application.
It includes all tables, enums, indexes, constraints, and triggers.
"""

import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from database.database_connection import get_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_database_schema():
    """Create the complete database schema for the fitness tracking application."""
    
    # SQL to create the complete schema
    schema_sql = """
    -- Create ENUM types
    CREATE TYPE user_role_enum AS ENUM ('admin', 'premium', 'basic');
    CREATE TYPE session_status_enum AS ENUM ('active', 'completed', 'paused', 'cancelled');
    CREATE TYPE feedback_type_enum AS ENUM ('form_correction', 'motivation', 'progress', 'warning', 'achievement');

    -- Create User table
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

    -- Create Workout table
    CREATE TABLE IF NOT EXISTS workouts (
        workout_id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
        workout_name VARCHAR(255) NOT NULL,
        description TEXT,
        target_muscles JSONB DEFAULT '[]',
        difficulty_level INTEGER CHECK (difficulty_level BETWEEN 1 AND 5),
        estimated_duration INTEGER, -- in minutes
        is_template BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Create Exercise table
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

    -- Create Session table
    CREATE TABLE IF NOT EXISTS sessions (
        session_id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
        workout_id INTEGER REFERENCES workouts(workout_id) ON DELETE SET NULL,
        exercise_id INTEGER REFERENCES exercises(exercise_id) ON DELETE CASCADE,
        session_date DATE DEFAULT CURRENT_DATE,
        session_time TIME DEFAULT CURRENT_TIME,
        duration INTEGER, -- in seconds
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

    -- Create SessionDetails table
    CREATE TABLE IF NOT EXISTS session_details (
        detail_id SERIAL PRIMARY KEY,
        session_id INTEGER REFERENCES sessions(session_id) ON DELETE CASCADE,
        rep_number INTEGER,
        timestamp_in_session DECIMAL(10,3), -- seconds from session start
        pose_keypoints JSONB,
        form_features JSONB,
        form_score DECIMAL(5,2),
        is_correct_form BOOLEAN,
        feedback_message TEXT,
        angles JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Create SystemFeedback table
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

    -- Create Comment table
    CREATE TABLE IF NOT EXISTS comments (
        comment_id SERIAL PRIMARY KEY,
        session_id INTEGER REFERENCES sessions(session_id) ON DELETE CASCADE,
        user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
        comment_text TEXT NOT NULL,
        is_public BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Create indexes for performance optimization
    CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
    CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
    CREATE INDEX IF NOT EXISTS idx_users_user_type ON users(user_type);
    CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
    CREATE INDEX IF NOT EXISTS idx_sessions_exercise_id ON sessions(exercise_id);
    CREATE INDEX IF NOT EXISTS idx_sessions_date ON sessions(session_date);
    CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(session_status);
    CREATE INDEX IF NOT EXISTS idx_session_details_session_id ON session_details(session_id);
    CREATE INDEX IF NOT EXISTS idx_session_details_rep_number ON session_details(rep_number);
    CREATE INDEX IF NOT EXISTS idx_workouts_user_id ON workouts(user_id);
    CREATE INDEX IF NOT EXISTS idx_exercises_name ON exercises(exercise_name);
    CREATE INDEX IF NOT EXISTS idx_exercises_category ON exercises(category);
    CREATE INDEX IF NOT EXISTS idx_system_feedback_session_id ON system_feedback(session_id);
    CREATE INDEX IF NOT EXISTS idx_system_feedback_user_id ON system_feedback(user_id);
    CREATE INDEX IF NOT EXISTS idx_system_feedback_type ON system_feedback(feedback_type);
    CREATE INDEX IF NOT EXISTS idx_comments_session_id ON comments(session_id);
    CREATE INDEX IF NOT EXISTS idx_comments_user_id ON comments(user_id);

    -- Create triggers for automatic updated_at timestamps
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = CURRENT_TIMESTAMP;
        RETURN NEW;
    END;
    $$ language 'plpgsql';

    CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

    CREATE TRIGGER update_workouts_updated_at BEFORE UPDATE ON workouts
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

    CREATE TRIGGER update_exercises_updated_at BEFORE UPDATE ON exercises
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

    CREATE TRIGGER update_sessions_updated_at BEFORE UPDATE ON sessions
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

    CREATE TRIGGER update_comments_updated_at BEFORE UPDATE ON comments
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """

    # Exercise seed data
    exercises_seed_sql = """
    -- Insert default exercises
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

    try:
        # Get database connection
        conn = get_connection()
        if conn is None:
            logger.error("Failed to connect to database")
            return False

        cursor = conn.cursor()
        
        logger.info("Creating database schema...")
        
        # Execute schema creation
        cursor.execute(schema_sql)
        logger.info("✓ Database schema created successfully")
        
        # Insert seed data
        cursor.execute(exercises_seed_sql)
        logger.info("✓ Exercise seed data inserted")
        
        # Commit changes
        conn.commit()
        logger.info("✓ Changes committed to database")
        
        # Verify tables were created
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        logger.info(f"✓ Created {len(tables)} tables:")
        for table in tables:
            logger.info(f"  - {table[0]}")
        
        cursor.close()
        conn.close()
        
        logger.info("Database schema creation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error creating database schema: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False


def drop_database_schema():
    """Drop all tables and types (use with caution)."""
    
    drop_sql = """
    -- Drop tables in reverse order of dependencies
    DROP TABLE IF EXISTS comments CASCADE;
    DROP TABLE IF EXISTS system_feedback CASCADE;
    DROP TABLE IF EXISTS session_details CASCADE;
    DROP TABLE IF EXISTS sessions CASCADE;
    DROP TABLE IF EXISTS exercises CASCADE;
    DROP TABLE IF EXISTS workouts CASCADE;
    DROP TABLE IF EXISTS users CASCADE;
    
    -- Drop functions
    DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE;
    
    -- Drop enums
    DROP TYPE IF EXISTS feedback_type_enum CASCADE;
    DROP TYPE IF EXISTS session_status_enum CASCADE;
    DROP TYPE IF EXISTS user_role_enum CASCADE;
    """
    
    try:
        conn = get_connection()
        if conn is None:
            logger.error("Failed to connect to database")
            return False

        cursor = conn.cursor()
        
        logger.info("Dropping existing database schema...")
        cursor.execute(drop_sql)
        conn.commit()
        
        cursor.close()
        conn.close()
        
        logger.info("✓ Database schema dropped successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error dropping database schema: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False


def main():
    """Main function to handle command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Database Schema Builder for Right Motion Fitness App')
    parser.add_argument('--drop', action='store_true', help='Drop existing schema before creating new one')
    parser.add_argument('--drop-only', action='store_true', help='Only drop the schema, do not recreate')
    
    args = parser.parse_args()
    
    try:
        if args.drop or args.drop_only:
            logger.info("WARNING: This will drop all existing data!")
            response = input("Are you sure you want to continue? (yes/no): ")
            if response.lower() != 'yes':
                logger.info("Operation cancelled")
                return
            
            if not drop_database_schema():
                logger.error("Failed to drop database schema")
                sys.exit(1)
        
        if not args.drop_only:
            if not create_database_schema():
                logger.error("Failed to create database schema")
                sys.exit(1)
        
        logger.info("Operation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()