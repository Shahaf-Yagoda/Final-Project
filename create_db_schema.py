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
    -- Create User table (singular name as per specification)
    CREATE TABLE IF NOT EXISTS "user" (
        user_id SERIAL PRIMARY KEY,
        email CHARACTER VARYING NOT NULL,
        password CHARACTER VARYING NOT NULL,
        first_name CHARACTER VARYING,
        last_name CHARACTER VARYING,
        profile_data JSON,
        registration_date TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP WITHOUT TIME ZONE,
        user_type CHARACTER VARYING,
        is_active BOOLEAN DEFAULT TRUE
    );

    -- Create Exercise table (singular name as per specification)
    CREATE TABLE IF NOT EXISTS exercise (
        exercise_id SERIAL PRIMARY KEY,
        name CHARACTER VARYING NOT NULL,
        description TEXT,
        target_muscles JSON,
        instructions TEXT
    );

    -- Create Workout table (singular name as per specification)
    CREATE TABLE IF NOT EXISTS workout (
        workout_id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES "user"(user_id) ON DELETE CASCADE,
        workout_date DATE,
        start_time TIMESTAMP WITHOUT TIME ZONE,
        end_time TIMESTAMP WITHOUT TIME ZONE,
        total_duration INTEGER
    );

    -- Create Session table (singular name as per specification)
    CREATE TABLE IF NOT EXISTS session (
        session_id SERIAL PRIMARY KEY,
        workout_id INTEGER REFERENCES workout(workout_id) ON DELETE SET NULL,
        exercise_id INTEGER REFERENCES exercise(exercise_id) ON DELETE CASCADE,
        session_order INTEGER,
        start_time TIMESTAMP WITHOUT TIME ZONE,
        end_time TIMESTAMP WITHOUT TIME ZONE,
        duration INTEGER,
        planned_reps INTEGER,
        actual_reps INTEGER,
        session_status CHARACTER VARYING,
        video_path TEXT
    );

    -- Create SessionDetails table (as per specification)
    CREATE TABLE IF NOT EXISTS sessiondetails (
        detail_id SERIAL PRIMARY KEY,
        session_id INTEGER REFERENCES session(session_id) ON DELETE CASCADE,
        rep_number INTEGER,
        timestamp TIMESTAMP WITHOUT TIME ZONE,
        features_json JSON,
        is_correct_form BOOLEAN,
        incorrect_duration DOUBLE PRECISION
    );

    -- Create SystemFeedback table (as per specification)
    CREATE TABLE IF NOT EXISTS systemfeedback (
        feedback_id SERIAL PRIMARY KEY,
        session_id INTEGER REFERENCES session(session_id) ON DELETE CASCADE,
        timestamp TIMESTAMP WITHOUT TIME ZONE,
        feedback_type CHARACTER VARYING,
        message TEXT,
        related_rep INTEGER
    );

    -- Create Comment table (singular name as per specification)
    CREATE TABLE IF NOT EXISTS comment (
        comment_id SERIAL PRIMARY KEY,
        session_id INTEGER REFERENCES session(session_id) ON DELETE CASCADE,
        user_id INTEGER REFERENCES "user"(user_id) ON DELETE CASCADE,
        timestamp TIMESTAMP WITHOUT TIME ZONE,
        comment_text TEXT
    );

    -- Create indexes for performance optimization
    CREATE INDEX IF NOT EXISTS idx_user_email ON "user"(email);
    CREATE INDEX IF NOT EXISTS idx_user_user_type ON "user"(user_type);
    CREATE INDEX IF NOT EXISTS idx_session_workout_id ON session(workout_id);
    CREATE INDEX IF NOT EXISTS idx_session_exercise_id ON session(exercise_id);
    CREATE INDEX IF NOT EXISTS idx_session_start_time ON session(start_time);
    CREATE INDEX IF NOT EXISTS idx_session_status ON session(session_status);
    CREATE INDEX IF NOT EXISTS idx_sessiondetails_session_id ON sessiondetails(session_id);
    CREATE INDEX IF NOT EXISTS idx_sessiondetails_rep_number ON sessiondetails(rep_number);
    CREATE INDEX IF NOT EXISTS idx_workout_user_id ON workout(user_id);
    CREATE INDEX IF NOT EXISTS idx_exercise_name ON exercise(name);
    CREATE INDEX IF NOT EXISTS idx_systemfeedback_session_id ON systemfeedback(session_id);
    CREATE INDEX IF NOT EXISTS idx_systemfeedback_type ON systemfeedback(feedback_type);
    CREATE INDEX IF NOT EXISTS idx_comment_session_id ON comment(session_id);
    CREATE INDEX IF NOT EXISTS idx_comment_user_id ON comment(user_id);
    """

    # Exercise seed data
    exercises_seed_sql = """
    -- Insert default exercises
    INSERT INTO exercise (name, target_muscles, instructions) 
    VALUES 
    ('lunge', '["quadriceps", "glutes", "hamstrings", "calves"]', 
     'Stand with feet hip-width apart. Step forward with one leg, lowering hips until both knees are bent at 90 degrees. Push back to starting position.'),
    ('overhead_press', '["shoulders", "triceps", "upper_chest"]', 
     'Stand with feet shoulder-width apart. Hold weights at shoulder level. Press weights overhead until arms are fully extended. Lower back to starting position.'),
    ('plank', '["core", "shoulders", "glutes"]', 
     'Start in push-up position. Lower to forearms, keeping body in straight line from head to heels. Hold position.');
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
    DROP TABLE IF EXISTS comment CASCADE;
    DROP TABLE IF EXISTS systemfeedback CASCADE;
    DROP TABLE IF EXISTS sessiondetails CASCADE;
    DROP TABLE IF EXISTS session CASCADE;
    DROP TABLE IF EXISTS exercise CASCADE;
    DROP TABLE IF EXISTS workout CASCADE;
    DROP TABLE IF EXISTS "user" CASCADE;
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