-- Migration: Implement Comprehensive Fitness Tracking Schema
-- Date: 2025-06-14
-- Description: Update existing schema to match comprehensive fitness tracking requirements

BEGIN;

-- Step 1: Update User table to match new requirements
-- The User table was recently improved, but needs some adjustments

-- Create user_role_enum if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_role_enum') THEN
        CREATE TYPE user_role_enum AS ENUM ('admin', 'trainer', 'user');
    END IF;
END $$;

ALTER TABLE "User" 
ADD COLUMN IF NOT EXISTS first_name VARCHAR(100),
ADD COLUMN IF NOT EXISTS last_name VARCHAR(100),
ADD COLUMN IF NOT EXISTS last_login TIMESTAMP,
ADD COLUMN IF NOT EXISTS user_type user_role_enum DEFAULT 'user',
ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;

-- Migrate existing role column to user_type if needed
UPDATE "User" SET user_type = role WHERE user_type IS NULL;

-- Step 2: Create Workout table
CREATE TABLE IF NOT EXISTS Workout (
    workout_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    workout_date DATE NOT NULL,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    total_duration INTEGER, -- Duration in seconds
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES "User"(user_id) ON DELETE CASCADE
);

-- Step 3: Update Exercise table to match requirements
-- Check if Exercise table exists, if not create it
CREATE TABLE IF NOT EXISTS Exercise (
    exercise_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    target_muscles JSONB, -- Array of muscle groups: ["chest", "shoulders", "triceps"]
    instructions TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Update existing Exercise table structure if it exists
ALTER TABLE Exercise 
ADD COLUMN IF NOT EXISTS target_muscles JSONB,
ADD COLUMN IF NOT EXISTS instructions TEXT,
ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

-- Step 4: Update Session table to include workout relationship
-- Add workout_id and other missing fields
ALTER TABLE Session 
ADD COLUMN IF NOT EXISTS workout_id INTEGER,
ADD COLUMN IF NOT EXISTS session_order INTEGER, -- Order of exercise within workout (1, 2, 3...)
ADD COLUMN IF NOT EXISTS planned_reps INTEGER,
ADD COLUMN IF NOT EXISTS actual_reps INTEGER,
ADD COLUMN IF NOT EXISTS session_status VARCHAR(20) DEFAULT 'in_progress',
ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

-- Create session_status enum for better data integrity
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'session_status_enum') THEN
        CREATE TYPE session_status_enum AS ENUM ('completed', 'failed', 'skipped', 'in_progress');
    END IF;
END $$;

-- Update session_status column to use enum carefully
DO $$
BEGIN
    -- Check if session_status column exists and what type it is
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'session' AND column_name = 'session_status') THEN
        -- First drop any default constraint
        ALTER TABLE Session ALTER COLUMN session_status DROP DEFAULT;
        -- Then convert existing data to enum
        ALTER TABLE Session ALTER COLUMN session_status TYPE session_status_enum 
        USING CASE 
            WHEN session_status IN ('completed', 'failed', 'skipped', 'in_progress') 
            THEN session_status::session_status_enum 
            ELSE 'in_progress'::session_status_enum 
        END;
        -- Set new default
        ALTER TABLE Session ALTER COLUMN session_status SET DEFAULT 'in_progress'::session_status_enum;
    END IF;
END $$;

-- Add foreign key constraint for workout_id
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints 
                   WHERE constraint_name = 'fk_session_workout' AND table_name = 'session') THEN
        ALTER TABLE Session 
        ADD CONSTRAINT fk_session_workout 
        FOREIGN KEY (workout_id) REFERENCES Workout(workout_id) ON DELETE CASCADE;
    END IF;
END $$;

-- Step 5: Update SessionDetails table structure
-- Add new columns and handle column renames properly
DO $$ 
BEGIN
    -- Add rep_number column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'sessiondetails' AND column_name = 'rep_number') THEN
        ALTER TABLE SessionDetails ADD COLUMN rep_number INTEGER;
        -- Migrate data from rep_num to rep_number
        UPDATE SessionDetails SET rep_number = rep_num WHERE rep_num IS NOT NULL;
    END IF;
    
    -- Add is_correct_form column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'sessiondetails' AND column_name = 'is_correct_form') THEN
        ALTER TABLE SessionDetails ADD COLUMN is_correct_form BOOLEAN DEFAULT FALSE;
        -- Migrate data from is_correct to is_correct_form
        UPDATE SessionDetails SET is_correct_form = is_correct WHERE is_correct IS NOT NULL;
    END IF;
END $$;

-- Step 6: Update SystemFeedback table structure
-- Add feedback_type enum and related_rep column
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'feedback_type_enum') THEN
        CREATE TYPE feedback_type_enum AS ENUM ('form_correction', 'encouragement', 'warning', 'completion', 'critical');
    END IF;
END $$;

ALTER TABLE SystemFeedback 
ADD COLUMN IF NOT EXISTS feedback_type feedback_type_enum DEFAULT 'form_correction',
ADD COLUMN IF NOT EXISTS related_rep INTEGER; -- Which rep this feedback relates to (optional)

-- Step 7: Update Comment table structure
-- Add comment_text column and migrate data if needed
DO $$ 
BEGIN
    -- Add comment_text column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'comment' AND column_name = 'comment_text') THEN
        ALTER TABLE Comment ADD COLUMN comment_text TEXT;
        -- Migrate data from comment to comment_text if comment column exists
        IF EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'comment' AND column_name = 'comment') THEN
            UPDATE Comment SET comment_text = comment WHERE comment IS NOT NULL;
        END IF;
    END IF;
END $$;

-- Step 8: Add performance indexes for all tables
-- User table indexes
CREATE INDEX IF NOT EXISTS idx_user_email ON "User"(email);
CREATE INDEX IF NOT EXISTS idx_user_username ON "User"(username);
CREATE INDEX IF NOT EXISTS idx_user_active ON "User"(is_active);

-- Workout table indexes
CREATE INDEX IF NOT EXISTS idx_workout_user_id ON Workout(user_id);
CREATE INDEX IF NOT EXISTS idx_workout_date ON Workout(workout_date);

-- Exercise table indexes
CREATE INDEX IF NOT EXISTS idx_exercise_name ON Exercise(name);

-- Session table indexes
CREATE INDEX IF NOT EXISTS idx_session_workout_id ON Session(workout_id);
CREATE INDEX IF NOT EXISTS idx_session_exercise_id ON Session(exercise_id);
CREATE INDEX IF NOT EXISTS idx_session_user_id ON Session(user_id);
CREATE INDEX IF NOT EXISTS idx_session_start_time ON Session(start_time);

-- SessionDetails table indexes
CREATE INDEX IF NOT EXISTS idx_sessiondetails_session_id ON SessionDetails(session_id);
CREATE INDEX IF NOT EXISTS idx_sessiondetails_timestamp ON SessionDetails(timestamp);
CREATE INDEX IF NOT EXISTS idx_sessiondetails_rep_number ON SessionDetails(rep_number);

-- SystemFeedback table indexes
CREATE INDEX IF NOT EXISTS idx_systemfeedback_session_id ON SystemFeedback(session_id);
CREATE INDEX IF NOT EXISTS idx_systemfeedback_timestamp ON SystemFeedback(timestamp);
CREATE INDEX IF NOT EXISTS idx_systemfeedback_type ON SystemFeedback(feedback_type);

-- Comment table indexes
CREATE INDEX IF NOT EXISTS idx_comment_session_id ON Comment(session_id);
CREATE INDEX IF NOT EXISTS idx_comment_user_id ON Comment(user_id);
CREATE INDEX IF NOT EXISTS idx_comment_timestamp ON Comment(timestamp);

-- Step 9: Add updated_at triggers for tables that need them
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers to tables that have updated_at columns
DROP TRIGGER IF EXISTS update_workout_updated_at ON Workout;
CREATE TRIGGER update_workout_updated_at 
    BEFORE UPDATE ON Workout 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_exercise_updated_at ON Exercise;
CREATE TRIGGER update_exercise_updated_at 
    BEFORE UPDATE ON Exercise 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_session_updated_at ON Session;
CREATE TRIGGER update_session_updated_at 
    BEFORE UPDATE ON Session 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Step 10: Add constraints for data integrity
-- Ensure session_order is positive
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints 
                   WHERE constraint_name = 'chk_session_order_positive' AND table_name = 'session') THEN
        ALTER TABLE Session 
        ADD CONSTRAINT chk_session_order_positive 
        CHECK (session_order > 0);
    END IF;
END $$;

-- Ensure planned_reps and actual_reps are non-negative
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints 
                   WHERE constraint_name = 'chk_planned_reps_non_negative' AND table_name = 'session') THEN
        ALTER TABLE Session 
        ADD CONSTRAINT chk_planned_reps_non_negative 
        CHECK (planned_reps >= 0);
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints 
                   WHERE constraint_name = 'chk_actual_reps_non_negative' AND table_name = 'session') THEN
        ALTER TABLE Session 
        ADD CONSTRAINT chk_actual_reps_non_negative 
        CHECK (actual_reps >= 0);
    END IF;
END $$;

-- Ensure rep_number is positive
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints 
                   WHERE constraint_name = 'chk_rep_number_positive' AND table_name = 'sessiondetails') THEN
        ALTER TABLE SessionDetails 
        ADD CONSTRAINT chk_rep_number_positive 
        CHECK (rep_number > 0);
    END IF;
END $$;

-- Ensure incorrect_duration is non-negative
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints 
                   WHERE constraint_name = 'chk_incorrect_duration_non_negative' AND table_name = 'sessiondetails') THEN
        ALTER TABLE SessionDetails 
        ADD CONSTRAINT chk_incorrect_duration_non_negative 
        CHECK (incorrect_duration >= 0);
    END IF;
END $$;

COMMIT;