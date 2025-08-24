-- Drop the trigger first
DROP TRIGGER IF EXISTS update_workouts_updated_at ON workouts;

-- Drop the check constraint
ALTER TABLE workouts DROP CONSTRAINT IF EXISTS workouts_difficulty_level_check;

-- Drop existing columns that we don't need
ALTER TABLE workouts 
    DROP COLUMN IF EXISTS workout_name,
    DROP COLUMN IF EXISTS description,
    DROP COLUMN IF EXISTS target_muscles,
    DROP COLUMN IF EXISTS difficulty_level,
    DROP COLUMN IF EXISTS estimated_duration,
    DROP COLUMN IF EXISTS is_template,
    DROP COLUMN IF EXISTS created_at,
    DROP COLUMN IF EXISTS updated_at;

-- Add new columns
ALTER TABLE workouts 
    ADD COLUMN workout_date date DEFAULT CURRENT_DATE,
    ADD COLUMN start_time timestamp without time zone,
    ADD COLUMN end_time timestamp without time zone,
    ADD COLUMN total_duration integer;
