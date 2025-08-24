-- Drop the trigger first
DROP TRIGGER IF EXISTS update_sessions_updated_at ON sessions;

-- Drop existing columns that we don't need
ALTER TABLE sessions 
    DROP COLUMN IF EXISTS user_id,
    DROP COLUMN IF EXISTS session_date,
    DROP COLUMN IF EXISTS session_time,
    DROP COLUMN IF EXISTS calories_burned,
    DROP COLUMN IF EXISTS notes,
    DROP COLUMN IF EXISTS performance_score,
    DROP COLUMN IF EXISTS form_accuracy,
    DROP COLUMN IF EXISTS created_at,
    DROP COLUMN IF EXISTS updated_at;

-- Rename columns to match the model
ALTER TABLE sessions 
    RENAME COLUMN reps TO planned_reps;

-- Add new columns
ALTER TABLE sessions 
    ADD COLUMN IF NOT EXISTS actual_reps integer DEFAULT 0,
    ADD COLUMN IF NOT EXISTS session_order integer DEFAULT 1;
