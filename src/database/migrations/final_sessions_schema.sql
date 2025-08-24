-- Drop the table and recreate it with the exact schema we need
DROP TABLE IF EXISTS sessions CASCADE;

-- Recreate the table with the exact columns we need
CREATE TABLE sessions (
    session_id SERIAL PRIMARY KEY,
    workout_id INTEGER REFERENCES workouts(workout_id) ON DELETE SET NULL,
    exercise_id INTEGER REFERENCES exercises(exercise_id) ON DELETE CASCADE,
    session_order INTEGER DEFAULT 1,
    start_time TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITHOUT TIME ZONE,
    duration INTEGER,
    planned_reps INTEGER DEFAULT 0,
    actual_reps INTEGER DEFAULT 0,
    session_status session_status_enum DEFAULT 'active',
    video_path VARCHAR(500)
);

-- Recreate the indexes
CREATE INDEX idx_sessions_exercise_id ON sessions(exercise_id);
CREATE INDEX idx_sessions_status ON sessions(session_status);
