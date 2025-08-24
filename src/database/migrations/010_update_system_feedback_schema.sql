-- Drop the table and recreate it with the exact schema needed
DROP TABLE IF EXISTS system_feedback CASCADE;

-- Create the table with the exact schema
CREATE TABLE system_feedback (
    feedback_id SERIAL PRIMARY KEY,
    session_id INTEGER,
    timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    feedback_type VARCHAR,
    message TEXT,
    related_rep INTEGER
);

-- Add index for session_id for better performance
CREATE INDEX idx_system_feedback_session_id ON system_feedback(session_id);
