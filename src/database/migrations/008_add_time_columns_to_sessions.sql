-- Add start_time and end_time columns to sessions table
ALTER TABLE sessions 
    ADD COLUMN IF NOT EXISTS start_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    ADD COLUMN IF NOT EXISTS end_time timestamp without time zone;
