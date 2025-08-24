-- Rename created_at to timestamp in system_feedback table
ALTER TABLE system_feedback RENAME COLUMN created_at TO timestamp;
