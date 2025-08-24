-- Remove username field from users table
ALTER TABLE users DROP COLUMN username;

-- Remove username index
DROP INDEX IF EXISTS idx_users_username;
