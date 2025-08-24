-- Remove registration_time, created_at, and updated_at fields from users table
ALTER TABLE users DROP COLUMN registration_time;
ALTER TABLE users DROP COLUMN created_at;
ALTER TABLE users DROP COLUMN updated_at;

-- Drop the trigger since we no longer have updated_at
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
