-- Migration: Improve User table structure
-- Date: 2025-06-14
-- Description: Split registration_date, improve role field, add audit fields

BEGIN;

-- Step 1: Create enum type for user roles
CREATE TYPE user_role_enum AS ENUM ('user', 'coach', 'admin');

-- Step 2: Add new columns
ALTER TABLE "User" 
ADD COLUMN registration_date_new DATE,
ADD COLUMN registration_time_new TIME,
ADD COLUMN role_new user_role_enum DEFAULT 'user',
ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

-- Step 3: Migrate existing data
UPDATE "User" 
SET 
    registration_date_new = registration_date::DATE,
    registration_time_new = registration_date::TIME,
    role_new = CASE 
        WHEN role::text LIKE '%user%' THEN 'user'::user_role_enum
        WHEN role::text LIKE '%coach%' THEN 'coach'::user_role_enum
        WHEN role::text LIKE '%admin%' THEN 'admin'::user_role_enum
        ELSE 'user'::user_role_enum
    END,
    created_at = registration_date,
    updated_at = registration_date;

-- Step 4: Drop old columns and rename new ones
ALTER TABLE "User" 
DROP COLUMN registration_date,
DROP COLUMN role;

ALTER TABLE "User" 
RENAME COLUMN registration_date_new TO registration_date;
ALTER TABLE "User" 
RENAME COLUMN registration_time_new TO registration_time;
ALTER TABLE "User" 
RENAME COLUMN role_new TO role;

-- Step 5: Add constraints and indexes
ALTER TABLE "User" 
ALTER COLUMN registration_date SET NOT NULL,
ALTER COLUMN registration_time SET NOT NULL,
ALTER COLUMN role SET NOT NULL;

-- Step 6: Convert profile_data from JSON to JSONB for better performance
ALTER TABLE "User" 
ALTER COLUMN profile_data TYPE JSONB USING profile_data::JSONB;

-- Step 7: Add indexes for better query performance
CREATE INDEX idx_user_registration_date ON "User"(registration_date);
CREATE INDEX idx_user_role ON "User"(role);
CREATE INDEX idx_user_created_at ON "User"(created_at);

-- Step 8: Add trigger for automatic updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_user_updated_at 
    BEFORE UPDATE ON "User" 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

COMMIT;