# Database Update Instructions

This document explains how to update your database to match the latest schema.

## Prerequisites

1. Make sure you have:
   - Python 3.x installed
   - PostgreSQL installed and running
   - Virtual environment activated
   - All project dependencies installed (`pip install -r requirements.txt`)

2. Ensure your `.env` file has the correct database connection settings:
   ```
   DB_NAME_LOCAL=right_motion
   DB_USER=your_username
   DB_PASSWORD=your_password
   DB_HOST=localhost
   DB_PORT=5432
   ```

## Running the Update

1. Open a terminal in the project root directory

2. Activate your virtual environment:
   ```bash
   source venv/bin/activate  # On Unix/macOS
   # OR
   .\venv\Scripts\activate  # On Windows
   ```

3. Run the update script:
   ```bash
   python3 update_database.py
   ```

4. Watch the output:
   - ✓ checkmarks indicate successful steps
   - ⚠️ warnings indicate potential issues
   - ❌ errors indicate failed steps

## What the Update Does

The script will:
1. Remove old triggers and enums
2. Run all necessary migrations in the correct order
3. Verify the final schema

## Schema Changes

The update will ensure your database has the following schema:

### Users Table
- Removed: username, registration_time, created_at, updated_at
- Added: first_name, last_name
- Modified: registration_date to timestamp

### Workouts Table
- Removed: workout_name, description, target_muscles, difficulty_level, estimated_duration, is_template
- Added: workout_date, start_time, end_time, total_duration

### Sessions Table
- Added: session_order, start_time, end_time
- Renamed: reps to planned_reps
- Added: actual_reps
- Modified: duration field

### System Feedback Table
- Simplified schema with: feedback_id, session_id, timestamp, feedback_type, message, related_rep

## Troubleshooting

If you encounter any errors:

1. Check your database connection:
   ```sql
   psql -d right_motion -c "SELECT version();"
   ```

2. Make sure you have the right permissions:
   ```sql
   psql -d right_motion -c "\du"
   ```

3. Verify your virtual environment has all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. If you see specific table errors, you might need to clean up your database first:
   ```sql
   DROP SCHEMA public CASCADE;
   CREATE SCHEMA public;
   ```

## Need Help?

If you encounter any issues:
1. Take a screenshot of the error
2. Note which step failed (look for ⚠️ or ❌)
3. Contact the development team

## After the Update

After successfully updating:
1. Restart your application
2. Try logging in
3. Test the exercise tracking features
4. Verify feedback is being saved correctly

If everything works, you're all set! If not, contact the development team.
