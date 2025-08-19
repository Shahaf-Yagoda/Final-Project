# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fitness tracking application called "Right Motion" that uses computer vision and MediaPipe to analyze exercise form in real-time. The system provides feedback on posture and counts repetitions for exercises like lunges, overhead press, and planks.

### Core Architecture

- **Frontend**: Streamlit web application (`src/app/app.py`) with video streaming capability
- **Backend**: Flask video streamer (`src/app/video_streamer.py`) for real-time pose analysis
- **Processing**: Exercise form analysis using MediaPipe pose estimation (`src/processing/forms_check.py`)
- **Database**: PostgreSQL with support for both local and cloud instances
- **User Management**: Authentication system with user profiles and session tracking

### Key Components

1. **Video Analysis Pipeline**: 
   - MediaPipe pose detection → form analysis → feedback generation → database storage
   - Supports both live tracking and uploaded video analysis

2. **Exercise Form Checking**:
   - Modular form checkers for different exercises in `src/processing/forms_check.py`
   - Real-time angle calculations and posture validation
   - Audio feedback system with cooldown timers

3. **Database Schema**:
   - Users, sessions, session details, and system feedback tables
   - OOP models for User and Session management

## Common Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables for database connection
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python  # Required for streamlit compatibility

# Copy and configure environment file
cp .env.example .env  # Edit with your database settings
```

### Running the Application
```bash
# IMPORTANT: Always activate virtual environment first
source venv/bin/activate

# Option 1: Use startup script (recommended)
./start_app.sh

# Option 2: Manual start
streamlit run src/app/app.py

# Run standalone exercise session (command line)
python main.py --exercise lunge --user_id 1 [--no-window] [--save-keypoints]

# Test database connection
python src/database/test_db.py
```

### Running Tests
```bash
# Run all tests
python tests/run_tests.py

# Run specific test module
python tests/run_tests.py test_exercise_session
python tests/run_tests.py test_forms_check
python tests/run_tests.py test_video_streamer
python tests/run_tests.py test_integration
```

### Database Operations
- Connection management: `src/database/database_connection.py`
- Use `get_connection()` for database access
- Supports both local and cloud PostgreSQL via environment variables
- Session data automatically saved during live tracking

### Video Processing
- Live sessions are recorded to `/tmp/` during processing
- Final videos saved to `videos/` directory with timestamp naming
- Temporary files used for inter-process communication (reps count, session details)

## Development Notes

### Exercise Form Analysis (Object-Oriented Design)
- **BaseExerciseChecker**: Abstract base class defining the interface for all exercise checkers
- **LungeChecker**: Specialized checker for lunge exercises with front/back leg detection
- **OverheadPressChecker**: Handles overhead press with ready position detection and rep phases
- **PlankChecker**: Static exercise checker with duration tracking and timer visualization
- **Factory Pattern**: `get_exercise_checker(exercise_name)` returns appropriate checker instance
- **Backward Compatibility**: Legacy functions `check_form()` and `init_state()` still work
- State management tracks rep counting, posture correctness, and timing
- Audio feedback uses `speak_async()` with cooldown mechanisms
- Visual feedback overlays angles and messages on video frames

### Database Integration
- Use OOP Session and User classes for database operations
- Session details and feedback automatically logged during live tracking
- Support for both video analysis and live exercise modes

### Streamlit Application Flow
- Multi-page application with session state management
- Video streaming handled via Flask backend on port 5050 using **VideoStreamManager** class
- User authentication required for most features
- Session history accessible through web interface

### Object-Oriented Architecture
- **ExerciseSession** class (`main.py`): Manages complete exercise sessions with pose detection
- **VideoStreamManager** class (`video_streamer.py`): Handles video streaming, recording, and session management
- **Exercise Checker Classes** (`forms_check.py`): Polymorphic exercise form analysis
- **Type Hints**: Full typing support for better IDE integration and error catching
- **Comprehensive Test Suite**: 43 tests covering all major components with 100% pass rate

### Key File Locations
- Main entry points: `main.py` (ExerciseSession class), `src/app/app.py`
- Exercise logic: `src/processing/forms_check.py` (BaseExerciseChecker and subclasses)
- Database models: `src/database/users/user.py`, `src/database/session.py`
- Video streaming: `src/app/video_streamer.py` (VideoStreamManager class)
- Test suite: `tests/` directory with comprehensive coverage