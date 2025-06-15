# Command Line Session Transaction - Comprehensive Documentation

## Transaction Overview

The **Command Line Session Transaction** handles exercise sessions executed through the command line interface using `main.py`. This transaction provides a standalone exercise tracking capability independent of the web interface, with direct video capture, pose analysis, and database storage.

### Transaction Classification
- **Priority Level**: 2 (Supporting Infrastructure)
- **Complexity**: High
- **Tables Involved**: 4 primary tables (Exercise, Session, SessionDetails, SystemFeedback)
- **Transaction Type**: Multi-table ACID compliant
- **Execution Context**: Command line with OpenCV video capture

---

## Data Flow Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Command Line    │────│  Argument Parse  │────│  ExerciseSession│
│   Arguments     │    │   & Validation   │    │  Initialization │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                Real-time Processing Pipeline                    │
│  ▪ OpenCV camera capture                                       │
│  ▪ MediaPipe pose detection                                    │
│  ▪ Exercise-specific form analysis                             │
│  ▪ Rep counting and validation                                 │
│  ▪ Video recording (optional)                                  │
│  ▪ Session data buffering                                      │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Database Transaction Layer                      │
│                                                                │
│  1. Exercise Lookup    →  Exercise Table                       │
│  2. Session Creation   →  Session Table                        │
│  3. Details Logging    →  SessionDetails Table                 │
│  4. Feedback Storage   →  SystemFeedback Table                 │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Transaction Components

### 1. **Command Line Interface**
**Location**: `main.py:279-311`

```python
def main():
    """Main function to run exercise session from command line."""
    import argparse
    parser = argparse.ArgumentParser(description="Run an exercise session with pose analysis and automatic saving")
    parser.add_argument("--exercise", required=True, help="Exercise type (lunge, press, plank)")
    parser.add_argument("--user_id", type=int, required=True, help="User ID")
    parser.add_argument("--no-window", action="store_true", help="Run without display window")
    parser.add_argument("--save-keypoints", action="store_true", help="Save keypoints to database")
    parser.add_argument("--no-video", action="store_true", help="Disable video recording")
    args = parser.parse_args()

    session = ExerciseSession(
        exercise_name=args.exercise,
        user_id=args.user_id,
        show_window=not args.no_window,
        save_keypoints=args.save_keypoints,
        save_video=not args.no_video
    )
    
    session_data = session.run()
```

**Command Line Options**:
- `--exercise`: Required exercise type (lunge, press, plank)
- `--user_id`: Required user identifier
- `--no-window`: Headless mode without video display
- `--save-keypoints`: Enable keypoint data storage
- `--no-video`: Disable video recording

**Usage Examples**:
```bash
# Basic session with video display
python main.py --exercise lunge --user_id 1

# Headless session without video recording
python main.py --exercise press --user_id 1 --no-window --no-video

# Full featured session with keypoint saving
python main.py --exercise plank --user_id 1 --save-keypoints
```

### 2. **ExerciseSession Class**
**Location**: `main.py:28-272`

```python
class ExerciseSession:
    """Manages a complete exercise session with pose detection and form analysis."""
    
    def __init__(self, exercise_name, user_id, show_window=True, save_keypoints=False, save_video=True):
        self.exercise_name = exercise_name
        self.user_id = user_id
        self.show_window = show_window
        self.save_keypoints = save_keypoints
        self.save_video = save_video
        self.state = {
            "exercise": {
                exercise_name: init_state(exercise_name)
            }
        }
        self.pose_detector = PoseDetectorFactory.create_live_exercise_detector()
        self.start_time = None
        self.end_time = None
        self.video_writer = None
        self.temp_video_path = None
        self.final_video_path = None
        self.session_details = []
        self.feedback_data = []
```

**Key Components**:
- **State Management**: Exercise-specific state tracking
- **Pose Detection**: MediaPipe pose detector optimized for live sessions
- **Video Recording**: Optional MP4 video recording with codec selection
- **Data Collection**: Session details and feedback buffering
- **Timing**: Start/end timestamp tracking

### 3. **Real-time Processing Loop**
**Location**: `main.py:152-179`

```python
try:
    with self.pose_detector as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                self.process_frame(frame, landmarks)
            
            # Write frame to video if recording
            if self.video_writer:
                self.video_writer.write(frame)

            if self.show_window:
                cv2.imshow("Pose Tracker - Press ESC to stop and save", frame)
                key = cv2.waitKey(5) & 0xFF
                if key == 27:  # ESC key
                    print("\n🛑 ESC pressed - stopping and saving session...")
                    break
                elif key == ord('q'):  # Also allow 'q' to quit
                    print("\n🛑 'Q' pressed - stopping and saving session...")
                    break
except KeyboardInterrupt:
    print("\n🛑 Ctrl+C pressed - stopping and saving session...")
    session_interrupted = True
```

**Features**:
- **Pose Detection**: Real-time MediaPipe pose landmark detection
- **Form Analysis**: Exercise-specific form checking with feedback
- **Video Recording**: Optional frame-by-frame video recording
- **Interactive Display**: OpenCV window with pose overlay (optional)
- **Graceful Exit**: ESC key, 'Q' key, or Ctrl+C to stop and save

### 4. **Session Data Collection**
**Location**: `main.py:89-109`

```python
def save_session_detail(self, landmarks, feedback, reps, current_time):
    """Save session detail for database storage using new schema."""
    # Only save if we have actual reps (> 0) to satisfy database constraint
    if reps > 0:
        detail = {
            "timestamp": current_time,
            "rep_num": reps,
            # Use new schema - no keypoints, focus on form analysis
            "features_json": {
                "form_correct": len(feedback) == 0,
                "feedback_count": len(feedback),
                "timestamp": current_time
            },
            "is_correct": len(feedback) == 0,
            "incorrect_duration": 0
        }
        self.session_details.append(detail)
    
    # Save feedback
    for msg in feedback:
        self.feedback_data.append({"timestamp": current_time, "message": msg, "related_rep": reps if reps > 0 else None})
```

**Data Structure**:
- **Session Details**: Rep-by-rep form analysis data
- **Feedback Data**: Form correction messages with timestamps
- **Features JSON**: Structured metadata for analysis
- **Rep Validation**: Only save details for completed reps

---

## Database Transaction Steps

### Step 1: Exercise Validation
**Function**: `get_exercise_id_by_name()`
**Location**: `src/database/db_utils.py`

```python
exercise_id = get_exercise_id_by_name(self.exercise_name)
if not exercise_id:
    raise ValueError(f"Exercise '{self.exercise_name}' not found")
```

**Purpose**: Validate exercise exists in database before proceeding.

### Step 2: Session Creation
**Function**: `Session.save()`
**Location**: `main.py:219-236`

```sql
INSERT INTO Session (
    user_id, exercise_id, start_time, end_time, duration,
    actual_reps, planned_reps, session_status, video_path,
    session_order, duration_sec, reps_count
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
RETURNING session_id
```

**Key Fields**:
- `user_id`: Command line argument
- `exercise_id`: From exercise lookup
- `start_time`/`end_time`: Session temporal boundaries
- `actual_reps`: Final rep count from form analysis
- `video_path`: Path to recorded video file (if enabled)
- `session_status`: 'completed' for successful sessions

### Step 3: Session Details Bulk Insert
**Function**: `SessionDetails.create()`
**Location**: `main.py:241-251`

```python
if self.session_details:
    print(f"💾 Saving {len(self.session_details)} session details...")
    for detail in self.session_details:
        SessionDetails.create(
            session_id=session_id,
            rep_number=detail["rep_num"],
            features_json=detail["features_json"],
            is_correct_form=detail["is_correct"],
            incorrect_duration=detail["incorrect_duration"],
            timestamp=datetime.fromtimestamp(detail["timestamp"])
        )
```

**Data Elements**:
- Rep-by-rep form analysis
- Correctness validation
- Structured features metadata
- Precise timestamps

### Step 4: System Feedback Bulk Insert
**Function**: `SystemFeedback.create()`
**Location**: `main.py:253-264`

```python
if self.feedback_data:
    print(f"💾 Saving {len(self.feedback_data)} feedback entries...")
    for feedback in self.feedback_data:
        SystemFeedback.create(
            session_id=session_id,
            message=feedback["message"],
            feedback_type="form_correction",
            related_rep=feedback.get("related_rep"),
            timestamp=datetime.fromtimestamp(feedback["timestamp"])
        )
```

**Feedback Features**:
- Form correction messages
- Rep association tracking
- Timestamp precision
- Standardized feedback types

---

## Video Recording System

### 1. **Video Setup**
**Location**: `main.py:51-65`

```python
def setup_video_recording(self, cap):
    """Setup video recording for the session."""
    if not self.save_video:
        return None
        
    self.temp_video_path = f"/tmp/user{self.user_id}_{self.exercise_name}_session.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    self.video_writer = cv2.VideoWriter(self.temp_video_path, fourcc, fps, (width, height))
    print(f"📹 Video recording started: {self.temp_video_path}")
    return self.video_writer
```

### 2. **Video Finalization**
**Location**: `main.py:67-87`

```python
def finalize_video_recording(self):
    """Finalize video recording and move to permanent location."""
    if self.video_writer:
        self.video_writer.release()
        self.video_writer = None
        
    if self.temp_video_path and os.path.exists(self.temp_video_path):
        # Create videos directory
        videos_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "videos"))
        os.makedirs(videos_dir, exist_ok=True)
        
        # Generate final video filename with timestamp
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        final_video_name = f"user{self.user_id}_{self.exercise_name}_session_{timestamp}.mp4"
        self.final_video_path = os.path.join(videos_dir, final_video_name)
        
        # Move video to final location
        shutil.move(self.temp_video_path, self.final_video_path)
        print(f"💾 Video saved: {self.final_video_path}")
        return self.final_video_path
    return None
```

**Video Pipeline**:
1. Temporary recording during session (`/tmp/`)
2. MP4 codec with automatic FPS detection
3. Final video moved to `videos/` directory
4. Timestamped filename generation
5. Database path storage for later retrieval

---

## Transaction Execution Flow

### Phase 1: Initialization
```python
# Parse command line arguments
args = parser.parse_args()

# Create ExerciseSession instance
session = ExerciseSession(
    exercise_name=args.exercise,
    user_id=args.user_id,
    show_window=not args.no_window,
    save_keypoints=args.save_keypoints,
    save_video=not args.no_video
)
```

### Phase 2: Camera and Video Setup
```python
# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open camera")
    return None

# Setup video recording (if enabled)
self.setup_video_recording(cap)
```

### Phase 3: Real-time Processing
```python
# Process frames in real-time
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Pose detection and form analysis
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        self.process_frame(frame, landmarks)
    
    # Video recording
    if self.video_writer:
        self.video_writer.write(frame)
```

### Phase 4: Session Finalization
```python
# Cleanup resources
cap.release()
if self.show_window:
    cv2.destroyAllWindows()

# Finalize video recording
self.finalize_video_recording()

# Save to database
self.save_session_to_database(final_reps)
```

---

## Error Handling Scenarios

### 1. **Camera Access Failure**
```python
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open camera")
    return None
```

**Rollback**: Session terminates immediately, no database operations.

### 2. **Exercise Validation Failure**
```python
exercise_id = get_exercise_id_by_name(self.exercise_name)
if not exercise_id:
    raise ValueError(f"Exercise '{self.exercise_name}' not found")
```

**Rollback**: Session terminates with error message.

### 3. **Video Recording Failure**
```python
try:
    self.finalize_video_recording()
except Exception as e:
    print(f"⚠️ Video save failed: {e}")
    # Continue with database save (video_path = None)
```

**Rollback**: Partial rollback - session saves without video.

### 4. **Database Transaction Failure**
```python
try:
    session = Session(...)
    session.save()
    # Save details and feedback
except Exception as e:
    print(f"❌ Error saving session to database: {e}")
    import traceback
    traceback.print_exc()
```

**Rollback**: Complete transaction rollback with error logging.

### 5. **Keyboard Interrupt Handling**
```python
except KeyboardInterrupt:
    print("\n🛑 Ctrl+C pressed - stopping and saving session...")
    session_interrupted = True
finally:
    # Always cleanup and save, regardless of how session ended
    cap.release()
    self.finalize_video_recording()
    self.save_session_to_database(final_reps)
```

**Graceful Shutdown**: Ensures data is saved even with interruption.

---

## Command Line Features

### 1. **Flexible Exercise Support**
```bash
python main.py --exercise lunge --user_id 1
python main.py --exercise press --user_id 2
python main.py --exercise plank --user_id 3
```

### 2. **Headless Operation**
```bash
# Run without video display window
python main.py --exercise lunge --user_id 1 --no-window
```

**Use Case**: Server environments or automated testing.

### 3. **Selective Features**
```bash
# Disable video recording for faster processing
python main.py --exercise press --user_id 1 --no-video

# Enable keypoint data storage for analysis
python main.py --exercise plank --user_id 1 --save-keypoints
```

### 4. **Session Output Information**
```python
if session_data:
    print(f"🎉 Session completed successfully!")
    if session_data.get("video_path"):
        print(f"📹 Video saved: {session_data['video_path']}")
    print(f"📊 Final stats: {session_data['reps']} reps in {(session_data['end_time'] - session_data['start_time']).total_seconds():.1f} seconds")
```

**Output Example**:
```
🎉 Session completed successfully!
📹 Video saved: /path/to/videos/user1_lunge_session_20250615_143022.mp4
📊 Final stats: 12 reps in 45.3 seconds
```

---

## Performance Considerations

### 1. **Resource Management**
```python
# Proper resource cleanup
try:
    with self.pose_detector as pose:
        # Processing loop
finally:
    cap.release()
    if self.show_window:
        cv2.destroyAllWindows()
```

### 2. **Memory Optimization**
```python
# Process frames individually instead of buffering
while cap.isOpened():
    success, frame = cap.read()
    # Process immediately and discard
```

### 3. **Video Recording Efficiency**
```python
# Direct frame writing during capture
if self.video_writer:
    self.video_writer.write(frame)
```

### 4. **Database Batch Operations**
```python
# Buffer session data during capture, batch insert at end
for detail in self.session_details:
    SessionDetails.create(...)  # Could be optimized with executemany()
```

---

## Business Rules & Constraints

### 1. **Required Parameters**
```python
parser.add_argument("--exercise", required=True)
parser.add_argument("--user_id", type=int, required=True)
```
**Rule**: Exercise type and user ID are mandatory.

### 2. **Valid Exercise Types**
```python
# Exercise must exist in database
exercise_id = get_exercise_id_by_name(self.exercise_name)
```
**Rule**: Only exercises in the Exercise table are supported.

### 3. **Rep Count Validation**
```python
if reps > 0:  # Only save details for actual reps
    self.session_details.append(detail)
```
**Rule**: SessionDetails only created for completed reps.

### 4. **Video File Naming**
```python
final_video_name = f"user{self.user_id}_{self.exercise_name}_session_{timestamp}.mp4"
```
**Rule**: Video files include user ID, exercise, and timestamp.

### 5. **Session Status**
```python
session_status='completed'  # Always completed for command line sessions
```
**Rule**: Command line sessions are always marked as completed.

---

## Integration Points

### 1. **MediaPipe Integration**
```python
from src.processing.pose_detector import PoseDetectorFactory
self.pose_detector = PoseDetectorFactory.create_live_exercise_detector()
```

### 2. **Exercise Form Checkers**
```python
from src.processing.forms_check import check_form, init_state
feedback, reps_count = check_form(self.exercise_name, frame, landmarks, exercise_state)
```

### 3. **Database Models**
```python
from src.database.session import Session
from src.database.session_details import SessionDetails
from src.database.system_feedback import SystemFeedback
```

### 4. **Video File System**
```python
# Integration with video serving system
videos_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "videos"))
```

---

## Testing & Validation

### 1. **Command Line Testing**
```bash
# Test different exercise types
python main.py --exercise lunge --user_id 1
python main.py --exercise press --user_id 1
python main.py --exercise plank --user_id 1

# Test different configurations
python main.py --exercise lunge --user_id 1 --no-window --no-video
```

### 2. **Error Condition Testing**
- Invalid exercise names
- Non-existent user IDs
- Camera access failures
- Database connection issues

### 3. **Performance Testing**
- Long session duration handling
- Memory usage monitoring
- Video file size optimization

### 4. **Integration Testing**
- Database transaction consistency
- Video file creation and storage
- Session data accuracy

---

## Future Enhancement Opportunities

### 1. **Enhanced Command Line Interface**
```python
# Additional options
parser.add_argument("--duration", type=int, help="Session duration in minutes")
parser.add_argument("--target-reps", type=int, help="Target rep count")
parser.add_argument("--config", help="Configuration file path")
```

### 2. **Real-time Feedback**
```python
# Audio feedback during command line sessions
import pyttsx3
def speak_feedback(message):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()
```

### 3. **Batch Processing**
```python
# Multiple exercises in one session
def run_workout(exercises, user_id):
    for exercise in exercises:
        session = ExerciseSession(exercise, user_id)
        session.run()
```

### 4. **Configuration File Support**
```python
# YAML/JSON configuration files
import yaml
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
```

---

## Summary

The Command Line Session Transaction provides a robust, standalone exercise tracking capability with comprehensive database integration and optional video recording. The transaction ensures reliable data capture and storage while offering flexible configuration options for different use cases.

**Key Strengths**:
- Standalone operation independent of web interface
- Comprehensive argument parsing and validation
- Robust error handling with graceful shutdown
- Optional video recording with proper file management
- Complete database integration with transaction safety

**Areas for Improvement**:
- Batch database operations for better performance
- Configuration file support for complex setups
- Real-time audio feedback capabilities
- Enhanced monitoring and logging features