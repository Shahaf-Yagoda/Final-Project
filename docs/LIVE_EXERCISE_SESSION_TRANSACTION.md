# Live Exercise Session Transaction - Comprehensive Documentation

## Transaction Overview

The **Live Exercise Session Transaction** is the core business logic transaction in the Right Motion fitness tracking application. It orchestrates real-time exercise tracking with computer vision pose detection, form analysis, and comprehensive database operations across multiple tables.

### Transaction Classification
- **Priority Level**: 1 (Core Business Logic)
- **Complexity**: High
- **Tables Involved**: 5 primary tables (Exercise, Workout, Session, SessionDetails, SystemFeedback)
- **Transaction Type**: Multi-table ACID compliant
- **Execution Context**: Real-time video streaming with database persistence

---

## Data Flow Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Camera Feed   │────│  MediaPipe Pose  │────│ Form Analysis   │
│                 │    │   Detection      │    │   Engine        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Session State Manager                        │
│  ▪ Real-time rep counting                                      │
│  ▪ Form validation                                             │
│  ▪ Feedback generation                                         │
│  ▪ Video recording                                             │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Database Transaction Layer                     │
│                                                                │
│  1. Exercise Lookup    →  Exercise Table                       │
│  2. Workout Context    →  Workout Table (optional)             │
│  3. Session Creation   →  Session Table                        │
│  4. Detail Logging     →  SessionDetails Table                 │
│  5. Feedback Storage   →  SystemFeedback Table                 │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Transaction Components

### 1. **Session Initialization**
**Location**: `main.py:28-50`, `video_streamer.py:20-32`

```python
class ExerciseSession:
    def __init__(self, exercise_name, user_id, show_window=True, save_keypoints=False, save_video=True):
        self.exercise_name = exercise_name
        self.user_id = user_id
        self.state = {"exercise": {exercise_name: init_state(exercise_name)}}
        self.pose_detector = PoseDetectorFactory.create_live_exercise_detector()
        self.session_details = []
        self.feedback_data = []
```

**Purpose**: Initialize session state, pose detection pipeline, and data collection structures.

### 2. **Real-time Processing Loop**
**Location**: `main.py:152-176`, `video_streamer.py:242-272`

**Key Operations**:
- Camera frame capture
- MediaPipe pose landmark detection
- Form analysis using exercise-specific checkers
- Rep counting with state persistence
- Real-time feedback generation
- Video frame recording
- Temporary data buffering

### 3. **Session Data Collection**
**Location**: `main.py:89-109`, `video_streamer.py:175-192`

```python
def save_session_detail(self, landmarks, feedback, reps, current_time):
    if reps > 0:  # Database constraint: only save details with actual reps
        detail = {
            "timestamp": current_time,
            "rep_num": reps,
            "features_json": {
                "form_correct": len(feedback) == 0,
                "feedback_count": len(feedback),
                "timestamp": current_time
            },
            "is_correct": len(feedback) == 0,
            "incorrect_duration": 0
        }
        self.session_details.append(detail)
    
    # Save feedback separately
    for msg in feedback:
        self.feedback_data.append({
            "timestamp": current_time, 
            "message": msg, 
            "related_rep": reps if reps > 0 else None
        })
```

---

## Database Transaction Steps

### Step 1: Exercise Lookup
**Function**: `get_exercise_id_by_name()`
**Location**: `src/database/db_utils.py:25`

```sql
SELECT exercise_id FROM Exercise WHERE name = %s
```

**Purpose**: Validate exercise exists and retrieve exercise_id for foreign key relationships.

**Error Handling**: Returns None if exercise not found, causing transaction to fail gracefully.

### Step 2: Workout Context (Optional)
**Function**: `Workout.create()` or existing workout assignment
**Location**: `src/database/workout.py`

```sql
-- For structured workouts (optional)
INSERT INTO Workout (
    user_id, workout_name, planned_date, actual_date,
    workout_status, notes, created_at, updated_at
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) 
RETURNING workout_id

-- OR assign to existing workout
SELECT workout_id FROM Workout WHERE user_id = %s AND workout_status = 'in_progress'
```

**Purpose**: Group sessions within structured workouts (optional for standalone sessions).

**Key Fields**:
- `user_id`: Foreign key to User table
- `workout_name`: Descriptive name for the workout session
- `workout_status`: 'in_progress', 'completed', or 'planned'
- `actual_date`: When the workout was performed

**Note**: Live exercise sessions can be standalone (`workout_id = NULL`) or part of a structured workout.

### Step 3: Session Creation
**Function**: `Session.save()`
**Location**: `src/database/session.py:34-82`

```sql
INSERT INTO Session (
    workout_id, exercise_id, user_id, session_order,
    start_time, end_time, duration_sec, planned_reps, actual_reps,
    session_status, video_path, created_at, updated_at,
    reps_count, feedback_count, performance_score
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
RETURNING session_id
```

**Key Fields**:
- `user_id`: Foreign key to User table
- `exercise_id`: Foreign key to Exercise table
- `workout_id`: Foreign key to Workout table (NULL for standalone sessions)
- `start_time`/`end_time`: Session temporal boundaries
- `actual_reps`: Final rep count from form analysis
- `session_status`: 'completed' (default after successful session)
- `video_path`: File path to recorded session video

### Step 4: Session Details Bulk Insert
**Function**: `SessionDetails.create()`
**Location**: `src/database/session_details.py:34-58`

```sql
INSERT INTO SessionDetails (
    session_id, rep_number, features_json, 
    is_correct_form, incorrect_duration, timestamp
) VALUES (%s, %s, %s, %s, %s, %s)
```

**Batch Operation**: Each rep generates one SessionDetails record.

**Data Structure**:
- `features_json`: JSONB field containing form analysis metadata
- `is_correct_form`: Boolean derived from feedback presence
- `rep_number`: Cumulative rep count at time of recording

### Step 5: System Feedback Bulk Insert
**Function**: `SystemFeedback.create()`
**Location**: `src/database/system_feedback.py:34-58`

```sql
INSERT INTO SystemFeedback (
    session_id, message, feedback_type, 
    related_rep, timestamp
) VALUES (%s, %s, %s, %s, %s)
```

**Batch Operation**: Each form correction generates one SystemFeedback record.

**Key Features**:
- `message`: Human-readable form correction advice
- `feedback_type`: 'form_correction' (standardized)
- `related_rep`: Links feedback to specific rep (nullable)

---

## Transaction Execution Flow

### Phase 1: Pre-Transaction Setup
```python
# Initialize session state and resources
self.start_time = datetime.now()
self.setup_video_recording(cap)
exercise_id = get_exercise_id_by_name(self.exercise_name)
```

### Phase 2: Real-time Data Collection
```python
# Continuous loop during exercise
while cap.isOpened():
    # Pose detection and form analysis
    results = pose.process(image_rgb)
    feedback, reps = exercise_checker.check_form(frame, landmarks, state)
    
    # Buffer data for database transaction
    self.save_session_detail(landmarks, feedback, reps, current_time)
    
    # Write to video file
    if self.video_writer:
        self.video_writer.write(frame)
```

### Phase 3: Transaction Commit
```python
def save_session_to_database(self, reps_count):
    try:
        # Step 1: Create session record
        session = Session(...)
        session.save()  # Returns session_id
        
        # Step 2: Bulk insert session details
        for detail in self.session_details:
            SessionDetails.create(session_id=session_id, ...)
            
        # Step 3: Bulk insert feedback
        for feedback in self.feedback_data:
            SystemFeedback.create(session_id=session_id, ...)
            
        conn.commit()  # ACID transaction commit
        
    except Exception as e:
        conn.rollback()  # Automatic rollback on error
        raise e
```

---

## Error Handling & Rollback Scenarios

### 1. **Exercise Validation Failure**
```python
exercise_id = get_exercise_id_by_name(self.exercise_name)
if not exercise_id:
    raise ValueError(f"Exercise '{self.exercise_name}' not found")
```

**Rollback**: No database changes, session terminated gracefully.

### 2. **Session Creation Failure**
```python
try:
    session.save()
except psycopg2.Error as e:
    conn.rollback()
    raise e
```

**Rollback**: Session record creation fails, no child records created.

### 3. **Detail/Feedback Insert Failure**
```python
try:
    # Bulk insert operations
    for detail in self.session_details:
        SessionDetails.create(...)
except Exception as e:
    conn.rollback()  # Rollback session and all details
    raise e
```

**Rollback**: Complete transaction rollback, including session record.

### 4. **Video Recording Failure**
```python
try:
    self.finalize_video_recording()
except Exception as e:
    print(f"⚠️ Video save failed: {e}")
    # Continue with database save (video_path = None)
```

**Rollback**: Partial rollback - database save continues with empty video_path.

### 5. **Database Connection Loss**
```python
conn = get_connection()
if not conn:
    print("❌ Failed to connect to database!")
    return False
```

**Rollback**: Complete transaction failure, temporary data preserved in memory.

---

## Performance Considerations

### 1. **Batch Operations**
- SessionDetails and SystemFeedback use individual INSERTs within transaction
- **Optimization Opportunity**: Could be improved with `executemany()` for bulk inserts

### 2. **Memory Management**
```python
self.session_details = []  # In-memory buffer during session
self.feedback_data = []    # Released after database commit
```

### 3. **Video Recording**
- Temporary video file in `/tmp/` during recording
- Moved to permanent location after session completion
- **Optimization**: Streaming directly to final location

### 4. **Database Connection Pooling**
```python
conn = get_connection()  # New connection per transaction
# Could benefit from connection pooling for better performance
```

---

## Business Rules & Constraints

### 1. **Rep Validation**
```python
if reps > 0:  # Only save SessionDetails for actual reps
    detail = {...}
    self.session_details.append(detail)
```
**Rule**: SessionDetails records only created for completed reps (database constraint).

### 2. **Feedback Throttling** (Video Streamer Only)
```python
def should_throttle_feedback(self, user_id, message, current_time, throttle_seconds=2.0):
    # Prevent duplicate feedback within 2-second window
```
**Rule**: Duplicate feedback messages throttled to improve user experience.

### 3. **Session Status Management**
```python
self.session_status = 'completed'  # Always 'completed' for successful sessions
```
**Rule**: Live sessions always marked as 'completed' (no 'in_progress' persistence).

### 4. **Video Path Handling**
```python
video_path=self.final_video_path or ""  # Never NULL in database
```
**Rule**: Video path defaults to empty string if recording fails.

---

## Integration Points

### 1. **MediaPipe Pose Detection**
```python
self.pose_detector = PoseDetectorFactory.create_live_exercise_detector()
results = self.pose_detector.process(image_rgb)
```

### 2. **Exercise Form Checkers**
```python
exercise_checker = get_exercise_checker(exercise)
feedback, reps = exercise_checker.check_form(frame, landmarks, state)
```

### 3. **Video Streaming (Streamlit Integration)**
```python
# video_streamer.py serves Flask endpoints for Streamlit
@app.route("/video_feed")
def video_feed():
    return Response(video_manager.generate_stream(exercise, user_id))
```

### 4. **File System Operations**
```python
# Temporary files for inter-process communication
with open(f"/tmp/reps_{user_id}.txt", "w") as f:
    f.write(str(reps))
```

---

## Testing & Validation

### 1. **Unit Tests**
- `tests/test_exercise_session.py`: ExerciseSession class methods
- `tests/test_database_operations.py`: Database transaction validation

### 2. **Integration Tests**
- `tests/test_integration.py`: End-to-end session flow testing
- Video recording validation
- Database consistency checks

### 3. **Performance Tests**
- Session duration vs. database size correlation
- Memory usage during long sessions
- Video file size optimization

---

## Monitoring & Observability

### 1. **Database Metrics**
```sql
-- Session completion rates
SELECT session_status, COUNT(*) 
FROM Session 
GROUP BY session_status;

-- Average session duration by exercise
SELECT e.name, AVG(s.duration_sec) 
FROM Session s 
JOIN Exercise e ON s.exercise_id = e.exercise_id 
GROUP BY e.name;
```

### 2. **Application Metrics**
```python
print(f"💾 Session saved to database (ID: {session_id})")
print(f"💾 Saving {len(self.session_details)} session details...")
print(f"💾 Saving {len(self.feedback_data)} feedback entries...")
```

### 3. **Error Tracking**
```python
except Exception as e:
    print(f"❌ Error saving session to database: {e}")
    import traceback
    traceback.print_exc()
```

---

## Security Considerations

### 1. **SQL Injection Prevention**
```python
# All queries use parameterized statements
cur.execute("INSERT INTO Session (...) VALUES (%s, %s, ...)", (values,))
```

### 2. **File Path Validation**
```python
# Security check for video serving
if not os.path.abspath(video_path).startswith(os.path.abspath(videos_dir)):
    return "Access denied", 403
```

### 3. **User ID Validation**
```python
try:
    user_id = int(user_id)
except (TypeError, ValueError):
    return "Invalid user_id", 400
```

---

## Future Optimization Opportunities

### 1. **Database Optimizations**
- Implement connection pooling
- Use `executemany()` for bulk inserts
- Add database indexes for frequently queried fields

### 2. **Real-time Performance**
- Implement background database writes
- Add session data compression
- Optimize video codec selection

### 3. **Scalability Improvements**
- Add horizontal scaling support
- Implement distributed session storage
- Add load balancing for video streaming

### 4. **Reliability Enhancements**
- Add circuit breaker patterns
- Implement retry mechanisms for database operations
- Add comprehensive health checks

---

## Summary

The Live Exercise Session Transaction represents the core value proposition of the Right Motion application, combining real-time computer vision analysis with comprehensive database persistence. The transaction ensures ACID compliance while managing complex multi-table relationships and maintaining performance during real-time video processing.

**Key Strengths**:
- Complete ACID transaction support with automatic rollback
- Comprehensive data capture (session, details, feedback)
- Real-time performance with buffering strategy
- Robust error handling and graceful degradation

**Areas for Improvement**:
- Bulk insert optimization for better performance
- Connection pooling for reduced database overhead
- Background processing for non-critical operations
- Enhanced monitoring and observability features