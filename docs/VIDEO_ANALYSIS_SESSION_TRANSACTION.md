# Video Analysis Session Transaction - Comprehensive Documentation

## Transaction Overview

The **Video Analysis Session Transaction** handles offline video file analysis using computer vision pose detection and form analysis. Unlike live sessions, this transaction processes pre-recorded videos frame-by-frame, generating timestamped feedback and saving analyzed results to the database.

### Transaction Classification
- **Priority Level**: 1 (Core Business Logic)
- **Complexity**: High
- **Tables Involved**: 2 primary tables (Session, Exercise)
- **Transaction Type**: Single-transaction with file processing
- **Execution Context**: Streamlit web interface with file upload

---

## Data Flow Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Video Upload   │────│   File Storage   │────│  Format Valid.  │
│  (MP4/AVI/MOV)  │    │    (temp file)   │    │   & Size Check  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                Frame-by-Frame Processing Pipeline               │
│  ▪ Video codec optimization for browser compatibility          │
│  ▪ MediaPipe pose detection per frame                          │
│  ▪ Exercise-specific form analysis                             │
│  ▪ Timestamped feedback generation                             │
│  ▪ Processed video output with pose landmarks                  │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Database Transaction Layer                      │
│                                                                │
│  1. Exercise Lookup    →  Exercise Table                       │
│  2. Session Creation   →  Session Table (with video_path)      │
│  3. Video Serving      →  Flask Backend (HTTP streaming)       │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Transaction Components

### 1. **Video Upload & Validation**
**Location**: `src/app/app.py:334-367`

```python
# File upload with type and size validation
video_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
selected_exercise = option_menu(options=["lunge", "press", "plank"])

# Size validation (200MB limit)
if video_file.size > 200 * 1024 * 1024:
    st.error("Video file is too large. Please upload a video smaller than 200MB.")
    return

# Exercise validation
if not selected_exercise:
    st.error("Please select an exercise type.")
    return
```

**Purpose**: Validate file format, size constraints, and exercise selection before processing.

### 2. **MediaPipe Pose Detection Setup**
**Location**: `src/app/app.py:377-382`

```python
from src.processing.pose_detector import PoseDetectorFactory, get_mp_drawing_utils, get_mp_pose_solutions

pose = PoseDetectorFactory.create_video_analysis_detector()
mp_drawing = get_mp_drawing_utils()
mp_pose = get_mp_pose_solutions()
```

**Configuration**: Optimized for video analysis with different tracking confidence settings than live mode.

### 3. **Video Processing Pipeline**
**Location**: `src/app/app.py:383-476`

**Key Operations**:
- Temporary file storage
- Multi-codec video writer setup for browser compatibility
- Frame-by-frame pose detection
- Exercise-specific form analysis
- Timestamped feedback generation
- Processed video output with pose landmarks

```python
# Multi-codec support for browser compatibility
fourcc_options = [
    cv2.VideoWriter_fourcc(*'H264'),  # H.264 (best browser support)
    cv2.VideoWriter_fourcc(*'avc1'),  # H.264 alternative
    cv2.VideoWriter_fourcc(*'XVID'),  # XVID (good compatibility)
    cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4 (fallback)
    cv2.VideoWriter_fourcc(*'MJPG'),  # Motion JPEG (last resort)
]

# Frame processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # MediaPipe pose detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        # Draw pose landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Form analysis
        feedback, reps = check_form(selected_exercise, frame, results.pose_landmarks.landmark, state)
        rep_count = max(rep_count, reps)
        
        # Add timestamp to feedback
        if feedback:
            for msg in feedback:
                timestamp_str = f"{int(current_timestamp//60):02d}:{int(current_timestamp%60):02d}"
                feedback_messages.append(f"{timestamp_str} - {msg}")
    
    # Write processed frame
    out.write(frame)
```

---

## Database Transaction Steps

### Step 1: Exercise Lookup
**Function**: `get_exercise_id_by_name()`
**Location**: `src/database/db_utils.py`

```sql
SELECT exercise_id FROM Exercise WHERE name = %s
```

**Purpose**: Validate exercise exists and retrieve exercise_id for foreign key relationship.

### Step 2: Session Creation
**Function**: `Session.save()`
**Location**: `src/database/session.py:34-82`

```sql
INSERT INTO Session (
    user_id, exercise_id, start_time, end_time,
    video_path, reps_count, feedback_count,
    session_status, created_at, updated_at
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
RETURNING session_id
```

**Key Fields**:
- `user_id`: Foreign key to User table
- `exercise_id`: Foreign key to Exercise table
- `video_path`: Path to analyzed video file
- `reps_count`: Total reps detected in video
- `feedback_count`: Number of form corrections generated
- `session_status`: 'completed' (default for analyzed videos)

### Step 3: Video File Storage
**Location**: `src/app/app.py:386-394`

```python
# Create output directory and generate unique filename
videos_dir = os.path.join(os.path.dirname(__file__), "..", "..", "videos")
os.makedirs(videos_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"analyzed_video_{timestamp}.mp4"
output_path = os.path.join(videos_dir, output_filename)
```

**Purpose**: Store processed video with pose landmarks in permanent location.

---

## Transaction Execution Flow

### Phase 1: Upload & Validation
```python
# File upload with validation
if video_file and selected_exercise:
    # Size and format validation
    if video_file.size > 200 * 1024 * 1024:
        st.error("Video file too large")
        return
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        temp_path = tmp_file.name
```

### Phase 2: Video Processing
```python
# Initialize video processing
cap = cv2.VideoCapture(temp_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Setup output video with codec fallback
for fourcc in fourcc_options:
    try:
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            used_codec = fourcc
            break
    except Exception:
        continue

# Frame-by-frame processing
state = init_state(selected_exercise)
rep_count = 0
feedback_messages = []

with pose:
    while cap.isOpened():
        # Pose detection and form analysis per frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame with MediaPipe and form checker
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Form analysis and feedback generation
            feedback, reps = check_form(selected_exercise, frame, results.pose_landmarks.landmark, state)
            # Accumulate results and write processed frame
```

### Phase 3: Database & File Finalization
```python
# Save session to database
exercise_id = get_exercise_id_by_name(selected_exercise)
session = Session(
    user_id=user_id,
    exercise_id=exercise_id,
    start_time=datetime.now(),
    end_time=datetime.now(),
    video_path=output_path,
    reps_count=rep_count,
    feedback_count=len(feedback_messages)
)
session.save()

# Clean up temporary files
cap.release()
out.release()
os.unlink(temp_path)
```

---

## Error Handling Scenarios

### 1. **File Upload Validation Failure**
```python
if video_file.size > 200 * 1024 * 1024:
    st.error("Video file is too large. Please upload a video smaller than 200MB.")
    return
```

**Rollback**: No processing initiated, user prompted to upload smaller file.

### 2. **Video Codec Initialization Failure**
```python
out = None
for fourcc in fourcc_options:
    try:
        test_out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if test_out.isOpened():
            out = test_out
            used_codec = fourcc
            break
    except Exception as e:
        st.warning(f"Failed to initialize with codec {fourcc}: {e}")
        continue

if out is None:
    st.error("Failed to initialize video writer with any supported codec")
    return
```

**Rollback**: Fallback through multiple codecs, error display if all fail.

### 3. **MediaPipe Processing Failure**
```python
try:
    with pose:
        while cap.isOpened():
            # Processing loop
except Exception as e:
    st.error(f"Error during video analysis: {e}")
    # Clean up resources
    cap.release()
    if out:
        out.release()
    if os.path.exists(temp_path):
        os.unlink(temp_path)
```

**Rollback**: Resource cleanup, temporary file deletion, error message display.

### 4. **Database Transaction Failure**
```python
try:
    session = Session(...)
    session.save()
    st.success(f"Analysis complete! Session saved (ID: {session.session_id})")
except Exception as e:
    st.error(f"Failed to save session to database: {e}")
    # Video file remains, but no database record
```

**Rollback**: Video analysis completes but database save fails, partial success state.

### 5. **Video File Serving Failure**
```python
# Video readiness checking
video_ready = False
max_wait_time = 30  # Maximum 30 seconds
while not video_ready and waited_time < max_wait_time:
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size > 0:
            try:
                test_cap = cv2.VideoCapture(output_path)
                if test_cap.isOpened():
                    ret, frame = test_cap.read()
                    if ret and frame is not None:
                        video_ready = True
                test_cap.release()
            except:
                pass
```

**Rollback**: Graceful degradation with retry mechanisms and fallback display options.

---

## Performance Considerations

### 1. **Video Codec Optimization**
```python
fourcc_options = [
    cv2.VideoWriter_fourcc(*'H264'),  # Best browser support
    cv2.VideoWriter_fourcc(*'avc1'),  # H.264 alternative
    cv2.VideoWriter_fourcc(*'XVID'),  # Good compatibility
    cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4 fallback
    cv2.VideoWriter_fourcc(*'MJPG'),  # Last resort
]
```

**Optimization**: Prioritized codec selection for optimal browser compatibility and file size.

### 2. **Progress Tracking**
```python
if frame_count % 30 == 0:  # Update every 30 frames
    progress_percent = min(40 + int((frame_count / total_frames) * 50), 90)
    progress_bar.progress(progress_percent)
    status_text.text(f"🏃 Analyzing frame {frame_count}/{total_frames}...")
```

**User Experience**: Real-time progress updates prevent user confusion during long processing.

### 3. **Memory Management**
```python
# Process frame-by-frame instead of loading entire video into memory
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process single frame and release immediately
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    # Frame is garbage collected after processing
```

### 4. **File Size Optimization**
```python
# Resize frame if needed to match expected dimensions
if frame.shape[1] != width or frame.shape[0] != height:
    frame = cv2.resize(frame, (width, height))
```

**Optimization**: Ensures consistent output dimensions for optimal codec compression.

---

## Business Rules & Constraints

### 1. **File Size Limit**
```python
if video_file.size > 200 * 1024 * 1024:  # 200MB limit
    st.error("Video file is too large...")
```
**Rule**: Maximum 200MB file size to prevent server resource exhaustion.

### 2. **Supported Formats**
```python
video_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
```
**Rule**: Only MP4, AVI, and MOV formats supported for compatibility.

### 3. **Rep Count Calculation**
```python
rep_count = max(rep_count, reps)  # Use cumulative maximum
```
**Rule**: Rep count represents maximum detected reps throughout video analysis.

### 4. **Feedback Deduplication**
```python
# Remove duplicates while preserving order
unique_feedback = []
seen = set()
for msg in feedback_messages:
    if msg not in seen:
        unique_feedback.append(msg)
        seen.add(msg)
```
**Rule**: Duplicate feedback messages filtered to improve user experience.

### 5. **Timestamp Formatting**
```python
timestamp_str = f"{int(current_timestamp//60):02d}:{int(current_timestamp%60):02d}"
feedback_messages.append(f"{timestamp_str} - {msg}")
```
**Rule**: All feedback includes MM:SS timestamp for video reference.

---

## Integration Points

### 1. **Streamlit File Upload**
```python
video_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
```

### 2. **MediaPipe Pose Detection**
```python
pose = PoseDetectorFactory.create_video_analysis_detector()
results = pose.process(rgb_frame)
```

### 3. **Exercise Form Checkers**
```python
from src.processing.forms_check import check_form
feedback, reps = check_form(selected_exercise, frame, results.pose_landmarks.landmark, state)
```

### 4. **Flask Video Serving**
```python
video_url = f"http://localhost:5050/serve_video/{output_filename}"
```

### 5. **Database Session Storage**
```python
session = Session(user_id=user_id, exercise_id=exercise_id, video_path=output_path)
session.save()
```

---

## Advanced Features

### 1. **Multi-Codec Fallback System**
```python
# Try each codec until one works
for fourcc in fourcc_options:
    try:
        test_out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if test_out.isOpened():
            out = test_out
            used_codec = fourcc
            break
    except Exception:
        continue
```

### 2. **Enhanced Video Player**
```python
video_html = f"""
<video id="analyzed-video" width="100%" height="auto" controls preload="metadata">
    <source src="{video_url}" type="video/mp4; codecs=&quot;mp4v.20.9,mp4a.40.2&quot;">
    <source src="{video_url}" type="video/mp4">
    Your browser does not support the video tag or the video codec.
</video>
"""
```

### 3. **Real-time Processing Feedback**
```python
# Progress bar and status updates
progress_bar = st.progress(0)
status_text = st.empty()
progress_bar.progress(progress_percent)
status_text.text(f"🏃 Analyzing frame {frame_count}/{total_frames}...")
```

### 4. **Video Integrity Verification**
```python
# Verify video file is readable before serving
try:
    test_cap = cv2.VideoCapture(output_path)
    if test_cap.isOpened():
        ret, frame = test_cap.read()
        if ret and frame is not None:
            video_ready = True
    test_cap.release()
except:
    video_ready = False
```

---

## Security Considerations

### 1. **File Type Validation**
```python
video_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
```

### 2. **File Size Limits**
```python
if video_file.size > 200 * 1024 * 1024:
    st.error("Video file is too large...")
```

### 3. **Secure File Paths**
```python
# Use secure temporary file handling
with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
    tmp_file.write(video_file.read())
    temp_path = tmp_file.name
```

### 4. **Path Validation in Video Serving**
```python
# Security check in video_streamer.py
if not os.path.abspath(video_path).startswith(os.path.abspath(videos_dir)):
    return "Access denied", 403
```

---

## Testing & Validation

### 1. **Supported Video Formats**
- MP4 with H.264 codec (primary)
- AVI with various codecs
- MOV format support
- Codec fallback testing

### 2. **Performance Testing**
- Large file processing (up to 200MB)
- Long video duration handling
- Memory usage optimization
- Browser compatibility validation

### 3. **Error Scenarios**
- Corrupted video files
- Unsupported codecs
- Network interruption during upload
- Database connection failures

---

## Future Optimization Opportunities

### 1. **Batch Processing**
- Multiple video upload support
- Background processing queue
- Parallel video analysis

### 2. **Cloud Storage Integration**
- Direct cloud upload/download
- CDN integration for video serving
- Distributed processing

### 3. **Advanced Analytics**
- Video quality metrics
- Performance benchmarking
- User engagement tracking

### 4. **Enhanced User Experience**
- Video thumbnail generation
- Drag-and-drop upload
- Progress estimation improvements

---

## Summary

The Video Analysis Session Transaction provides comprehensive offline video analysis capabilities, combining computer vision pose detection with exercise-specific form analysis. The transaction ensures robust error handling, multi-codec compatibility, and optimal user experience through progress tracking and enhanced video playback.

**Key Strengths**:
- Comprehensive video format support with codec fallback
- Frame-by-frame analysis with timestamped feedback
- Robust error handling and resource cleanup
- Browser-optimized video serving
- Real-time progress tracking

**Areas for Improvement**:
- Batch processing capabilities for multiple videos
- Background processing to improve UI responsiveness
- Cloud storage integration for scalability
- Enhanced video quality optimization