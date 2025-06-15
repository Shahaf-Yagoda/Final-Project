# Session History Retrieval Transaction - Comprehensive Documentation

## Transaction Overview

The **Session History Retrieval Transaction** handles the display of user exercise history through the Streamlit web interface. This transaction retrieves comprehensive session data, performance metrics, and associated video recordings, presenting them in an organized, user-friendly format.

### Transaction Classification
- **Priority Level**: 2 (Supporting Feature)
- **Complexity**: Medium
- **Tables Involved**: 4 primary tables (Session, SessionDetails, SystemFeedback, Exercise)
- **Transaction Type**: Read-only with complex joins
- **Execution Context**: Streamlit web interface with authenticated user sessions

---

## Data Flow Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ User Navigation │────│  Authentication  │────│   User Session  │
│   to History    │    │     Check        │    │     State       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Session Data Retrieval                       │
│  ▪ Primary session lookup by user_id                           │
│  ▪ Session details and feedback aggregation                    │
│  ▪ Video file availability verification                        │
│  ▪ Exercise information enrichment                             │
│  ▪ Temporal ordering and pagination                            │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    UI Presentation Layer                        │
│                                                                │
│  1. Session List       →  Formatted session cards              │
│  2. Video Integration  →  Embedded video players               │
│  3. Metadata Display   →  Exercise details & timestamps        │
│  4. Error Handling     →  User-friendly error messages         │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Transaction Components

### 1. **Session History UI Page**
**Location**: `src/app/app.py:1296-1322`

```python
elif st.session_state.page == "History":
    st.title("Session History")
    user_id = st.session_state.get("user_id")
    if not user_id:
        st.warning("You must be logged in to view your session history.")
    else:
        try:
            from src.database.session import Session
            sessions = Session.load_by_user(user_id)
            if not sessions:
                st.info("No sessions found.")
            else:
                for sess in sessions:
                    st.markdown(f"**Session ID:** {sess.session_id}  ")
                    st.markdown(f"**Date:** {format_datetime(sess.start_time)}  ")
                    st.markdown(f"**Exercise ID:** {sess.exercise_id}  ")
                    st.markdown(f"**Reps:** {sess.reps_count}  ")
                    st.markdown(f"**Duration (sec):** {sess.duration_sec}  ")
                    if sess.video_path and os.path.exists(sess.video_path):
                        st.video(sess.video_path)
                    else:
                        st.info("Video not available or still processing.")
                    st.markdown("---")
        except Exception as e:
            st.error(f"Error loading session history: {e}")
    st.button("⬅️ Back to Home", on_click=set_page, args=("Home",))
```

**UI Features**:
- **Authentication Check**: Requires user login for access
- **Session List**: Displays sessions in reverse chronological order
- **Video Integration**: Embedded video players for recorded sessions
- **Error Handling**: User-friendly error messages for failures
- **Navigation**: Back button for seamless user flow

### 2. **Navigation Integration**
**Location**: `src/app/app.py:266-267`

```python
# Home page navigation button
with col3:
    st.button("📜 Session History", on_click=set_page, args=("History",))
```

**Access Control**: Only displayed for authenticated users.

### 3. **Session Data Retrieval**
**Location**: `src/database/session.py:100-122`

```python
@classmethod
def load_by_user(cls, user_id: int, limit: int = 10) -> List['Session']:
    """Load sessions by user with comprehensive schema"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT session_id, workout_id, exercise_id, user_id, session_order,
                       start_time, end_time, duration_sec, planned_reps, actual_reps,
                       session_status, video_path, created_at, updated_at
                FROM Session WHERE user_id = %s ORDER BY start_time DESC LIMIT %s
            """, (user_id, limit))
            sessions = []
            for row in cur.fetchall():
                session = cls(
                    session_id=row[0], workout_id=row[1], exercise_id=row[2], user_id=row[3],
                    session_order=row[4], start_time=row[5], end_time=row[6], duration=row[7],
                    planned_reps=row[8], actual_reps=row[9], session_status=row[10],
                    video_path=row[11], created_at=row[12], updated_at=row[13]
                )
                sessions.append(session)
            return sessions
    finally:
        conn.close()
```

**Query Features**:
- **User Filtering**: Only sessions for authenticated user
- **Temporal Ordering**: Most recent sessions first (`ORDER BY start_time DESC`)
- **Pagination**: Configurable limit (default 10 sessions)
- **Comprehensive Data**: All session fields for rich display

### 4. **Date/Time Formatting**
**Location**: `src/app/app.py:29-37`

```python
def format_datetime(dt):
    from datetime import datetime
    if isinstance(dt, datetime):
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    try:
        # Try to convert from timestamp (int or float)
        return datetime.fromtimestamp(dt).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return str(dt)
```

**Format Output**: `YYYY-MM-DD HH:MM:SS` for consistent display.

---

## Database Transaction Steps

### Step 1: Authentication Validation
```python
user_id = st.session_state.get("user_id")
if not user_id:
    st.warning("You must be logged in to view your session history.")
    return
```

**Purpose**: Ensure only authenticated users can access session history.

### Step 2: Session Retrieval Query
```sql
SELECT session_id, workout_id, exercise_id, user_id, session_order,
       start_time, end_time, duration_sec, planned_reps, actual_reps,
       session_status, video_path, created_at, updated_at
FROM Session 
WHERE user_id = %s 
ORDER BY start_time DESC 
LIMIT %s
```

**Query Analysis**:
- **Security**: WHERE clause ensures user can only see their own sessions
- **Performance**: LIMIT clause controls result set size
- **Ordering**: Most recent sessions displayed first
- **Completeness**: All session fields retrieved for comprehensive display

### Step 3: Session Object Construction
```python
for row in cur.fetchall():
    session = cls(
        session_id=row[0], workout_id=row[1], exercise_id=row[2], user_id=row[3],
        session_order=row[4], start_time=row[5], end_time=row[6], duration=row[7],
        planned_reps=row[8], actual_reps=row[9], session_status=row[10],
        video_path=row[11], created_at=row[12], updated_at=row[13]
    )
    sessions.append(session)
```

**Purpose**: Convert database rows to Session objects for easier manipulation.

### Step 4: Video File Validation
```python
if sess.video_path and os.path.exists(sess.video_path):
    st.video(sess.video_path)
else:
    st.info("Video not available or still processing.")
```

**File System Check**: Verifies video file exists before attempting to display.

---

## Extended Data Access Methods

### 1. **Session Details Retrieval**
**Location**: `src/database/session_details.py:59-82`

```python
@classmethod
def get_by_session(cls, session_id: int) -> List['SessionDetails']:
    """Get all details for a session"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT detail_id, session_id, rep_number, timestamp, 
                       features_json, is_correct_form, incorrect_duration
                FROM SessionDetails 
                WHERE session_id = %s 
                ORDER BY rep_number
            """, (session_id,))
            details = []
            for row in cur.fetchall():
                detail = cls(
                    detail_id=row[0], session_id=row[1], rep_number=row[2],
                    timestamp=row[3], features_json=row[4], is_correct_form=row[5],
                    incorrect_duration=row[6]
                )
                details.append(detail)
            return details
    finally:
        conn.close()
```

**Usage**: `session.get_details()` provides rep-by-rep analysis data.

### 2. **System Feedback Retrieval**
**Location**: `src/database/system_feedback.py:59-82`

```python
@classmethod
def get_by_session(cls, session_id: int) -> List['SystemFeedback']:
    """Get all feedback for a session"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT feedback_id, session_id, message, feedback_type,
                       related_rep, timestamp
                FROM SystemFeedback 
                WHERE session_id = %s 
                ORDER BY timestamp
            """, (session_id,))
            feedback_list = []
            for row in cur.fetchall():
                feedback = cls(
                    feedback_id=row[0], session_id=row[1], message=row[2],
                    feedback_type=row[3], related_rep=row[4], timestamp=row[5]
                )
                feedback_list.append(feedback)
            return feedback_list
    finally:
        conn.close()
```

**Usage**: `session.get_feedback()` provides form correction history.

### 3. **Workout Context Retrieval**
**Location**: `src/database/workout.py:84-102`

```python
@classmethod
def get_by_user(cls, user_id: int, limit: int = 10) -> List['Workout']:
    """Get workouts by user"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT workout_id, user_id, workout_name, planned_date,
                       actual_date, workout_status, notes, created_at, updated_at
                FROM Workout 
                WHERE user_id = %s 
                ORDER BY actual_date DESC, planned_date DESC 
                LIMIT %s
            """, (user_id, limit))
            workouts = []
            for row in cur.fetchall():
                workout = cls(
                    workout_id=row[0], user_id=row[1], workout_name=row[2],
                    planned_date=row[3], actual_date=row[4], workout_status=row[5],
                    notes=row[6], created_at=row[7], updated_at=row[8]
                )
                workouts.append(workout)
            return workouts
    finally:
        conn.close()
```

**Usage**: Groups individual sessions within structured workouts.

---

## Video Integration System

### 1. **Video Serving Backend**
**Location**: `src/app/video_streamer.py:350-397`

```python
@app.route("/serve_video/<filename>", methods=["GET"])
def serve_video(filename):
    """Serve analyzed video files."""
    try:
        videos_dir = os.path.join(os.path.dirname(__file__), "..", "..", "videos")
        video_path = os.path.join(videos_dir, filename)
        
        # Security check: ensure the file is within the videos directory
        if not os.path.abspath(video_path).startswith(os.path.abspath(videos_dir)):
            return "Access denied", 403
            
        # Check if file exists
        if not os.path.exists(video_path):
            return "Video not found", 404
            
        # Serve the video file
        response = send_file(
            video_path,
            mimetype='video/mp4',
            as_attachment=False,
            download_name=filename
        )
        
        # Set headers for video streaming
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Content-Disposition'] = f'inline; filename="{filename}"'
        
        return response
        
    except Exception as e:
        return f"Server error: {str(e)}", 500
```

**Security Features**:
- Path traversal protection
- File existence validation
- Proper MIME type handling
- Streaming headers for video playback

### 2. **Video Display Logic**
```python
if sess.video_path and os.path.exists(sess.video_path):
    st.video(sess.video_path)
else:
    st.info("Video not available or still processing.")
```

**Fallback Handling**: Graceful degradation when videos are unavailable.

---

## Transaction Execution Flow

### Phase 1: Page Navigation
```python
# User clicks "📜 Session History" button
with col3:
    st.button("📜 Session History", on_click=set_page, args=("History",))
```

### Phase 2: Authentication Check
```python
user_id = st.session_state.get("user_id")
if not user_id:
    st.warning("You must be logged in to view your session history.")
    return
```

### Phase 3: Data Retrieval
```python
sessions = Session.load_by_user(user_id)
if not sessions:
    st.info("No sessions found.")
    return
```

### Phase 4: UI Rendering
```python
for sess in sessions:
    # Display session metadata
    st.markdown(f"**Session ID:** {sess.session_id}")
    st.markdown(f"**Date:** {format_datetime(sess.start_time)}")
    st.markdown(f"**Exercise ID:** {sess.exercise_id}")
    st.markdown(f"**Reps:** {sess.reps_count}")
    st.markdown(f"**Duration (sec):** {sess.duration_sec}")
    
    # Video integration
    if sess.video_path and os.path.exists(sess.video_path):
        st.video(sess.video_path)
    else:
        st.info("Video not available or still processing.")
    
    st.markdown("---")  # Visual separator
```

---

## Error Handling Scenarios

### 1. **Unauthenticated Access**
```python
if not user_id:
    st.warning("You must be logged in to view your session history.")
```

**Response**: Warning message with login requirement.

### 2. **Database Connection Failure**
```python
try:
    sessions = Session.load_by_user(user_id)
except Exception as e:
    st.error(f"Error loading session history: {e}")
```

**Response**: User-friendly error message without technical details.

### 3. **No Sessions Found**
```python
if not sessions:
    st.info("No sessions found.")
```

**Response**: Informational message for empty state.

### 4. **Video File Missing**
```python
if sess.video_path and os.path.exists(sess.video_path):
    st.video(sess.video_path)
else:
    st.info("Video not available or still processing.")
```

**Response**: Graceful fallback with explanatory message.

### 5. **Video Serving Errors**
```python
# In video_streamer.py
if not os.path.exists(video_path):
    return "Video not found", 404

# Security check
if not os.path.abspath(video_path).startswith(os.path.abspath(videos_dir)):
    return "Access denied", 403
```

**Response**: HTTP error codes with appropriate messages.

---

## Performance Considerations

### 1. **Query Optimization**
```sql
-- Efficient query with proper indexing
SELECT session_id, workout_id, exercise_id, user_id, session_order,
       start_time, end_time, duration_sec, planned_reps, actual_reps,
       session_status, video_path, created_at, updated_at
FROM Session 
WHERE user_id = %s 
ORDER BY start_time DESC 
LIMIT %s
```

**Optimizations**:
- Indexed WHERE clause on user_id
- LIMIT clause prevents large result sets
- Specific column selection (no SELECT *)

### 2. **Pagination Strategy**
```python
def load_by_user(cls, user_id: int, limit: int = 10) -> List['Session']:
```

**Benefits**:
- Configurable page size
- Reduced memory usage
- Faster initial page load

### 3. **Lazy Loading Video Validation**
```python
if sess.video_path and os.path.exists(sess.video_path):
    st.video(sess.video_path)
```

**Optimization**: File existence check only when needed.

### 4. **Connection Management**
```python
conn = get_connection()
try:
    # Database operations
finally:
    conn.close()
```

**Benefit**: Proper connection cleanup prevents resource leaks.

---

## User Experience Features

### 1. **Chronological Organization**
```sql
ORDER BY start_time DESC
```

**Benefit**: Most recent sessions appear first for relevance.

### 2. **Rich Session Metadata**
```python
st.markdown(f"**Session ID:** {sess.session_id}  ")
st.markdown(f"**Date:** {format_datetime(sess.start_time)}  ")
st.markdown(f"**Exercise ID:** {sess.exercise_id}  ")
st.markdown(f"**Reps:** {sess.reps_count}  ")
st.markdown(f"**Duration (sec):** {sess.duration_sec}  ")
```

**Information Provided**:
- Unique session identifier
- Human-readable timestamp
- Exercise type identification
- Performance metrics (reps, duration)

### 3. **Integrated Video Playback**
```python
st.video(sess.video_path)
```

**Features**:
- Native Streamlit video player
- Automatic codec detection
- Playback controls (play, pause, seek)

### 4. **Visual Separation**
```python
st.markdown("---")  # Horizontal divider between sessions
```

**UX Benefit**: Clear visual separation between session entries.

### 5. **Error Feedback**
```python
st.info("Video not available or still processing.")
st.error(f"Error loading session history: {e}")
st.warning("You must be logged in to view your session history.")
```

**Message Types**:
- Info: Non-critical information
- Warning: User action required
- Error: System failure notification

---

## Security Considerations

### 1. **User Isolation**
```sql
WHERE user_id = %s
```

**Protection**: Users can only access their own session data.

### 2. **Video File Security**
```python
# Path traversal protection
if not os.path.abspath(video_path).startswith(os.path.abspath(videos_dir)):
    return "Access denied", 403
```

**Protection**: Prevents access to files outside videos directory.

### 3. **SQL Injection Prevention**
```python
cur.execute("""
    SELECT ... FROM Session WHERE user_id = %s ORDER BY start_time DESC LIMIT %s
""", (user_id, limit))
```

**Protection**: Parameterized queries prevent SQL injection.

### 4. **Authentication Enforcement**
```python
if not user_id:
    st.warning("You must be logged in to view your session history.")
```

**Protection**: Session state authentication required.

---

## Current Display Format

### Session Card Layout
```
**Session ID:** 123
**Date:** 2025-06-15 14:30:22
**Exercise ID:** 2
**Reps:** 15
**Duration (sec):** 120
[Video Player or "Video not available"]
---
```

### Limitations of Current Display
1. **Basic Format**: Simple text-based display
2. **No Analytics**: No performance trends or comparisons
3. **Limited Filtering**: No search or filter capabilities
4. **No Exercise Names**: Shows exercise ID instead of readable names
5. **No Detailed Analysis**: No access to rep-by-rep or feedback data

---

## Enhancement Opportunities

### 1. **Rich Data Visualization**
```python
# Potential improvements
import plotly.express as px

# Performance trends over time
fig = px.line(x=dates, y=reps, title="Rep Count Progression")
st.plotly_chart(fig)

# Exercise distribution
exercise_counts = sessions.groupby('exercise_name').size()
fig = px.pie(values=exercise_counts.values, names=exercise_counts.index)
st.plotly_chart(fig)
```

### 2. **Advanced Filtering**
```python
# Filter options
selected_exercise = st.selectbox("Filter by Exercise", ["All"] + exercise_list)
date_range = st.date_input("Date Range", value=[start_date, end_date])
min_reps = st.slider("Minimum Reps", 0, 50, 0)
```

### 3. **Detailed Session Analysis**
```python
# Expandable session details
with st.expander(f"Session {sess.session_id} Details"):
    # Rep-by-rep analysis
    details = sess.get_details()
    for detail in details:
        st.write(f"Rep {detail.rep_number}: {'✅' if detail.is_correct_form else '❌'}")
    
    # Feedback timeline
    feedback = sess.get_feedback()
    for fb in feedback:
        st.write(f"{fb.timestamp}: {fb.message}")
```

### 4. **Exercise Name Resolution**
```python
# Join with Exercise table for readable names
SELECT s.*, e.name as exercise_name 
FROM Session s 
JOIN Exercise e ON s.exercise_id = e.exercise_id 
WHERE s.user_id = %s 
ORDER BY s.start_time DESC
```

### 5. **Export Functionality**
```python
# Data export options
if st.button("Export Session Data"):
    df = pd.DataFrame([sess.to_dict() for sess in sessions])
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "session_history.csv", "text/csv")
```

---

## Integration Points

### 1. **Streamlit Session State**
```python
user_id = st.session_state.get("user_id")
```

### 2. **Database Models**
```python
from src.database.session import Session
from src.database.session_details import SessionDetails
from src.database.system_feedback import SystemFeedback
```

### 3. **Video Serving System**
```python
# Integration with Flask backend for video streaming
st.video(sess.video_path)  # Uses video_streamer.py endpoints
```

### 4. **Navigation System**
```python
st.button("⬅️ Back to Home", on_click=set_page, args=("Home",))
```

---

## Testing & Validation

### 1. **Data Retrieval Testing**
- User session filtering accuracy
- Temporal ordering verification
- Pagination limit enforcement
- Error handling for missing data

### 2. **Video Integration Testing**
- Video file availability checking
- Streamlit video player compatibility
- File security validation
- Error handling for missing videos

### 3. **UI Testing**
- Authentication flow validation
- Error message display accuracy
- Navigation functionality
- Responsive layout verification

### 4. **Performance Testing**
- Large session list handling
- Database query performance
- Video loading performance
- Memory usage optimization

---

## Summary

The Session History Retrieval Transaction provides essential user access to their exercise history with integrated video playback and comprehensive session metadata. While functionally complete, the current implementation offers opportunities for enhanced analytics, filtering, and detailed performance visualization.

**Key Strengths**:
- Secure user data isolation
- Comprehensive session metadata display
- Integrated video playback capability
- Robust error handling and graceful degradation
- Clean, chronological organization

**Areas for Improvement**:
- Enhanced data visualization with charts and trends
- Advanced filtering and search capabilities
- Detailed rep-by-rep analysis display
- Exercise name resolution for better readability
- Data export functionality for user convenience