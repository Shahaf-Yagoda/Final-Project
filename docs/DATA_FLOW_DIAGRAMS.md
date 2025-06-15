# Right Motion - Data Flow Diagrams (DFD)

## DFD Level 0 - Context Diagram

```
                         ┌─────────────────┐
                         │   E1 - User     │
                         └─────────┬───────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    │ T4,T5        │ T1,T2        │ T3
                    ▼              ▼              ▼
        ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
        │ Registration/   │ │  Live Exercise  │ │ Session History │
        │ Authentication  │ │   Tracking      │ │   Viewing       │
        └─────────┬───────┘ └─────────┬───────┘ └─────────┬───────┘
                  │                   │                   │
                  └───────────────────┼───────────────────┘
                                      │
                             ┌────────▼────────┐
                             │                 │
                             │  Right Motion   │
                             │ Fitness System  │
                             │                 │
                             └────────┬────────┘
                                      │
                        ┌─────────────┼─────────────┐
                        │             │             │
                        ▼             ▼             ▼
                ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
                │ E2-Camera   │ │  Database   │ │E3-File Sys  │
                │  Hardware   │ │   Storage   │ │   Storage   │
                └─────────────┘ └─────────────┘ └─────────────┘
```

---

## DFD Level 1 - System Overview

```
┌─────────────────┐
│   E1 - User     │
└─────────┬───────┘
          │
          │ T4,T5: registration/login
          ▼
    ┌──────────┐    user_data     ┌─────────────────┐
    │   P1     │◄─────────────────│      DS1        │
    │   User   │                  │   User Store    │
    │   Mgmt   │─────────────────►│                 │
    └─────┬────┘    auth_status   └─────────────────┘
          │
          │ T1: start_workout / T2: upload_video
          ▼
    ┌──────────┐    exercise_data ┌─────────────────┐
    │   P2     │◄─────────────────│      DS2        │
    │ Exercise │                  │ Exercise Store  │
    │ Session  │─────────────────►│                 │
    └─────┬────┘   session_data   └─────────────────┘
          │                      
          │ T7: session_end       ┌─────────────────┐
          ├──────────────────────►│      DS3        │
          │                      │ Session Store   │
          │ session_details      │                 │
          ├─────────────────────►└─────────────────┘
          │                      
          │ feedback_data        ┌─────────────────┐
          ├─────────────────────►│      DS4        │
          │                      │Session Details  │
          │ T6: pose_correction  │     Store       │
          ├─────────────────────►└─────────────────┘
          │                      
          │ system_feedback      ┌─────────────────┐
          ├─────────────────────►│      DS5        │
          │                      │System Feedback  │
          │                      │     Store       │
          │                      └─────────────────┘
          │
          │ processed_video
          ▼
    ┌──────────┐    video_files   ┌─────────────────┐
    │   P3     │◄─────────────────│      DS6        │
    │  Video   │                  │  Video Store    │
    │Analysis  │─────────────────►│                 │
    └─────┬────┘                  └─────────────────┘
          │
          │ T3: history_request
          ▼
    ┌──────────┐
    │   P4     │
    │ History  │
    │ Display  │
    └─────┬────┘
          │
          │ video_stream_request
          ▼
    ┌──────────┐    video_files   ┌─────────────────┐
    │   P5     │◄─────────────────│      DS6        │
    │ Video    │                  │  Video Store    │
    │Streaming │─────────────────►│                 │
    └──────────┘                  └─────────────────┘
```

---

## DFD Level 2 - Exercise Session Process (Process P2)

```
┌─────────────────┐
│   E1 - User     │
└─────────┬───────┘
          │
          │ T1: start_session(exercise_type, user_id)
          ▼
    ┌──────────┐    exercise_id   ┌─────────────────┐
    │  P2.1    │◄─────────────────│      DS2        │
    │ Session  │                  │ Exercise Store  │
    │  Init    │                  │                 │
    └─────┬────┘                  └─────────────────┘
          │
          │ camera_feed (from E2)
          ▼
    ┌──────────┐    pose_landmarks
    │  P2.2    │─────────────────┐
    │  Pose    │                 │
    │Detection │                 │
    └──────────┘                 │
                                 ▼
                         ┌──────────┐
                         │  P2.3    │
                         │  Form    │
                         │ Analysis │
                         └─────┬────┘
                               │
                               │ T6: incorrect_pose → feedback
                               │ rep_count, feedback
                               ▼
                         ┌──────────┐    session_data  ┌─────────────────┐
                         │  P2.4    │─────────────────►│      DS3        │
                         │ Rep      │                  │ Session Store   │
                         │Counting  │◄─────────────────│                 │
                         └─────┬────┘   session_id     └─────────────────┘
                               │
                               │ T7: session_end
                               ▼
                         ┌──────────┐    session_data  ┌─────────────────┐
                         │  P2.5    │─────────────────►│      DS4        │
                         │ Session  │                  │Session Details  │
                         │ Storage  │◄─────────────────│     Store       │
                         └─────┬────┘   detail_id      └─────────────────┘
                               │
                               │ processed_video
                               ▼
                         ┌──────────┐    video_file    ┌─────────────────┐
                         │  P2.6    │─────────────────►│      DS6        │
                         │  Video   │                  │  Video Store    │
                         │Recording │                  │                 │
                         └──────────┘                  └─────────────────┘
```

---

## DFD Level 2 - Video Analysis Process (Process P3)

```
┌─────────────────┐
│   E1 - User     │
└─────────┬───────┘
          │
          │ T2: video_file_upload(exercise_type, user_id)
          ▼
    ┌──────────┐    exercise_rules ┌─────────────────┐
    │   P3.1   │◄─────────────────│      DS2        │
    │  Video   │                  │ Exercise Store  │
    │ Upload   │                  │                 │
    └─────┬────┘                  └─────────────────┘
          │
          │ uploaded_video
          ▼
    ┌──────────┐
    │   P3.2   │
    │Frame-by- │
    │  Frame   │
    │Processing│
    └─────┬────┘
          │
          │ pose_landmarks_per_frame
          ▼
    ┌──────────┐
    │   P3.3   │
    │Timestamped│
    │ Feedback │
    │Generation│
    └─────┬────┘
          │
          │ analyzed_video + feedback
          ▼
    ┌──────────┐    processed_video ┌─────────────────┐
    │   P3.4   │─────────────────►  │      DS6        │
    │Processed │                    │  Video Store    │
    │  Video   │◄─────────────────  │                 │
    │ Storage  │    video_path      └─────────────────┘
    └─────┬────┘
          │
          │ session_data
          ▼
    ┌─────────────────┐
    │      DS3        │
    │ Session Store   │
    └─────────────────┘
```

---

## DFD Level 2 - Form Analysis Process (Process 2.3)

```
                    ┌─────────────────┐
                    │ pose_landmarks  │
                    │  (from 2.2)     │
                    └─────────┬───────┘
                              │
                              ▼
                        ┌──────────┐    exercise_rules ┌─────────────────┐
                        │  2.3.1   │◄─────────────────│       D2        │
                        │Exercise  │                  │ Exercise Store  │
                        │ Checker  │                  │                 │
                        └─────┬────┘                  └─────────────────┘
                              │
                              │ angle_calculations
                              ▼
                        ┌──────────┐
                        │  2.3.2   │
                        │   Rep    │
                        │ Counter  │
                        └─────┬────┘
                              │
                              │ rep_count
                              ▼
                        ┌──────────┐
                        │  2.3.3   │
                        │ Feedback │
                        │Generator │
                        └─────┬────┘
                              │
                              │ feedback_messages
                              ▼
                        ┌──────────┐
                        │  2.3.4   │
                        │ Session  │
                        │ Details  │
                        └─────┬────┘
                              │
                              │ session_details
                              ▼
                    ┌─────────────────┐
                    │   to Process    │
                    │      2.4        │
                    └─────────────────┘
```

---

## DFD Level 2 - User Management Process (Process P1)

```
┌─────────────────┐
│   E1 - User     │
└─────────┬───────┘
          │
          │ T4: registration_data
          ▼
    ┌──────────┐    encrypted_pwd   ┌─────────────────┐
    │  P1.1    │─────────────────►  │      DS1        │
    │  User    │   user_record      │   User Store    │
    │Register  │◄─────────────────  │                 │
    └──────────┘                    └─────────────────┘
          
          T5: login_credentials
          ▼
    ┌──────────┐    user_lookup     ┌─────────────────┐
    │  P1.2    │◄─────────────────  │      DS1        │
    │   User   │   auth_result      │   User Store    │
    │  Login   │─────────────────►  │                 │
    └─────┬────┘                    └─────────────────┘
          │
          │ auth_status
          ▼
    ┌──────────┐
    │  P1.3    │
    │ Session  │
    │  State   │
    │ Manager  │
    └──────────┘
```

---

## DFD Level 2 - History Display Process (Process P4)

```
┌─────────────────┐
│   E1 - User     │
└─────────┬───────┘
          │
          │ T3: history_request(user_id)
          ▼
    ┌──────────┐    user_sessions   ┌─────────────────┐
    │  P4.1    │◄─────────────────  │      DS3        │
    │ Session  │                    │ Session Store   │
    │Retrieval │                    │                 │
    └─────┬────┘                    └─────────────────┘
          │
          │ session_list
          ▼
    ┌──────────┐    session_details ┌─────────────────┐
    │  P4.2    │◄─────────────────  │      DS4        │
    │ Details  │                    │Session Details  │
    │Enrichment│                    │     Store       │
    └─────┬────┘                    └─────────────────┘
          │
          │ enriched_sessions
          ▼
    ┌──────────┐    video_files     ┌─────────────────┐
    │  P4.3    │◄─────────────────  │      DS6        │
    │  Video   │                    │  Video Store    │
    │Integration│                   │                 │
    └─────┬────┘                    └─────────────────┘
          │
          │ formatted_history
          ▼
    ┌──────────┐
    │  P4.4    │
    │    UI    │
    │ Display  │
    └──────────┘
```

---

## Data Stores Description

### DS1 - User Store (מאגר משתמשים)
- **Tables**: User
- **Data**: user_id, email, username, password_hash, profile_data, user_type, registration_data, authentication_data
- **Operations**: CREATE, READ, UPDATE (authentication, profile updates)

### DS2 - Exercise Store (מאגר תרגילים)
- **Tables**: Exercise, Workout
- **Data**: exercise_id, exercise_name, target_muscles, instructions, workout_plans, analysis_rules, performance_standards
- **Operations**: READ (exercise definitions, workout templates, analysis criteria)

### DS3 - Session Store (מאגר סשנים)
- **Tables**: Session
- **Data**: session_id, user_id, exercise_id, workout_id, timestamps, duration, session_status, video_path, metadata
- **Operations**: CREATE, READ (session tracking, history retrieval)

### DS4 - Session Details Store (מאגר פרטי סשן)
- **Tables**: SessionDetails
- **Data**: detail_id, session_id, rep_number, timestamp, features_json, is_correct_form, incorrect_duration, detailed_analysis
- **Operations**: CREATE, READ (rep-by-rep tracking, detailed performance analysis)

### DS5 - System Feedback Store (מאגר משוב מערכתי)
- **Tables**: SystemFeedback
- **Data**: feedback_id, session_id, message, feedback_type, related_rep, timestamp, correction_messages, recommendations
- **Operations**: CREATE, READ (real-time feedback, guidance messages)

### DS6 - Video Store (אחסון וידאו)
- **Location**: File system (/videos directory)
- **Data**: video_files, file_paths, metadata, processed_videos, temporary_files, video_metadata
- **Operations**: CREATE, READ (video recording, playback, file management)

---

## External Entities

### E1 - User (משתמש)
- **Description**: Main contact point with the system, performs registration, login and workouts
- **Input**: registration_data, login_credentials, exercise_requests, history_requests
- **Output**: authentication_status, exercise_feedback, session_history, video_playback

### E2 - Camera (מצלמה)
- **Description**: External hardware providing real-time video stream for pose detection
- **Output**: camera_feed, video_frames, real_time_video_stream
- **Type**: External hardware device

### E3 - File System (מערכת קבצים)
- **Description**: External storage for processed video files and multimedia data
- **Input**: processed_video_streams, temporary_files
- **Output**: stored_video_files, video_metadata
- **Type**: File system storage

---

## System Triggers (טריגרים)

### T1 - Start Workout Button Click (לחיצה על כפתור התחלת אימון)
- **Activates**: Live Exercise Session Process (P2)
- **Input**: exercise_type, user_id, session_options
- **Result**: Initiates real-time pose detection and form analysis

### T2 - Video File Upload (העלאת קובץ וידאו)
- **Activates**: Video Analysis Process (P3)
- **Input**: video_file, exercise_type, user_id
- **Result**: Frame-by-frame analysis and feedback generation

### T3 - History Button Click (לחיצה על היסטוריה)
- **Activates**: History Display Process (P4)
- **Input**: user_id, filter_criteria
- **Result**: Retrieval and display of session history

### T4 - Registration Form Submit (שליחת טופס רישום)
- **Activates**: User Registration Process (P1.1)
- **Input**: registration_data, profile_information
- **Result**: New user account creation

### T5 - Login Credentials Submit (שליחת נתוני התחברות)
- **Activates**: User Authentication Process (P1.2)
- **Input**: login_credentials (email/username, password)
- **Result**: User authentication and session establishment

### T6 - Incorrect Pose Detection (זיהוי תנוחה שגויה)
- **Activates**: Real-time Feedback Process
- **Input**: pose_landmarks, form_analysis_result
- **Result**: Immediate correction feedback

### T7 - Exercise Session End (סיום סשן אימון)
- **Activates**: Session Data Storage Process (P2.5)
- **Input**: session_summary, final_statistics
- **Result**: Complete session data saved to database

---

## Data Flow Definitions

| Data Flow | Description | Composition |
|-----------|-------------|-------------|
| user_credentials | Login information | email/username, password |
| exercise_request | Session start request | exercise_type, user_id, options |
| pose_data | MediaPipe landmarks | x,y,z coordinates, visibility |
| session_data | Complete session info | timestamps, reps, duration, status |
| feedback_data | Form corrections | messages, timestamps, rep_association |
| processed_video | Recorded session | video_file, pose_overlays, metadata |
| history_request | User history query | user_id, filters, pagination |

---

## DFD Level 2 - Video Streaming Process (Process P5)

```
┌─────────────────┐
│      User       │
└─────────┬───────┘
          │
          │ video_request(session_id, video_path)
          ▼
    ┌──────────┐    video_files     ┌─────────────────┐
    │   P5.1   │◄─────────────────  │      DS6        │
    │Real-time │                    │  Video Store    │
    │  Video   │                    │                 │
    │Streaming │                    └─────────────────┘
    └─────┬────┘
          │
          │ live_video_stream
          ▼
    ┌──────────┐
    │   P5.2   │
    │Processed │
    │  Video   │
    │ Serving  │
    └─────┬────┘
          │
          │ video_stream_response
          ▼
┌─────────────────┐
│   User Browser  │
│  Video Player   │
└─────────────────┘
```

---

## Process Definitions

| Process | Description | Inputs | Outputs |
|---------|-------------|--------|---------|
| P1 User Mgmt | Handle registration/login | credentials | auth_status |
| P1.1 User Registration | New user account creation | registration_data | user_record |
| P1.2 User Authentication | Login verification | login_credentials | auth_result |
| P1.3 User Profile Mgmt | Profile management | profile_updates | updated_profile |
| P2 Exercise Session | Live exercise tracking | exercise_request, camera_feed | session_data |
| P2.1 Session Init | Initialize exercise session | exercise_type, user_id | session_state |
| P2.2 Pose Detection | MediaPipe pose analysis | camera_feed | pose_landmarks |
| P2.3 Form Analysis | Exercise form checking | pose_landmarks | rep_count, feedback |
| P2.4 Rep Counting | Count exercise repetitions | angle_calculations | rep_count |
| P2.5 Session Storage | Save session data | session_details | session_id |
| P3 Video Analysis | Recorded video analysis | video_file | analyzed_video |
| P3.1 Video Upload | Handle video file upload | video_file, exercise_type | uploaded_video |
| P3.2 Frame Processing | Frame-by-frame analysis | video_frames | pose_landmarks_per_frame |
| P3.3 Timestamped Feedback | Generate timed feedback | analysis_results | timestamped_feedback |
| P3.4 Processed Video Storage | Save analyzed video | processed_video | video_path |
| P4 History Display | Show user history | history_request | formatted_history |
| P4.1 Session Retrieval | Fetch user sessions | user_id | session_list |
| P4.2 Workout List Display | Show workout history | sessions | workout_display |
| P4.3 Video & Stats Display | Show videos and statistics | session_data | enriched_display |
| P5 Video Streaming | Serve video files | video_request | video_stream |
| P5.1 Real-time Streaming | Live video streaming | video_files | live_stream |
| P5.2 Processed Video Serving | Serve analyzed videos | video_path | video_response |

This DFD structure provides a clear, hierarchical view of the Right Motion system's data flows while keeping it as simple as possible. Each level adds appropriate detail without overwhelming complexity.