# Right Motion - UML Class Diagrams

## Overview

This document contains the UML Class Diagrams for the Right Motion fitness tracking system, showing the relationships between database models, processing classes, and system components.

---

## Core Database Models Class Diagram

```
┌─────────────────────────────────────┐
│                User                 │
├─────────────────────────────────────┤
│ - user_id: int (PK)                │
│ - email: str (UNIQUE)              │
│ - username: str (UNIQUE)           │
│ - password: str                    │
│ - first_name: str                  │
│ - last_name: str                   │
│ - registration_date: date          │
│ - registration_time: time          │
│ - profile_data: JSONB              │
│ - user_type: user_role_enum        │
│ - is_active: bool                  │
│ - last_login: datetime             │
│ - created_at: datetime             │
│ - updated_at: datetime             │
├─────────────────────────────────────┤
│ + register(email, username, pwd)   │
│ + authenticate(identifier, pwd)    │
│ + get_by_id(user_id)              │
│ + hash_password(password)          │
│ + verify_password(plain, hash)     │
│ + to_dict()                        │
│ + get_full_name()                  │
│ + deactivate()                     │
│ + update_profile(data)             │
│ + change_role(role)                │
└─────────────────────────────────────┘
                    │
                    │ 1
                    │
                    │ *
┌─────────────────────────────────────┐
│               Workout               │
├─────────────────────────────────────┤
│ - workout_id: int (PK)             │
│ - user_id: int (FK)                │
│ - workout_name: str                │
│ - planned_date: date               │
│ - actual_date: date                │
│ - workout_status: workout_status   │
│ - notes: str                       │
│ - created_at: datetime             │
│ - updated_at: datetime             │
├─────────────────────────────────────┤
│ + create(user_id, name)            │
│ + get_by_user(user_id)             │
│ + get_by_id(workout_id)            │
│ + get_sessions()                   │
│ + complete()                       │
│ + to_dict()                        │
└─────────────────────────────────────┘
                    │
                    │ 1
                    │
                    │ *
┌─────────────────────────────────────┐
│               Session               │
├─────────────────────────────────────┤
│ - session_id: int (PK)             │
│ - workout_id: int (FK, NULL)       │
│ - exercise_id: int (FK)            │
│ - user_id: int (FK)                │
│ - session_order: int               │
│ - start_time: datetime             │
│ - end_time: datetime               │
│ - duration_sec: int                │
│ - planned_reps: int                │
│ - actual_reps: int                 │
│ - session_status: session_status   │
│ - video_path: str                  │
│ - created_at: datetime             │
│ - updated_at: datetime             │
│ - reps_count: int (legacy)         │
│ - feedback_count: int (legacy)     │
│ - performance_score: float         │
├─────────────────────────────────────┤
│ + save()                           │
│ + create(exercise_id, user_id)     │
│ + load_by_user(user_id)            │
│ + load_by_id(session_id)           │
│ + get_by_workout(workout_id)       │
│ + finish(reps, end_time)           │
│ + get_details()                    │
│ + get_feedback()                   │
│ + to_dict()                        │
└─────────────────────────────────────┘
                    │
                    │ 1
                    │
                    │ *
┌─────────────────────────────────────┐
│            SessionDetails           │
├─────────────────────────────────────┤
│ - detail_id: int (PK)              │
│ - session_id: int (FK)             │
│ - rep_number: int                  │
│ - timestamp: datetime              │
│ - features_json: JSONB             │
│ - is_correct_form: bool            │
│ - incorrect_duration: float        │
├─────────────────────────────────────┤
│ + create(session_id, rep_num)      │
│ + get_by_session(session_id)       │
│ + get_session_summary(session_id)  │
│ + get_by_rep(session_id, rep)      │
│ + to_dict()                        │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│            SystemFeedback           │
├─────────────────────────────────────┤
│ - feedback_id: int (PK)            │
│ - session_id: int (FK)             │
│ - message: str                     │
│ - feedback_type: feedback_type     │
│ - related_rep: int (NULL)          │
│ - timestamp: datetime              │
├─────────────────────────────────────┤
│ + create(session_id, message)      │
│ + get_by_session(session_id)       │
│ + get_feedback_summary(session_id) │
│ + get_form_corrections(session_id) │
│ + to_dict()                        │
└─────────────────────────────────────┘
                    │
                    │ *
                    │
                    │ 1
┌─────────────────────────────────────┐
│               Exercise              │
├─────────────────────────────────────┤
│ - exercise_id: int (PK)            │
│ - name: str (UNIQUE)               │
│ - description: str                 │
│ - target_muscles: JSONB            │
│ - instructions: str                │
│ - created_at: datetime             │
│ - updated_at: datetime             │
├─────────────────────────────────────┤
│ + get_by_name(name)                │
│ + get_all()                        │
│ + create(name, description)        │
│ + to_dict()                        │
└─────────────────────────────────────┘
```

---

## Exercise Processing Classes

```
┌─────────────────────────────────────┐
│         BaseExerciseChecker         │
│            <<abstract>>             │
├─────────────────────────────────────┤
│ # state: dict                      │
│ # exercise_name: str               │
├─────────────────────────────────────┤
│ + check_form(frame, landmarks,     │
│   state): tuple[list, int]         │
│ + init_state(): dict               │
│ # calculate_angle(p1, p2, p3)      │
│ # is_point_visible(landmark)       │
│ # speak_async(message)             │
└─────────────────────────────────────┘
                    △
                    │
        ┌───────────┼───────────┐
        │           │           │
        │           │           │
┌───────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  LungeChecker │ │OverheadPress    │ │  PlankChecker   │
│               │ │    Checker      │ │                 │
├───────────────┤ ├─────────────────┤ ├─────────────────┤
│ - rep_count   │ │ - rep_count     │ │ - start_time    │
│ - stage       │ │ - stage         │ │ - duration      │
│ - last_msg    │ │ - last_msg      │ │ - last_msg      │
│ - last_time   │ │ - last_time     │ │ - last_time     │
├───────────────┤ ├─────────────────┤ ├─────────────────┤
│ + check_form()│ │ + check_form()  │ │ + check_form()  │
│ + init_state()│ │ + init_state()  │ │ + init_state()  │
│ - check_lunge │ │ - check_press   │ │ - check_plank   │
│ - detect_side │ │ - detect_ready  │ │ - format_time   │
└───────────────┘ └─────────────────┘ └─────────────────┘
```

---

## Pose Detection and Processing

```
┌─────────────────────────────────────┐
│         PoseDetectorFactory         │
│            <<factory>>              │
├─────────────────────────────────────┤
│ + LIVE_EXERCISE_CONFIG: dict       │
│ + VIDEO_ANALYSIS_CONFIG: dict      │
│ + DEFAULT_CONFIG: dict             │
├─────────────────────────────────────┤
│ + create_pose_detector(config)     │
│ + create_live_exercise_detector()  │
│ + create_video_analysis_detector() │
│ + create_default_detector()        │
└─────────────────────────────────────┘
                    │
                    │ creates
                    ▼
┌─────────────────────────────────────┐
│           MediaPipe.Pose            │
│           <<external>>              │
├─────────────────────────────────────┤
│ - static_image_mode: bool          │
│ - model_complexity: int            │
│ - enable_segmentation: bool        │
│ - min_detection_confidence: float  │
│ - min_tracking_confidence: float   │
├─────────────────────────────────────┤
│ + process(image): PoseResults      │
│ + close()                          │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│           ExerciseSession           │
├─────────────────────────────────────┤
│ - exercise_name: str               │
│ - user_id: int                     │
│ - show_window: bool                │
│ - save_keypoints: bool             │
│ - save_video: bool                 │
│ - state: dict                      │
│ - pose_detector: mp.Pose           │
│ - start_time: datetime             │
│ - end_time: datetime               │
│ - video_writer: cv2.VideoWriter    │
│ - temp_video_path: str             │
│ - final_video_path: str            │
│ - session_details: list            │
│ - feedback_data: list              │
├─────────────────────────────────────┤
│ + run(): dict                      │
│ + setup_video_recording(cap)       │
│ + finalize_video_recording()       │
│ + save_session_detail()            │
│ + process_frame(frame, landmarks)  │
│ + save_session_to_database(reps)   │
└─────────────────────────────────────┘
```

---

## Video Streaming and Web Interface

```
┌─────────────────────────────────────┐
│        VideoStreamManager           │
├─────────────────────────────────────┤
│ - mp_pose: mp.solutions.pose       │
│ - mp_drawing: mp.drawing_utils     │
│ - pose_detector: mp.Pose           │
│ - session_states: dict             │
│ - video_writers: dict              │
│ - video_temp_paths: dict           │
│ - feedback_throttle: dict          │
├─────────────────────────────────────┤
│ + get_session_key(user_id, ex)     │
│ + get_or_create_session_state()    │
│ + set_session_active(active)       │
│ + save_reps_to_tempfile()          │
│ + append_session_detail()          │
│ + should_throttle_feedback()       │
│ + append_feedback()                │
│ + process_frame()                  │
│ + setup_video_recording()          │
│ + generate_stream()                │
│ + stop_session()                   │
└─────────────────────────────────────┘
                    │
                    │ uses
                    ▼
┌─────────────────────────────────────┐
│             Flask App               │
│           <<framework>>             │
├─────────────────────────────────────┤
│ + index()                          │
│ + video_feed()                     │
│ + stop_session()                   │
│ + serve_video(filename)            │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│           Streamlit App             │
│           <<framework>>             │
├─────────────────────────────────────┤
│ - session_state: dict              │
├─────────────────────────────────────┤
│ + show_registration_page()         │
│ + show_login_page()                │
│ + show_live_exercise_page()        │
│ + show_video_analysis_page()       │
│ + show_history_page()              │
│ + format_datetime(dt)              │
│ + set_page(page_name)              │
│ + logout()                         │
└─────────────────────────────────────┘
```

---

## Database Connection and Utilities

```
┌─────────────────────────────────────┐
│        DatabaseConnection           │
├─────────────────────────────────────┤
│ + get_connection(): psycopg2.conn  │
└─────────────────────────────────────┘
                    │
                    │ provides
                    ▼
┌─────────────────────────────────────┐
│            DatabaseUtils            │
├─────────────────────────────────────┤
│ + save_session_to_db()             │
│ + save_session_details_to_db()     │
│ + save_system_feedback_to_db()     │
│ + get_exercise_id_by_name()        │
│ + get_user_sessions()              │
│ + get_session_details()            │
│ + get_session_feedback()           │
└─────────────────────────────────────┘
```

---

## Class Relationships Summary

### Inheritance Relationships
1. **BaseExerciseChecker** ← **LungeChecker, OverheadPressChecker, PlankChecker**
   - All exercise checkers inherit from the abstract base class

### Composition Relationships
1. **User** ◆—→ **Workout** (1:*)
   - User has multiple workouts
2. **Workout** ◆—→ **Session** (1:*)
   - Workout contains multiple sessions
3. **Session** ◆—→ **SessionDetails** (1:*)
   - Session contains detailed rep-by-rep data
4. **Session** ◆—→ **SystemFeedback** (1:*)
   - Session contains feedback messages
5. **ExerciseSession** ◆—→ **BaseExerciseChecker**
   - ExerciseSession uses exercise checker for form analysis

### Association Relationships
1. **Exercise** ←→ **Session** (1:*)
   - Exercise type is referenced by sessions
2. **User** ←→ **Session** (1:*)
   - User performs multiple sessions
3. **VideoStreamManager** ←→ **ExerciseSession**
   - Both handle exercise session processing

### Dependency Relationships
1. **PoseDetectorFactory** → **MediaPipe.Pose**
   - Factory creates MediaPipe pose detectors
2. **All Models** → **DatabaseConnection**
   - All database models depend on connection utility
3. **Streamlit App** → **All Models**
   - Web interface depends on all database models

---

## Design Patterns Used

### 1. **Factory Pattern**
- **PoseDetectorFactory**: Creates different configurations of MediaPipe pose detectors
- **get_exercise_checker()**: Returns appropriate exercise checker based on exercise type

### 2. **Template Method Pattern**
- **BaseExerciseChecker**: Defines the algorithm structure, subclasses implement specific steps

### 3. **Active Record Pattern**
- **User, Session, SessionDetails, SystemFeedback**: Database models with built-in persistence methods

### 4. **Singleton Pattern**
- **VideoStreamManager**: Global instance for managing video streams

### 5. **Strategy Pattern**
- **Exercise Checkers**: Different strategies for analyzing different exercise types

---

## Key Enumerations

```
┌─────────────────────────────────────┐
│          user_role_enum             │
├─────────────────────────────────────┤
│ + user                             │
│ + coach                            │
│ + admin                            │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│         workout_status_enum         │
├─────────────────────────────────────┤
│ + planned                          │
│ + in_progress                      │
│ + completed                        │
│ + cancelled                        │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│         session_status_enum         │
├─────────────────────────────────────┤
│ + in_progress                      │
│ + completed                        │
│ + paused                           │
│ + cancelled                        │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│         feedback_type_enum          │
├─────────────────────────────────────┤
│ + form_correction                  │
│ + encouragement                    │
│ + warning                          │
│ + achievement                      │
└─────────────────────────────────────┘
```

---

## Architecture Notes

### **Layer Separation**
1. **Presentation Layer**: Streamlit UI components
2. **Business Logic Layer**: Exercise checkers, session management
3. **Data Access Layer**: Database models and utilities
4. **External Systems**: MediaPipe, OpenCV, PostgreSQL

### **Key Features**
- **Type Safety**: Full type hints throughout the codebase
- **Error Handling**: Comprehensive exception handling in all layers
- **Legacy Support**: Backward compatibility with older database schemas
- **Modularity**: Clear separation of concerns between components
- **Extensibility**: Easy to add new exercise types and form checkers

This class diagram provides a complete overview of the Right Motion system's object-oriented design, showing how the various components interact to provide comprehensive fitness tracking functionality.