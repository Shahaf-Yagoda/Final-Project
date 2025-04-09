

📘 **ERD: Training Platform – Schema Documentation**

This project generates an **Entity-Relationship Diagram (ERD)** for a fitness training platform.  
The schema is designed to support:

- Real-time pose analysis and session feedback
- Automated system-generated corrections
- Trainer/user comments
- Personalized workout data
- Structured session tracking

The ERD is exported in both PNG and PDF formats.

---

🔹 **User Table**

Represents every user on the platform — trainee, trainer, or admin.

**Fields:**

- `user_id`: int (Primary Key) – Unique identifier  
- `email`: string – Used for login  
- `password`: string – Hashed password (not stored in plain text)  
- `registration_date`: datetime – When the account was created  
- `profile_data`: json – Flexible user info (name, height, gender, etc.)  
- `role`: json – Role and permission set

Example `User`:
```

{
  "user_id": 1,
  "email": "trainer.lior@fitapp.com",
  "password": "hashed_pw_123",
  "registration_date": "2024-11-12T09:21:00",
  "profile_data": {
    "first_name": "Lior",
    "last_name": "Ben-David",
    "birth_date": "1993-10-15",
    "height_cm": 178,
    "weight_kg": 72,
    "gender": "male"
  },
  "role": {
    "role": "trainer",
    "permissions": [
      "view_sessions",
      "give_feedback",
      "access_dashboard"
    ]
  }
}

```

---

🔹 **Session Table**

Tracks each workout session performed by a user.

**Fields:**

- `session_id`: int (Primary Key)  
- `user_id`: int (Foreign Key → User)  
- `exercise_id`: int (Foreign Key → Exercise)  
- `start_time`: datetime  
- `end_time`: datetime  
- `duration_sec`: int – Total duration in seconds  
- `reps_count`: int – Total reps performed  
- `feedback_count`: int – Number of feedback entries  
- `performance_score`: float – Overall score (0–100)  
- `video_path`: string – URL or storage path to the session video

Example `Session`:
```
{
  "session_id": 101,
  "user_id": 5,
  "exercise_id": 3,
  "start_time": "2025-04-08T15:30:00",
  "end_time": "2025-04-08T15:42:30",
  "duration_sec": 750,
  "reps_count": 24,
  "feedback_count": 4,
  "performance_score": 88.5,
  "video_path": "s3://fitapp-recordings/u5/session101.mp4"
}
```


---

🔹 **Exercise Table**

Contains all exercises available on the platform.

**Fields:**

- `exercise_id`: int (Primary Key)  
- `name`: string – Exercise name (e.g., "Overhead Press")  
- `description`: string – Optional details

Example `Exercise`:

```
{
  "exercise_id": 3,
  "name": "Overhead Press",
  "description": "A shoulder exercise where you lift a weight overhead while keeping your core engaged."
}
```
---

🔹 **SessionDetails Table**

Captures rep-by-rep or frame-by-frame data during a session.

**Fields:**

- `detail_id`: int (Primary Key)  
- `session_id`: int (Foreign Key → Session)  
- `timestamp`: datetime – Frame or rep timestamp  
- `rep_num`: int – Repetition number  
- `keypoints_json`: json – Raw joint positions (e.g., from MediaPipe)  
- `features_json`: json – Calculated metrics like angles or posture flags  
- `is_correct`: boolean – Whether posture was correct  
- `incorrect_duration`: float – Duration of incorrect form (in seconds)

Example `SessionDetails`:
```
{
  "detail_id": 5231,
  "session_id": 101,
  "timestamp": "2025-04-08T15:34:12.042",
  "rep_num": 4,
  "keypoints_json": {
    "shoulder_l": [0.45, 0.31],
    "elbow_l": [0.47, 0.52],
    "wrist_l": [0.48, 0.74],
    "shoulder_r": [0.55, 0.30],
    "elbow_r": [0.56, 0.51],
    "wrist_r": [0.57, 0.72]
  },
  "features_json": {
    "elbow_angle_l": 171.2,
    "elbow_angle_r": 169.8,
    "back_angle_l": 166.5,
    "back_angle_r": 170.3,
    "wrist_above_l": 1,
    "wrist_above_r": 1
  },
  "is_correct": false,
  "incorrect_duration": 1.73
}

```

---

🔹 **SystemFeedback Table**

System-generated (AI) feedback provided during a session.

**Fields:**

- `feedback_id`: int (Primary Key)  
- `session_id`: int (Foreign Key → Session)  
- `timestamp`: datetime – When the feedback was created  
- `message`: text – Feedback explanation

Example `SystemFeedback`:
```
{
  "feedback_id": 303,
  "session_id": 101,
  "timestamp": "2025-04-08T15:35:10",
  "message": "Back arch detected — tighten your core and engage glutes."
}

{
  "feedback_id": 304,
  "session_id": 101,
  "timestamp": "2025-04-08T15:38:21",
  "message": "Wrist position too far forward — keep aligned above elbows."
}

```

---

🔹 **Comment Table**

Human-written feedback — typically from trainers.

**Fields:**

- `comment_id`: int (Primary Key)  
- `session_id`: int (Foreign Key → Session)  
- `user_id`: int (Foreign Key → User) – Author of the comment  
- `timestamp`: datetime  
- `comment`: text – The written feedback

Example `Comment`:
```
{
  "comment_id": 88,
  "session_id": 101,
  "user_id": 1,
  "timestamp": "2025-04-08T16:05:12",
  "comment": "Great job keeping your back straight in the last few reps!"
}

{
  "comment_id": 89,
  "session_id": 101,
  "user_id": 12,
  "timestamp": "2025-04-08T16:10:45",
  "comment": "Nice work! That last rep was really clean 👏"
}

```

> Note: While comments are attributed to users, they are considered part of the session context. The ERD only shows the direct relationship between Session → Comment.

---

🔗 **Entity Relationships**

- A **User** can have many **Sessions**
- A **Session** can have:
  - Many **SessionDetails**
  - Many **SystemFeedback** entries
  - Many **Comments**
- An **Exercise** can be linked to many **Sessions**

Visualized:

```
User      1 ─── N     Session
Exercise  1 ─── N     Session
Session   1 ─── N     SessionDetails
Session   1 ─── N     SystemFeedback
Session   1 ─── N     Comment
```

---

📂 **How to Generate the ERD**

To generate and export the diagram:

```bash
python generate_erd.py
```

This will output:

- `training_erd_png.png`
- `training_erd_pdf.pdf`

These contain a visual representation of the database schema, relationships, and field types.

---