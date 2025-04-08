# from graphviz import Digraph
#
# dot = Digraph(comment='ERD - Training System')
# dot.attr(rankdir='LR', fontsize='20')
#
#
# def add_table(name, fields):
#     label = f"<<TABLE BORDER='1' CELLBORDER='1' CELLSPACING='0'>"
#     label += f"<TR><TD COLSPAN='1' BGCOLOR='lightblue'><B>{name}</B></TD></TR>"
#     for field in fields:
#         label += f"<TR><TD ALIGN='LEFT'>{field}</TD></TR>"
#     label += "</TABLE>>"
#     dot.node(name, label=label, shape='plaintext')
#
#
# # הגדרת הטבלאות
# add_table('User', [
#     'user_id: int (PK)',
#     'name: string',
#     'email: string',
#     'date_of_birth: date',
#     'registration_date: datetime',
#     'role: string'
# ])
# add_table('Trainer', ['user_id: int (PK, FK)', 'specialty: string'])
# add_table('Trainee', ['user_id: int (PK, FK)', 'progress: json'])
# add_table('Admin', ['user_id: int (PK, FK)', 'permissions: json'])
#
# add_table('Session', [
#     'session_id: int (PK)',
#     'user_id: int (FK)',
#     'start_time: datetime',
#     'end_time: datetime',
#     'exercise_type: string',
#     'exercise_counter: int',
#     'feedback_count: int',
#     'performance_score: float',
#     'duration_seconds: int'
# ])
# add_table('Exercise', ['exercise_id: int (PK)', 'name: string'])
# add_table('Feedback', [
#     'feedback_id: int (PK)',
#     'session_id: int (FK)',
#     'trainer_id: int (FK)',
#     'comments: text',
#     'timestamp: datetime'
# ])
#
# # קשרים
# dot.edge('User', 'Trainer')
# dot.edge('User', 'Trainee')
# dot.edge('User', 'Admin')
# dot.edge('User', 'Session', label='1 ↔ N')
# dot.edge('Trainer', 'Feedback', label='1 ↔ N')
# dot.edge('Session', 'Feedback', label='1 ↔ N')
# dot.edge('Exercise', 'Session', label='1 ↔ N')
#
# # יצירת הקבצים
# dot.render('training_erd_png', format='png', cleanup=True)
# dot.render('training_erd_pdf', format='pdf', cleanup=True)
#
# print("✅ ERD saved as 'training_erd_png.png' and 'training_erd_pdf.pdf'")


from graphviz import Digraph

dot = Digraph(comment='ERD - Training System')
dot.attr(rankdir='LR', fontsize='20')

def add_table(name, fields):
    label = f"<<TABLE BORDER='1' CELLBORDER='1' CELLSPACING='0'>"
    label += f"<TR><TD COLSPAN='1' BGCOLOR='lightblue'><B>{name}</B></TD></TR>"
    for field in fields:
        label += f"<TR><TD ALIGN='LEFT'>{field}</TD></TR>"
    label += "</TABLE>>"
    dot.node(name, label=label, shape='plaintext')

# 🧍‍♂️ User
add_table('User', [
    'user_id: int (PK)',
    'email: string',
    'role: string',  # 'trainee' | 'trainer' | 'admin'
    'registration_date: datetime',
    'profile_data: json'  # includes name, dob, height, weight etc.
])

# 🏋️‍♀️ Training Sessions
add_table('Session', [
    'session_id: int (PK)',
    'user_id: int (FK)',
    'exercise_id: int (FK)',
    'start_time: datetime',
    'end_time: datetime',
    'duration_seconds: int',
    'exercise_counter: int',
    'feedback_count: int',
    'performance_score: float',
    'video_url: string'
])

# 📋 Exercise catalog
add_table('Exercise', [
    'exercise_id: int (PK)',
    'name: string',
    'description: string'
])

# 💬 Feedback by trainer or system
add_table('Feedback', [
    'feedback_id: int (PK)',
    'session_id: int (FK)',
    'trainer_id: int (FK, nullable)',
    'timestamp: datetime',
    'comments: text',
    'system_generated: boolean'
])

# 📈 Keypoints or performance snapshots
add_table('Keypoints', [
    'keypoint_id: int (PK)',
    'session_id: int (FK)',
    'timestamp: datetime',
    'rep_number: int',
    'keypoints_json: json',   # raw joint positions
    'angles_json: json',      # derived angles (elbow, back, etc.)
    'is_correct: boolean',
    'incorrect_duration: float'
])

# 🔗 Relationships
dot.edge('User', 'Session', label='1 ↔ N')
dot.edge('Session', 'Keypoints', label='1 ↔ N')
dot.edge('Session', 'Feedback', label='1 ↔ N')
dot.edge('Exercise', 'Session', label='1 ↔ N')

# 🖼️ Save as PNG and PDF
dot.render('training_erd_png', format='png', cleanup=True)
dot.render('training_erd_pdf', format='pdf', cleanup=True)

print("✅ ERD saved as 'training_erd_png.png' and 'training_erd_pdf.pdf'")



"""
📘 ERD: Training Platform – Schema Documentation & JSON Structure

This Python script generates a complete Entity-Relationship Diagram (ERD) for a fitness training platform,
visualized in both PNG and PDF formats. It reflects a normalized schema designed to support real-time
exercise feedback, session tracking, trainer involvement, and performance analysis.

──────────────────────────────────────────────
🧍‍♂️ User Table
──────────────────────────────────────────────
Represents all platform users, whether trainees, trainers, or admins.

Fields:
- user_id (PK): Unique ID per user
- email: Email used for login
- role: One of ['trainee', 'trainer', 'admin']
- registration_date: Account creation time
- profile_data (json): Dynamic personal information

Example profile_data:
{
  "first_name": "Lior",
  "last_name": "Ben-David",
  "birth_date": "1993-10-15",
  "height_cm": 178,
  "weight_kg": 72,
  "gender": "male"
}

──────────────────────────────────────────────
🏋️‍♀️ Session Table
──────────────────────────────────────────────
Logs each complete workout session performed by a user.

Fields:
- session_id (PK)
- user_id (FK → User)
- exercise_type: e.g. "overhead_press", "lunge"
- start_time / end_time: Timestamps for the session duration
- duration_seconds: Total time (derived from above)
- exercise_counter: Total reps done
- feedback_count: System/trainer feedbacks
- performance_score: Numeric score (e.g., 0.0 to 100.0)
- video_url: Location of recorded session (e.g., cloud bucket path)

──────────────────────────────────────────────
📋 Exercise Table
──────────────────────────────────────────────
Catalog of available exercises on the platform.

Fields:
- exercise_id (PK)
- name: e.g. "Overhead Press"
- description: Optional longer text

──────────────────────────────────────────────
💬 Feedback Table
──────────────────────────────────────────────
Stores feedback given by either the system (AI) or a human trainer.

Fields:
- feedback_id (PK)
- session_id (FK → Session)
- trainer_id (FK → User), nullable if system-generated
- timestamp: When the feedback was created
- comments: Written explanation of the feedback
- system_generated: True if AI-generated, False if from a human

Example:
{
  "comments": "Back arch detected – tighten your core.",
  "system_generated": true
}

──────────────────────────────────────────────
📈 Keypoints Table
──────────────────────────────────────────────
Stores detailed pose data and computed features for each rep or frame during a session.

Fields:
- keypoint_id: Unique identifier for the entry (Primary Key)
- session_id: Foreign Key linking to the Session
- timestamp: Time of capture in ISO 8601 format (e.g., "2025-04-08T15:42:00")
- rep_number: Repetition number this data corresponds to
- keypoints_json: Raw joint coordinates (e.g., from MediaPipe) for relevant body parts
- angles_json: Derived angles or posture metrics (e.g., elbow angle, back angle, etc.)
- is_correct: Whether the form was acceptable at this moment (True/False)
- incorrect_duration: Time (in seconds) form was incorrect; useful for flagging sustained mistakes


Example keypoints_json:
{
  "shoulder_l": [0.45, 0.31],
  "elbow_l": [0.47, 0.52],
  "wrist_l": [0.48, 0.74],
  "shoulder_r": [0.55, 0.30],
  "elbow_r": [0.56, 0.51],
  "wrist_r": [0.57, 0.72]
}

Example angles_json:
{
  "elbow_angle_l": 171.2,
  "elbow_angle_r": 169.8,
  "back_angle_l": 166.5,
  "back_angle_r": 170.3,
  "wrist_above_l": 1,
  "wrist_above_r": 1
}

Example complete row:
{
  "timestamp": "2025-04-08T15:42:00",
  "rep_number": 4,
  "keypoints_json": { ... },
  "angles_json": { ... },
  "is_correct": false,
  "incorrect_duration": 1.73
}

──────────────────────────────────────────────
🎯 Usage
──────────────────────────────────────────────
To generate your ERD, simply run:

    python generate_erd.py

This will output:
- training_erd_png.png
- training_erd_pdf.pdf

Each visual ERD contains color-coded tables, field types, and relationships between entities.
"""
