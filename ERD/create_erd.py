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


# 🧍‍♂️ User table
add_table('User', [
    'user_id: int (PK)',
    'email: string',
    'password: string',
    'registration_date: datetime',
    'profile_data: json',
    'role: json'
])

# 🏋️‍♀️ Training Session
add_table('Session', [
    'session_id: int (PK)',
    'user_id: int (FK)',
    'exercise_id: int (FK)',
    'start_time: datetime',
    'end_time: datetime',
    'duration_sec: int',
    'reps_count: int',
    'feedback_count: int',
    'performance_score: float',
    'video_path: string'
])

# 📋 Exercise catalog
add_table('Exercise', [
    'exercise_id: int (PK)',
    'name: string',
    'description: string'
])

# 📈 SessionDetails
add_table('SessionDetails', [
    'detail_id: int (PK)',
    'session_id: int (FK)',
    'timestamp: datetime',
    'rep_num: int',
    'keypoints_json: json',
    'features_json: json',
    'is_correct: boolean',
    'incorrect_duration: float'
])

# 🤖 System-generated feedback
add_table('SystemFeedback', [
    'feedback_id: int (PK)',
    'session_id: int (FK)',
    'timestamp: datetime',
    'message: text'
])

# 💬 Human-written comments
add_table('Comment', [
    'comment_id: int (PK)',
    'session_id: int (FK)',
    'user_id: int (FK)',
    'timestamp: datetime',
    'comment: text'
])

# 🔗 Relationships
dot.edge('User', 'Session', label='1 ↔ N')
dot.edge('Session', 'SessionDetails', label='1 ↔ N')
dot.edge('Session', 'SystemFeedback', label='1 ↔ N')
dot.edge('Session', 'Comment', label='1 ↔ N')
dot.edge('Exercise', 'Session', label='1 ↔ N')

# Save the diagram
dot.render('training_erd_png', format='png', cleanup=True)
dot.render('training_erd_pdf', format='pdf', cleanup=True)

print("✅ Updated ERD saved as 'training_erd_png.png' and 'training_erd_pdf.pdf'")