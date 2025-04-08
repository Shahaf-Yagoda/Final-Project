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

# הגדרת הטבלאות
add_table('User', [
    'user_id: int (PK)',
    'name: string',
    'email: string',
    'date_of_birth: date',
    'registration_date: datetime',
    'role: string'
])
add_table('Trainer', ['user_id: int (PK, FK)', 'specialty: string'])
add_table('Trainee', ['user_id: int (PK, FK)', 'progress: json'])
add_table('Admin', ['user_id: int (PK, FK)', 'permissions: json'])

add_table('Session', [
    'session_id: int (PK)',
    'user_id: int (FK)',
    'start_time: datetime',
    'end_time: datetime',
    'exercise_type: string',
    'exercise_counter: int',
    'feedback_count: int',
    'performance_score: float',
    'duration_seconds: int'
])
add_table('Exercise', ['exercise_id: int (PK)', 'name: string'])
add_table('Feedback', [
    'feedback_id: int (PK)',
    'session_id: int (FK)',
    'trainer_id: int (FK)',
    'comments: text',
    'timestamp: datetime'
])

# קשרים
dot.edge('User', 'Trainer')
dot.edge('User', 'Trainee')
dot.edge('User', 'Admin')
dot.edge('User', 'Session', label='1 ↔ N')
dot.edge('Trainer', 'Feedback', label='1 ↔ N')
dot.edge('Session', 'Feedback', label='1 ↔ N')
dot.edge('Exercise', 'Session', label='1 ↔ N')

# יצירת הקבצים
dot.render('training_erd_png', format='png', cleanup=True)
dot.render('training_erd_pdf', format='pdf', cleanup=True)

print("✅ ERD saved as 'training_erd_png.png' and 'training_erd_pdf.pdf'")
