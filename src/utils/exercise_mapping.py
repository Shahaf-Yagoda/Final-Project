"""
Exercise name mapping utilities for Right Motion application.
Handles the mapping between UI exercise names and database exercise names.
"""

# Mapping from UI names (used in frontend/forms_check) to database names
UI_TO_DB_MAPPING = {
    "press": "overhead_press",
    "lunge": "lunge", 
    "plank": "plank"
}

# Reverse mapping from database names to UI names
DB_TO_UI_MAPPING = {v: k for k, v in UI_TO_DB_MAPPING.items()}

def ui_to_db_name(exercise_name: str) -> str:
    """Convert UI exercise name to database exercise name."""
    return UI_TO_DB_MAPPING.get(exercise_name, exercise_name)

def db_to_ui_name(exercise_name: str) -> str:
    """Convert database exercise name to UI exercise name."""
    return DB_TO_UI_MAPPING.get(exercise_name, exercise_name)

def get_all_ui_exercises() -> list:
    """Get list of all UI exercise names."""
    return list(UI_TO_DB_MAPPING.keys())

def get_all_db_exercises() -> list:
    """Get list of all database exercise names.""" 
    return list(UI_TO_DB_MAPPING.values())

def is_valid_ui_exercise(exercise_name: str) -> bool:
    """Check if exercise name is valid UI exercise."""
    return exercise_name in UI_TO_DB_MAPPING

def is_valid_db_exercise(exercise_name: str) -> bool:
    """Check if exercise name is valid database exercise."""
    return exercise_name in DB_TO_UI_MAPPING