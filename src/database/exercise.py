from datetime import datetime
from src.database.database_connection import get_connection
import psycopg2
import json
from typing import Optional, List, Dict, Any

class Exercise:
    def __init__(self, exercise_id=None, name=None, description=None, 
                 target_muscles=None, instructions=None, created_at=None, updated_at=None):
        self.exercise_id = exercise_id
        self.name = name
        self.description = description
        self.target_muscles = self._safe_json_load(target_muscles)
        self.instructions = instructions
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()

    @staticmethod
    def _safe_json_load(val):
        """Safely load JSON data"""
        if val is None or val == '' or val == 'null':
            return None
        if isinstance(val, list):
            return val
        try:
            return json.loads(val) if isinstance(val, str) else val
        except Exception:
            return val

    @classmethod
    def create(cls, name: str, description: str = None, 
               target_muscles: List[str] = None, instructions: str = None) -> 'Exercise':
        """Create a new exercise"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                now = datetime.now()
                cur.execute("""
                    INSERT INTO exercises (name, description, target_muscles, instructions, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s) RETURNING exercise_id
                """, (name, description, json.dumps(target_muscles) if target_muscles else None, 
                      instructions, now, now))
                exercise_id = cur.fetchone()[0]
                conn.commit()
                
                return cls(
                    exercise_id=exercise_id, name=name, description=description,
                    target_muscles=target_muscles, instructions=instructions,
                    created_at=now, updated_at=now
                )
        except psycopg2.Error as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    @classmethod
    def get_by_id(cls, exercise_id: int) -> Optional['Exercise']:
        """Get exercise by ID"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT exercise_id, name, description, target_muscles, instructions, created_at, updated_at
                    FROM exercises WHERE exercise_id = %s
                """, (exercise_id,))
                row = cur.fetchone()
                if row:
                    return cls(*row)
                return None
        finally:
            conn.close()

    @classmethod
    def get_by_name(cls, name: str) -> Optional['Exercise']:
        """Get exercise by name"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT exercise_id, name, description, target_muscles, instructions, created_at, updated_at
                    FROM exercises WHERE name = %s
                """, (name,))
                row = cur.fetchone()
                if row:
                    return cls(*row)
                return None
        finally:
            conn.close()

    @classmethod
    def get_all(cls) -> List['Exercise']:
        """Get all exercises"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT exercise_id, name, description, target_muscles, instructions, created_at, updated_at
                    FROM exercises ORDER BY name
                """)
                rows = cur.fetchall()
                return [cls(*row) for row in rows]
        finally:
            conn.close()

    @classmethod
    def search_by_muscle_group(cls, muscle_group: str) -> List['Exercise']:
        """Search exercises by target muscle group"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT exercise_id, name, description, target_muscles, instructions, created_at, updated_at
                    FROM exercises 
                    WHERE target_muscles::text ILIKE %s
                    ORDER BY name
                """, (f'%{muscle_group}%',))
                rows = cur.fetchall()
                return [cls(*row) for row in rows]
        finally:
            conn.close()

    def update(self, description: str = None, target_muscles: List[str] = None, 
               instructions: str = None) -> bool:
        """Update exercise information"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                # Build dynamic update query
                updates = []
                params = []
                
                if description is not None:
                    updates.append("description = %s")
                    params.append(description)
                    self.description = description
                
                if target_muscles is not None:
                    updates.append("target_muscles = %s")
                    params.append(json.dumps(target_muscles))
                    self.target_muscles = target_muscles
                
                if instructions is not None:
                    updates.append("instructions = %s")
                    params.append(instructions)
                    self.instructions = instructions
                
                if updates:
                    updates.append("updated_at = %s")
                    params.append(datetime.now())
                    params.append(self.exercise_id)
                    
                    query = f"UPDATE exercises SET {', '.join(updates)} WHERE exercise_id = %s"
                    cur.execute(query, params)
                    conn.commit()
                    self.updated_at = datetime.now()
                
                return True
        except psycopg2.Error:
            conn.rollback()
            return False
        finally:
            conn.close()

    def to_dict(self) -> Dict[str, Any]:
        """Convert exercise to dictionary"""
        return {
            'exercise_id': self.exercise_id,
            'name': self.name,
            'description': self.description,
            'target_muscles': self.target_muscles,
            'instructions': self.instructions,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'updated_at': self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at
        }