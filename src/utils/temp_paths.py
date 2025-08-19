"""
Cross-platform temporary path utilities for Right Motion application.
Provides consistent temporary file handling across Windows, macOS, and Linux.
"""

import os
import tempfile
import platform
from typing import Union


def get_temp_dir() -> str:
    """Get the appropriate temporary directory for the current platform."""
    system = platform.system().lower()
    
    if system == 'windows':
        # Windows: Use system temp directory
        return tempfile.gettempdir()
    else:
        # macOS/Linux: Use /tmp/ if it exists, otherwise fallback to system temp
        if os.path.exists('/tmp/') and os.access('/tmp/', os.W_OK):
            return '/tmp'
        else:
            return tempfile.gettempdir()


def get_temp_path(filename: str) -> str:
    """Get full path for a temporary file with cross-platform compatibility."""
    temp_dir = get_temp_dir()
    return os.path.join(temp_dir, filename)


def get_user_session_video_path(user_id: Union[int, str], exercise: str = None) -> str:
    """Get temporary video path for user session."""
    if exercise:
        filename = f"user{user_id}_{exercise}_session.mp4"
    else:
        filename = f"user{user_id}_session.mp4"
    return get_temp_path(filename)


def get_user_reps_path(user_id: Union[int, str]) -> str:
    """Get temporary reps file path for user."""
    filename = f"reps_{user_id}.txt"
    return get_temp_path(filename)


def get_user_session_details_path(user_id: Union[int, str]) -> str:
    """Get temporary session details file path for user."""
    filename = f"sessiondetails_{user_id}.json"
    return get_temp_path(filename)


def get_user_feedback_path(user_id: Union[int, str]) -> str:
    """Get temporary feedback file path for user."""
    filename = f"feedback_{user_id}.json"
    return get_temp_path(filename)


def get_audio_temp_path(filename: str) -> str:
    """Get temporary audio file path."""
    return get_temp_path(filename)


def cleanup_user_temp_files(user_id: Union[int, str]) -> None:
    """Clean up all temporary files for a specific user."""
    import glob
    
    temp_dir = get_temp_dir()
    patterns = [
        f"reps_{user_id}.txt",
        f"sessiondetails_{user_id}*.json",
        f"feedback_{user_id}*.json",
        f"user{user_id}_*.mp4",
    ]
    
    for pattern in patterns:
        pattern_path = os.path.join(temp_dir, pattern)
        for file_path in glob.glob(pattern_path):
            try:
                os.remove(file_path)
                print(f"🗑️ Cleaned up temp file: {file_path}")
            except OSError as e:
                print(f"⚠️ Failed to clean up {file_path}: {e}")


# Platform detection for conditional logic
IS_WINDOWS = platform.system().lower() == 'windows'
IS_MACOS = platform.system().lower() == 'darwin'
IS_LINUX = platform.system().lower() == 'linux'

# Export commonly used temp directory
TEMP_DIR = get_temp_dir()