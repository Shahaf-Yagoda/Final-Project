"""Tests for the ExerciseSession class in main.py"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import cv2
import numpy as np
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import ExerciseSession


class TestExerciseSession(unittest.TestCase):
    """Test suite for ExerciseSession class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.session = ExerciseSession("lunge", 1, show_window=False, save_keypoints=False)
    
    def test_initialization(self):
        """Test ExerciseSession initialization."""
        self.assertEqual(self.session.exercise_name, "lunge")
        self.assertEqual(self.session.user_id, 1)
        self.assertFalse(self.session.show_window)
        self.assertFalse(self.session.save_keypoints)
        self.assertIn("exercise", self.session.state)
        self.assertIn("lunge", self.session.state["exercise"])
        self.assertIsNone(self.session.start_time)
        self.assertIsNone(self.session.end_time)
    
    def test_initialization_with_different_exercises(self):
        """Test initialization with different exercise types."""
        exercises = ["lunge", "press", "plank"]
        for exercise in exercises:
            session = ExerciseSession(exercise, 2, show_window=False)
            self.assertEqual(session.exercise_name, exercise)
            self.assertIn(exercise, session.state["exercise"])
    
    @patch('main.cv2.VideoCapture')
    @patch('main.mp_pose.Pose')
    def test_process_frame(self, mock_pose, mock_video_capture):
        """Test frame processing logic."""
        # Mock frame and landmarks
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_landmarks = [Mock() for _ in range(33)]  # 33 pose landmarks
        
        # Mock pose detection results
        mock_results = Mock()
        mock_results.pose_landmarks.landmark = mock_landmarks
        
        # Test frame processing
        with patch('main.check_form') as mock_check_form:
            mock_check_form.return_value = ([], 1)  # No feedback, 1 rep
            
            feedback, reps = self.session.process_frame(mock_frame, mock_landmarks)
            
            # Verify check_form was called correctly
            mock_check_form.assert_called_once()
            self.assertEqual(feedback, [])
            self.assertEqual(reps, 1)
    
    def test_save_keypoints_to_db(self):
        """Test keypoints saving to database."""
        # This is a placeholder method, so we just test it doesn't crash
        keypoints_data = [{"x": 0.5, "y": 0.5, "z": 0.5, "visibility": 0.9}]
        try:
            self.session.save_keypoints_to_db(keypoints_data, 1)
        except Exception as e:
            self.fail(f"save_keypoints_to_db raised {e} unexpectedly!")
    
    @patch('main.cv2.VideoCapture')
    @patch('main.cv2.imshow')
    @patch('main.cv2.waitKey')
    @patch('main.cv2.destroyAllWindows')
    def test_run_session_no_window(self, mock_destroy, mock_waitkey, mock_imshow, mock_video_capture):
        """Test running session without display window."""
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(False, None)]  # Simulate immediate end
        mock_video_capture.return_value = mock_cap
        
        # Mock pose detector
        with patch.object(self.session, 'pose_detector') as mock_pose:
            mock_pose.__enter__ = Mock(return_value=mock_pose)
            mock_pose.__exit__ = Mock(return_value=None)
            
            result = self.session.run()
            
            # Verify result structure
            self.assertIn("user_id", result)
            self.assertIn("exercise", result)
            self.assertIn("start_time", result)
            self.assertIn("end_time", result)
            self.assertIn("reps", result)
            
            # Verify no window functions called
            mock_imshow.assert_not_called()
            mock_waitkey.assert_not_called()
            mock_destroy.assert_not_called()
    
    @patch('main.cv2.VideoCapture')
    @patch('main.cv2.imshow')
    @patch('main.cv2.waitKey')
    @patch('main.cv2.destroyAllWindows')
    def test_run_session_with_window(self, mock_destroy, mock_waitkey, mock_imshow, mock_video_capture):
        """Test running session with display window."""
        # Create session with window enabled
        session = ExerciseSession("lunge", 1, show_window=True, save_keypoints=False)
        
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(False, None)]  # Simulate immediate end
        mock_video_capture.return_value = mock_cap
        
        # Mock window interactions
        mock_waitkey.return_value = 27  # ESC key
        
        # Mock pose detector
        with patch.object(session, 'pose_detector') as mock_pose:
            mock_pose.__enter__ = Mock(return_value=mock_pose)
            mock_pose.__exit__ = Mock(return_value=None)
            
            result = session.run()
            
            # Verify window cleanup was called
            mock_destroy.assert_called_once()


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility functions."""
    
    @patch('main.ExerciseSession')
    def test_run_session_function(self, mock_exercise_session):
        """Test the backward compatibility run_session function."""
        from main import run_session
        
        # Mock the session and its run method
        mock_session_instance = Mock()
        mock_session_instance.run.return_value = {"reps": 5}
        mock_exercise_session.return_value = mock_session_instance
        
        # Call the function
        result = run_session("lunge", 1, show_window=False, save_keypoints=True)
        
        # Verify ExerciseSession was created correctly
        mock_exercise_session.assert_called_once_with("lunge", 1, False, True)
        mock_session_instance.run.assert_called_once()
        self.assertEqual(result, {"reps": 5})


if __name__ == '__main__':
    unittest.main()