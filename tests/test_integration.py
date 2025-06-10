"""Integration tests for the refactored components."""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import numpy as np

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import ExerciseSession
from src.processing.forms_check import get_exercise_checker
from src.app.video_streamer import VideoStreamManager


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.exercise_session = ExerciseSession("lunge", 1, show_window=False)
        self.video_manager = VideoStreamManager()
    
    def test_exercise_session_with_real_checker(self):
        """Test ExerciseSession with actual exercise checker."""
        # Get a real exercise checker
        checker = get_exercise_checker("lunge")
        
        # Verify it's the correct type
        from src.processing.forms_check import LungeChecker
        self.assertIsInstance(checker, LungeChecker)
        
        # Test that it has the expected methods
        self.assertTrue(hasattr(checker, 'check_form'))
        self.assertTrue(hasattr(checker, 'init_state'))
        self.assertTrue(callable(checker.check_form))
        self.assertTrue(callable(checker.init_state))
    
    @patch('main.cv2.VideoCapture')
    def test_exercise_session_full_flow(self, mock_video_capture):
        """Test the complete ExerciseSession flow."""
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        # Simulate reading one frame then ending
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, mock_frame), (False, None)]
        mock_video_capture.return_value = mock_cap
        
        # Mock pose detection
        with patch.object(self.exercise_session, 'pose_detector') as mock_pose:
            mock_pose.__enter__ = Mock(return_value=mock_pose)
            mock_pose.__exit__ = Mock(return_value=None)
            
            # Mock pose results (no landmarks detected)
            mock_results = Mock()
            mock_results.pose_landmarks = None
            mock_pose.process.return_value = mock_results
            
            # Run the session
            result = self.exercise_session.run()
            
            # Verify result structure
            self.assertIn("user_id", result)
            self.assertIn("exercise", result)
            self.assertIn("start_time", result)
            self.assertIn("end_time", result)
            self.assertIn("reps", result)
            
            # Verify values
            self.assertEqual(result["user_id"], 1)
            self.assertEqual(result["exercise"], "lunge")
            self.assertEqual(result["reps"], 0)  # No landmarks = no reps
    
    def test_video_manager_with_real_checker(self):
        """Test VideoStreamManager with actual exercise checker."""
        # Create mock frame and landmarks
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_landmarks = []
        
        # Mock pose detection to return no landmarks
        self.video_manager.pose_detector.process = Mock()
        mock_results = Mock()
        mock_results.pose_landmarks = None
        self.video_manager.pose_detector.process.return_value = mock_results
        
        # Process frame
        result_frame = self.video_manager.process_frame(mock_frame, "lunge", 1)
        
        # Should return the frame unchanged (no landmarks detected)
        self.assertIsNotNone(result_frame)
        self.assertEqual(result_frame.shape, mock_frame.shape)
    
    def test_exercise_checkers_consistency(self):
        """Test that all exercise checkers have consistent interfaces."""
        exercises = ["lunge", "press", "plank"]
        
        for exercise in exercises:
            checker = get_exercise_checker(exercise)
            
            # Test init_state
            state = checker.init_state()
            self.assertIsInstance(state, dict)
            self.assertIn("count", state)
            self.assertIn("ready", state)
            
            # Test check_form method exists and is callable
            self.assertTrue(hasattr(checker, 'check_form'))
            self.assertTrue(callable(checker.check_form))
    
    @patch('src.processing.forms_check.calculate_angle')
    @patch('src.processing.forms_check.draw_joint_angle')
    @patch('cv2.putText')
    def test_lunge_checker_integration(self, mock_puttext, mock_draw_angle, mock_calc_angle):
        """Integration test for LungeChecker with mock data."""
        checker = get_exercise_checker("lunge")
        
        # Create mock landmarks
        mock_landmarks = [Mock() for _ in range(33)]
        for i, landmark in enumerate(mock_landmarks):
            landmark.x = 0.5
            landmark.y = 0.5
            landmark.z = 0.5 if i == 25 else 0.7  # Left knee closer
        
        # Mock angle calculations for good form
        mock_calc_angle.side_effect = [95, 95, 170, 25]  # Good angles
        
        # Mock image
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        state = checker.init_state()
        
        # Mock MediaPipe constants
        with patch('src.processing.forms_check.mp_pose') as mock_mp_pose:
            # Set up all required landmark indices
            mock_mp_pose.PoseLandmark.LEFT_KNEE.value = 25
            mock_mp_pose.PoseLandmark.RIGHT_KNEE.value = 26
            mock_mp_pose.PoseLandmark.LEFT_HIP.value = 23
            mock_mp_pose.PoseLandmark.RIGHT_HIP.value = 24
            mock_mp_pose.PoseLandmark.LEFT_ANKLE.value = 27
            mock_mp_pose.PoseLandmark.RIGHT_ANKLE.value = 28
            mock_mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value = 31
            mock_mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value = 32
            mock_mp_pose.PoseLandmark.LEFT_SHOULDER.value = 11
            mock_mp_pose.PoseLandmark.RIGHT_SHOULDER.value = 12
            
            # Mock audio feedback to avoid actual TTS calls
            with patch('src.processing.forms_check.speak_async'):
                feedback, reps = checker.check_form(mock_image, mock_landmarks, state)
                
                # Should have no feedback for good form
                self.assertEqual(len(feedback), 0)
                self.assertIsInstance(reps, int)
    
    def test_backward_compatibility(self):
        """Test that old function interfaces still work."""
        # Test that the old function imports work
        from src.processing.forms_check import check_form, init_state
        
        # Mock data
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_landmarks = [Mock() for _ in range(33)]
        
        # Test init_state function
        state = init_state("lunge")
        self.assertIsInstance(state, dict)
        
        # Test check_form function with mock data
        with patch('src.processing.forms_check.get_exercise_checker') as mock_get_checker:
            mock_checker = Mock()
            mock_checker.check_form.return_value = ([], 0)
            mock_get_checker.return_value = mock_checker
            
            feedback, reps = check_form("lunge", mock_image, mock_landmarks, state)
            
            self.assertEqual(feedback, [])
            self.assertEqual(reps, 0)
    
    def test_exercise_session_backward_compatibility(self):
        """Test that the old run_session function still works."""
        from main import run_session
        
        with patch('main.ExerciseSession') as mock_session_class:
            mock_session = Mock()
            mock_session.run.return_value = {"reps": 5}
            mock_session_class.return_value = mock_session
            
            result = run_session("lunge", 1)
            
            mock_session_class.assert_called_once_with("lunge", 1, True, False)
            self.assertEqual(result, {"reps": 5})


class TestErrorHandling(unittest.TestCase):
    """Test error handling in the refactored system."""
    
    def test_invalid_exercise_type(self):
        """Test handling of invalid exercise types."""
        with self.assertRaises(ValueError):
            get_exercise_checker("invalid_exercise")
    
    def test_exercise_session_invalid_exercise(self):
        """Test ExerciseSession with invalid exercise."""
        # This should raise an error when trying to initialize state
        with self.assertRaises(ValueError):
            ExerciseSession("invalid_exercise", 1)
    
    @patch('cv2.VideoCapture')
    def test_video_manager_camera_unavailable(self, mock_video_capture):
        """Test VideoStreamManager when camera is unavailable."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        manager = VideoStreamManager()
        
        with self.assertRaises(RuntimeError):
            list(manager.generate_stream("lunge", 1))  # Convert generator to list to trigger execution


if __name__ == '__main__':
    unittest.main()