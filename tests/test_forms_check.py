"""Tests for the exercise form checking classes."""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import cv2
import numpy as np
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from processing.forms_check import (
    BaseExerciseChecker, LungeChecker, OverheadPressChecker, PlankChecker,
    get_exercise_checker, check_form, init_state
)


class TestBaseExerciseChecker(unittest.TestCase):
    """Test the abstract base class."""
    
    def test_cannot_instantiate_directly(self):
        """Test that BaseExerciseChecker cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseExerciseChecker()


class TestLungeChecker(unittest.TestCase):
    """Test suite for LungeChecker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.checker = LungeChecker()
        self.mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.state = self.checker.init_state()
    
    def test_initialization(self):
        """Test LungeChecker initialization."""
        state = self.checker.init_state()
        
        expected_keys = ["ready", "direction", "count", "last_message", 
                        "message_timer", "incorrect_start_time", 
                        "last_spoken_time", "last_spoken_msg"]
        
        for key in expected_keys:
            self.assertIn(key, state)
        
        self.assertFalse(state["ready"])
        self.assertEqual(state["count"], 0)
    
    def test_get_point(self):
        """Test the get_point utility method."""
        # Create mock landmarks
        mock_landmarks = [Mock() for _ in range(33)]
        mock_landmarks[23].x = 0.5  # LEFT_HIP
        mock_landmarks[23].y = 0.6
        
        with patch('processing.forms_check.mp_pose') as mock_mp_pose:
            mock_mp_pose.PoseLandmark.LEFT_HIP.value = 23
            
            point = self.checker.get_point(mock_landmarks, "HIP", "LEFT")
            self.assertEqual(point, [0.5, 0.6])
    
    def test_determine_front_leg(self):
        """Test front leg determination logic."""
        # Create mock landmarks
        mock_landmarks = [Mock() for _ in range(33)]
        mock_landmarks[25].z = 0.3  # LEFT_KNEE closer (smaller z)
        mock_landmarks[26].z = 0.7  # RIGHT_KNEE farther
        
        with patch('processing.forms_check.mp_pose') as mock_mp_pose:
            mock_mp_pose.PoseLandmark.LEFT_KNEE.value = 25
            mock_mp_pose.PoseLandmark.RIGHT_KNEE.value = 26
            
            front_leg = self.checker.determine_front_leg(mock_landmarks)
            self.assertEqual(front_leg, "left")
    
    @patch('processing.forms_check.calculate_angle')
    @patch('processing.forms_check.draw_joint_angle')
    @patch('cv2.putText')
    def test_check_form_basic(self, mock_puttext, mock_draw_angle, mock_calc_angle):
        """Test basic form checking functionality."""
        # Mock landmarks with required points
        mock_landmarks = [Mock() for _ in range(33)]
        for i, landmark in enumerate(mock_landmarks):
            landmark.x = 0.5
            landmark.y = 0.5
            landmark.z = 0.5 if i == 25 else 0.7  # Make left knee closer
        
        # Mock angle calculations
        mock_calc_angle.side_effect = [95, 95, 170, 25]  # Good form angles
        
        # Mock MediaPipe pose constants
        with patch('processing.forms_check.mp_pose') as mock_mp_pose:
            # Set up landmark indices
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
            
            feedback, reps = self.checker.check_form(self.mock_image, mock_landmarks, self.state)
            
            self.assertIsInstance(feedback, list)
            self.assertIsInstance(reps, int)
    
    @patch('processing.forms_check.speak_async')
    def test_provide_audio_feedback(self, mock_speak):
        """Test audio feedback functionality."""
        feedback = ["Test feedback message"]
        current_time = time.time()
        
        # Test initial feedback
        self.checker.provide_audio_feedback(feedback, self.state, current_time)
        
        # Should not speak immediately (delay not met)
        mock_speak.assert_not_called()
        
        # Test after delay
        self.state["incorrect_start_time"] = current_time - 2.0  # 2 seconds ago
        self.checker.provide_audio_feedback(feedback, self.state, current_time)
        
        # Should speak now
        mock_speak.assert_called_once_with("Test feedback message")


class TestOverheadPressChecker(unittest.TestCase):
    """Test suite for OverheadPressChecker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.checker = OverheadPressChecker()
        self.mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.state = self.checker.init_state()
    
    def test_initialization(self):
        """Test OverheadPressChecker initialization."""
        state = self.checker.init_state()
        
        expected_keys = ["ready", "exercise_started", "rep_phase", "count", 
                        "auto_end", "ready_start_time", "missing_start_time"]
        
        for key in expected_keys:
            self.assertIn(key, state)
        
        self.assertFalse(state["ready"])
        self.assertFalse(state["exercise_started"])
        self.assertEqual(state["rep_phase"], "bottom")
    
    def test_all_keypoints_visible(self):
        """Test keypoint visibility checking."""
        # Create mock landmarks with high visibility
        mock_landmarks = [Mock() for _ in range(33)]
        for landmark in mock_landmarks:
            landmark.visibility = 0.8
        
        result = self.checker.all_keypoints_visible(mock_landmarks)
        self.assertTrue(result)
        
        # Test with low visibility
        mock_landmarks[11].visibility = 0.3  # LEFT_SHOULDER
        result = self.checker.all_keypoints_visible(mock_landmarks)
        self.assertFalse(result)
    
    def test_is_wrist_above_shoulder(self):
        """Test wrist position checking."""
        wrist = [0.5, 0.3]  # y=0.3
        shoulder = [0.5, 0.5]  # y=0.5
        
        # Wrist above shoulder (smaller y coordinate)
        result = self.checker.is_wrist_above_shoulder(wrist, shoulder)
        self.assertTrue(result)
        
        # Wrist below shoulder
        wrist[1] = 0.7
        result = self.checker.is_wrist_above_shoulder(wrist, shoulder)
        self.assertFalse(result)


class TestPlankChecker(unittest.TestCase):
    """Test suite for PlankChecker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.checker = PlankChecker()
        self.mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.state = self.checker.init_state()
    
    def test_initialization(self):
        """Test PlankChecker initialization."""
        state = self.checker.init_state()
        
        expected_keys = ["ready", "count", "plank_duration_sec", "plank_start_time"]
        
        for key in expected_keys:
            self.assertIn(key, state)
        
        self.assertFalse(state["ready"])
        self.assertEqual(state["plank_duration_sec"], 30)
        self.assertIsNone(state["plank_start_time"])


class TestFactoryAndCompatibility(unittest.TestCase):
    """Test factory functions and backward compatibility."""
    
    def test_get_exercise_checker_valid(self):
        """Test getting valid exercise checkers."""
        lunge_checker = get_exercise_checker("lunge")
        self.assertIsInstance(lunge_checker, LungeChecker)
        
        press_checker = get_exercise_checker("press")
        self.assertIsInstance(press_checker, OverheadPressChecker)
        
        plank_checker = get_exercise_checker("plank")
        self.assertIsInstance(plank_checker, PlankChecker)
    
    def test_get_exercise_checker_invalid(self):
        """Test getting invalid exercise checker."""
        with self.assertRaises(ValueError):
            get_exercise_checker("invalid_exercise")
    
    @patch('processing.forms_check.get_exercise_checker')
    def test_check_form_compatibility(self, mock_get_checker):
        """Test backward compatibility check_form function."""
        # Mock checker
        mock_checker = Mock()
        mock_checker.check_form.return_value = (["feedback"], 2)
        mock_get_checker.return_value = mock_checker
        
        # Test the compatibility function
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_landmarks = [Mock() for _ in range(33)]
        mock_state = {"count": 0}
        
        feedback, reps = check_form("lunge", mock_image, mock_landmarks, mock_state)
        
        # Verify calls
        mock_get_checker.assert_called_once_with("lunge")
        mock_checker.check_form.assert_called_once_with(mock_image, mock_landmarks, mock_state)
        self.assertEqual(feedback, ["feedback"])
        self.assertEqual(reps, 2)
    
    @patch('processing.forms_check.get_exercise_checker')
    def test_init_state_compatibility(self, mock_get_checker):
        """Test backward compatibility init_state function."""
        # Mock checker
        mock_checker = Mock()
        mock_checker.init_state.return_value = {"count": 0, "ready": False}
        mock_get_checker.return_value = mock_checker
        
        # Test the compatibility function
        state = init_state("press")
        
        # Verify calls
        mock_get_checker.assert_called_once_with("press")
        mock_checker.init_state.assert_called_once()
        self.assertEqual(state, {"count": 0, "ready": False})


if __name__ == '__main__':
    unittest.main()