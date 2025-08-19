"""Tests for the VideoStreamManager class in video_streamer.py"""
import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys
import os
import json
import numpy as np
import cv2

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'app'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from video_streamer import VideoStreamManager
from src.utils.temp_paths import get_user_reps_path, get_user_session_video_path


class TestVideoStreamManager(unittest.TestCase):
    """Test suite for VideoStreamManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = VideoStreamManager()
    
    def test_initialization(self):
        """Test VideoStreamManager initialization."""
        self.assertIsNotNone(self.manager.pose_detector)
        self.assertIsNotNone(self.manager.mp_pose)
        self.assertIsNotNone(self.manager.mp_drawing)
        self.assertEqual(len(self.manager.session_states), 0)
        self.assertEqual(len(self.manager.video_writers), 0)
        self.assertEqual(len(self.manager.video_temp_paths), 0)
    
    def test_get_session_key(self):
        """Test session key generation."""
        key = self.manager.get_session_key(123, "lunge")
        self.assertEqual(key, "123_lunge")
    
    @patch('video_streamer.init_state')
    def test_get_or_create_session_state(self, mock_init_state):
        """Test session state creation and retrieval."""
        # Mock init_state return value
        mock_state = {"count": 0, "ready": False}
        mock_init_state.return_value = mock_state
        
        # Test creating new state
        state = self.manager.get_or_create_session_state(1, "lunge")
        mock_init_state.assert_called_once_with("lunge")
        self.assertEqual(state, mock_state)
        
        # Test retrieving existing state
        mock_init_state.reset_mock()
        state2 = self.manager.get_or_create_session_state(1, "lunge")
        mock_init_state.assert_not_called()  # Should not create new state
        self.assertEqual(state2, mock_state)
    
    @patch('video_streamer.init_state')
    def test_set_session_active(self, mock_init_state):
        """Test setting session active state."""
        mock_init_state.return_value = {"count": 0}
        
        # Test setting active
        self.manager.set_session_active(1, "press", True)
        key = self.manager.get_session_key(1, "press")
        self.assertTrue(self.manager.session_states[key]["session_active"])
        
        # Test setting inactive
        self.manager.set_session_active(1, "press", False)
        self.assertFalse(self.manager.session_states[key]["session_active"])
    
    @patch('builtins.open', new_callable=mock_open)
    def test_save_reps_to_tempfile(self, mock_file):
        """Test saving reps to temporary file."""
        self.manager.save_reps_to_tempfile(123, 5)
        
        mock_file.assert_called_once_with(get_user_reps_path(123), "w")
        mock_file().write.assert_called_once_with("5")
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('json.load')
    @patch('json.dump')
    def test_append_session_detail(self, mock_json_dump, mock_json_load, mock_exists, mock_file):
        """Test appending session details."""
        # Test with existing file
        mock_exists.return_value = True
        mock_json_load.return_value = [{"existing": "detail"}]
        
        detail = {"timestamp": 123456, "rep_num": 1}
        self.manager.append_session_detail(123, detail)
        
        # Verify file operations
        mock_file.assert_called()
        mock_json_load.assert_called_once()
        mock_json_dump.assert_called_once()
        
        # Verify detail was appended
        dumped_data = mock_json_dump.call_args[0][0]
        self.assertEqual(len(dumped_data), 2)
        self.assertEqual(dumped_data[0], {"existing": "detail"})
        self.assertEqual(dumped_data[1], detail)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('json.load')
    @patch('json.dump')
    def test_append_feedback(self, mock_json_dump, mock_json_load, mock_exists, mock_file):
        """Test appending feedback."""
        # Test with new file
        mock_exists.return_value = False
        
        feedback = {"timestamp": 123456, "message": "Good form!"}
        self.manager.append_feedback(123, feedback)
        
        # Verify feedback was saved
        dumped_data = mock_json_dump.call_args[0][0]
        self.assertEqual(len(dumped_data), 1)
        self.assertEqual(dumped_data[0], feedback)
    
    @patch('video_streamer.get_exercise_checker')
    @patch('cv2.cvtColor')
    def test_process_frame(self, mock_cvtcolor, mock_get_checker):
        """Test frame processing."""
        # Mock inputs
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cvtcolor.return_value = mock_frame
        
        # Mock pose detection with properly configured landmarks
        mock_landmarks = []
        for i in range(33):
            landmark = Mock()
            landmark.x = 0.5
            landmark.y = 0.5
            landmark.z = 0.5
            landmark.visibility = 0.8  # High visibility
            mock_landmarks.append(landmark)
        
        mock_results = Mock()
        mock_results.pose_landmarks.landmark = mock_landmarks
        self.manager.pose_detector.process = Mock(return_value=mock_results)
        
        # Mock exercise checker
        mock_checker = Mock()
        mock_checker.check_form.return_value = (["feedback"], 1)
        mock_get_checker.return_value = mock_checker
        
        # Mock session state
        with patch.object(self.manager, 'get_or_create_session_state') as mock_get_state:
            mock_state = {"count": 0}
            mock_get_state.return_value = mock_state
            
            # Mock file operations and drawing functions
            with patch.object(self.manager, 'save_reps_to_tempfile') as mock_save_reps, \
                 patch.object(self.manager, 'append_session_detail') as mock_append_detail, \
                 patch.object(self.manager, 'append_feedback') as mock_append_feedback, \
                 patch.object(self.manager.mp_drawing, 'draw_landmarks') as mock_draw:
                
                result_frame = self.manager.process_frame(mock_frame, "lunge", 123)
                
                # Verify calls
                mock_get_checker.assert_called_once_with("lunge")
                mock_checker.check_form.assert_called_once()
                mock_save_reps.assert_called_once_with(123, 1)
                mock_append_detail.assert_called_once()
                mock_append_feedback.assert_called_once()
                mock_draw.assert_called_once()
                
                self.assertIsNotNone(result_frame)
    
    @patch('cv2.VideoWriter')
    def test_setup_video_recording(self, mock_video_writer):
        """Test video recording setup."""
        # Mock video capture
        mock_cap = Mock()
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0
        }.get(prop, 0)
        
        # Mock video writer
        mock_writer = Mock()
        mock_video_writer.return_value = mock_writer
        
        result = self.manager.setup_video_recording(123, "lunge", mock_cap)
        
        # Verify video writer creation
        mock_video_writer.assert_called_once()
        self.assertEqual(result, mock_writer)
        
        # Verify temp path storage
        self.assertIn((123, "lunge"), self.manager.video_temp_paths)
        self.assertIn((123, "lunge"), self.manager.video_writers)
    
    @patch('cv2.VideoCapture')
    @patch('time.sleep')
    def test_generate_stream(self, mock_sleep, mock_video_capture):
        """Test video stream generation."""
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, np.zeros((480, 640, 3), dtype=np.uint8)),
                                    (False, None)]  # First frame then end
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0
        }.get(prop, 0)
        mock_video_capture.return_value = mock_cap
        
        # Mock video writer
        with patch('cv2.VideoWriter') as mock_video_writer:
            mock_writer = Mock()
            mock_video_writer.return_value = mock_writer
            
            # Mock frame processing
            with patch.object(self.manager, 'process_frame') as mock_process:
                mock_process.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Mock image encoding
                with patch('cv2.imencode') as mock_imencode:
                    mock_imencode.return_value = (True, np.array([1, 2, 3]))
                    
                    # Generate stream (get first frame)
                    stream_gen = self.manager.generate_stream("lunge", 123)
                    first_frame = next(stream_gen)
                    
                    # Verify frame format
                    self.assertIn(b'--frame', first_frame)
                    self.assertIn(b'Content-Type: image/jpeg', first_frame)
    
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('shutil.move')
    @patch('time.strftime')
    def test_stop_session(self, mock_strftime, mock_move, mock_makedirs, mock_exists):
        """Test session stopping and video file handling."""
        # Set up temp video path
        self.manager.video_temp_paths[(123, "lunge")] = get_user_session_video_path(123)
        
        # Mock file operations
        mock_exists.return_value = True
        mock_strftime.return_value = "20240101_120000"
        
        # Mock path operations
        with patch('os.path.abspath') as mock_abspath, \
             patch('os.path.join') as mock_join:
            mock_abspath.return_value = "/project/videos"
            mock_join.return_value = "/project/videos/user123_session_20240101_120000.mp4"
            
            result = self.manager.stop_session(123, "lunge")
            
            # Verify operations
            mock_makedirs.assert_called_once_with("/project/videos", exist_ok=True)
            mock_move.assert_called_once_with(
                get_user_session_video_path(123), 
                "/project/videos/user123_session_20240101_120000.mp4"
            )
            self.assertEqual(result, "/project/videos/user123_session_20240101_120000.mp4")
    
    def test_stop_session_no_video(self):
        """Test stopping session when no video file exists."""
        result = self.manager.stop_session(123, "lunge")
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()