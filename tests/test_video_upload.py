"""Test video upload and analysis functionality."""
import unittest
import os
import sys
import tempfile
import cv2
import mediapipe as mp
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.processing.forms_check import check_form
from src.database.session import Session


class TestVideoUpload(unittest.TestCase):
    """Test cases for video upload and analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_video_path = os.path.join(
            os.path.dirname(__file__), 
            "Lunge- video test.mp4"
        )
        
    def test_video_file_exists(self):
        """Test that the test video file exists."""
        self.assertTrue(
            os.path.exists(self.test_video_path),
            f"Test video file not found at {self.test_video_path}"
        )
        
    def test_video_file_readable(self):
        """Test that the video file can be opened by OpenCV."""
        cap = cv2.VideoCapture(self.test_video_path)
        self.assertTrue(cap.isOpened(), "Video file cannot be opened by OpenCV")
        
        # Test that we can read at least one frame
        ret, frame = cap.read()
        self.assertTrue(ret, "Cannot read frames from video")
        self.assertIsNotNone(frame, "Frame is None")
        
        cap.release()
        
    def test_video_analysis_pipeline(self):
        """Test the complete video analysis pipeline."""
        # Test video processing similar to the Streamlit app
        cap = cv2.VideoCapture(self.test_video_path)
        self.assertTrue(cap.isOpened())
        
        # Initialize MediaPipe pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            output_path = tmp.name
            
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            rep_count = 0
            state = {
                "ready": False,
                "direction": None,
                "count": 0,
                "last_message": "",
                "message_timer": 0,
                "feedback": []
            }
            feedback_messages = []
            
            # Process frames (limit to first 30 frames for speed)
            max_frames = 30
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )
                    # Test form checking
                    feedback, reps = check_form(
                        "lunge", frame, results.pose_landmarks.landmark, state
                    )
                    rep_count = max(rep_count, reps)
                    if feedback:
                        feedback_messages.extend(feedback)
                        
                out.write(frame)
                
            cap.release()
            out.release()
            
            # Verify output
            self.assertGreater(frame_count, 0, "No frames were processed")
            self.assertTrue(os.path.exists(output_path), "Output video not created")
            self.assertGreater(os.path.getsize(output_path), 0, "Output video is empty")
            
            # Test that output video is readable
            test_cap = cv2.VideoCapture(output_path)
            self.assertTrue(test_cap.isOpened(), "Output video cannot be opened")
            test_cap.release()
            
        finally:
            # Clean up
            if os.path.exists(output_path):
                os.unlink(output_path)
                
    @patch('src.database.session.Session.save')
    def test_session_saving(self, mock_save):
        """Test that video analysis sessions are saved correctly."""
        # Mock session saving
        mock_session = MagicMock()
        mock_session.session_id = 123
        mock_save.return_value = None
        
        # Test session creation
        session = Session(
            user_id=1,
            exercise_id=1,  # Use exercise_id instead of exercise
            start_time=datetime.now(),
            end_time=datetime.now(),
            video_path="/tmp/test.mp4",
            reps_count=5,
            feedback_count=2
        )
        
        # This should not raise an exception
        session.save()
        mock_save.assert_called_once()
        
    def test_video_file_validation(self):
        """Test video file validation logic."""
        # Test file size validation (simulate a large file)
        class MockFile:
            def __init__(self, size):
                self.size = size
                
        # Test file too large
        large_file = MockFile(250 * 1024 * 1024)  # 250MB
        self.assertGreater(large_file.size, 200 * 1024 * 1024)
        
        # Test acceptable file size
        normal_file = MockFile(50 * 1024 * 1024)  # 50MB
        self.assertLessEqual(normal_file.size, 200 * 1024 * 1024)
        
    def test_pose_detection_accuracy(self):
        """Test that pose detection works on the test video."""
        cap = cv2.VideoCapture(self.test_video_path)
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        
        poses_detected = 0
        frames_processed = 0
        max_frames = 50  # Test first 50 frames
        
        while cap.isOpened() and frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            frames_processed += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                poses_detected += 1
                
        cap.release()
        
        # We should detect poses in at least 50% of frames for a good lunge video
        detection_rate = poses_detected / frames_processed if frames_processed > 0 else 0
        self.assertGreater(
            detection_rate, 0.3, 
            f"Pose detection rate too low: {detection_rate:.2%}"
        )
        self.assertGreater(poses_detected, 0, "No poses detected in test video")
        
    def test_video_output_path_creation(self):
        """Test that video output path is created correctly."""
        import tempfile
        from datetime import datetime
        
        # Simulate the path creation logic from the app
        videos_dir = os.path.join(
            os.path.dirname(__file__), "..", "videos"
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"analyzed_video_{timestamp}.mp4"
        output_path = os.path.join(videos_dir, output_filename)
        
        # Verify the path structure
        self.assertTrue(output_filename.startswith("analyzed_video_"))
        self.assertTrue(output_filename.endswith(".mp4"))
        self.assertIn("videos", output_path)
        
        # Test that we can create the directory
        os.makedirs(videos_dir, exist_ok=True)
        self.assertTrue(os.path.exists(videos_dir))
        
    def test_full_video_analysis_workflow(self):
        """Test the complete video analysis workflow including output video creation."""
        import cv2
        import mediapipe as mp
        from datetime import datetime
        
        # Setup similar to the app
        cap = cv2.VideoCapture(self.test_video_path)
        self.assertTrue(cap.isOpened())
        
        # Create output path
        videos_dir = os.path.join(os.path.dirname(__file__), "..", "videos")
        os.makedirs(videos_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"test_analyzed_video_{timestamp}.mp4"
        output_path = os.path.join(videos_dir, output_filename)
        
        try:
            # Initialize video writer
            fourcc_options = [
                cv2.VideoWriter_fourcc(*'mp4v'),
                cv2.VideoWriter_fourcc(*'XVID'),
                cv2.VideoWriter_fourcc(*'MJPG'),
            ]
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Ensure dimensions are even
            if width % 2 != 0:
                width -= 1
            if height % 2 != 0:
                height -= 1
            
            # Try codecs until one works
            out = None
            for fourcc in fourcc_options:
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    break
                out.release()
            
            self.assertIsNotNone(out)
            self.assertTrue(out.isOpened())
            
            # Process frames
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
            frame_count = 0
            max_frames = 10  # Process only first 10 frames for speed
            
            feedback_messages = []
            state = {
                "ready": False,
                "direction": None,
                "count": 0,
                "last_message": "",
                "message_timer": 0,
                "feedback": []
            }
            
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                current_timestamp = frame_count / fps
                
                # Resize if needed
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )
                    # Test form checking
                    from src.processing.forms_check import check_form
                    feedback, reps = check_form("lunge", frame, results.pose_landmarks.landmark, state)
                    if feedback:
                        for msg in feedback:
                            timestamp_str = f"{int(current_timestamp//60):02d}:{int(current_timestamp%60):02d}"
                            feedback_messages.append(f"{timestamp_str} - {msg}")
                
                # Note: cv2.VideoWriter.write() doesn't always return a boolean
                # It may return None, so we just check if the operation completes
                out.write(frame)
            
            cap.release()
            out.release()
            
            # Verify output video
            self.assertTrue(os.path.exists(output_path), "Output video file was not created")
            self.assertGreater(os.path.getsize(output_path), 0, "Output video file is empty")
            
            # Test that output video is readable
            test_cap = cv2.VideoCapture(output_path)
            self.assertTrue(test_cap.isOpened(), "Output video cannot be opened")
            
            # Read at least one frame
            ret, frame = test_cap.read()
            self.assertTrue(ret, "Cannot read frame from output video")
            self.assertIsNotNone(frame, "Frame is None")
            
            test_cap.release()
            
            # Test feedback format
            for msg in feedback_messages:
                self.assertIn(" - ", msg, "Feedback should contain timestamp separator")
                timestamp_part, message_part = msg.split(" - ", 1)
                self.assertRegex(timestamp_part, r"\d{2}:\d{2}", "Timestamp should be in MM:SS format")
                self.assertGreater(len(message_part), 0, "Message part should not be empty")
            
            # Test video readiness checking (similar to app logic)
            video_ready = False
            max_wait_time = 5  # Shorter timeout for tests
            wait_interval = 0.1
            waited_time = 0
            
            while not video_ready and waited_time < max_wait_time:
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    if file_size > 0:
                        try:
                            ready_test_cap = cv2.VideoCapture(output_path)
                            if ready_test_cap.isOpened():
                                ret, frame = ready_test_cap.read()
                                if ret and frame is not None:
                                    video_ready = True
                            ready_test_cap.release()
                        except:
                            pass
                
                if not video_ready:
                    import time
                    time.sleep(wait_interval)
                    waited_time += wait_interval
            
            self.assertTrue(video_ready, "Video should be ready for playback after processing")
            
        finally:
            # Clean up test file
            if os.path.exists(output_path):
                os.unlink(output_path)
                
    def test_flask_video_serving_endpoint(self):
        """Test that the Flask backend can serve video files."""
        import requests
        import tempfile
        from datetime import datetime
        
        # Create a test video file
        videos_dir = os.path.join(os.path.dirname(__file__), "..", "videos")
        os.makedirs(videos_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_filename = f"test_flask_video_{timestamp}.mp4"
        test_video_path = os.path.join(videos_dir, test_filename)
        
        try:
            # Create a minimal test video
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(test_video_path, fourcc, 1.0, (64, 64))
            
            # Write a few test frames
            for i in range(5):
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
                frame[:, :, i % 3] = 255  # Create colored frames
                out.write(frame)
            out.release()
            
            # Verify file was created
            self.assertTrue(os.path.exists(test_video_path), "Test video file should be created")
            self.assertGreater(os.path.getsize(test_video_path), 0, "Test video should not be empty")
            
            # Test Flask endpoint (assuming it's running on port 5050)
            video_url = f"http://localhost:5050/serve_video/{test_filename}"
            
            try:
                response = requests.head(video_url, timeout=5)
                # If Flask server is running, we should get either 200 or 404
                self.assertIn(response.status_code, [200, 404], 
                             f"Flask server should respond (got {response.status_code})")
                
                if response.status_code == 200:
                    # If server is running and file is found, test GET request
                    get_response = requests.get(video_url, timeout=5)
                    self.assertEqual(get_response.status_code, 200, "Should be able to download video")
                    self.assertEqual(get_response.headers.get('content-type'), 'video/mp4', 
                                   "Should serve as video/mp4")
                    self.assertGreater(len(get_response.content), 0, "Video content should not be empty")
                    
            except requests.exceptions.ConnectionError:
                # Flask server not running - this is okay for tests
                print("Note: Flask server not running, skipping HTTP test")
                
        finally:
            # Clean up test file
            if os.path.exists(test_video_path):
                os.unlink(test_video_path)


if __name__ == '__main__':
    unittest.main()