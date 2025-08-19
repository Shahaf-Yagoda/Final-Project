# video_streamer.py
import sys
import os

# Add error handling for imports
try:
    from flask import Flask, Response, request, jsonify, send_file
    from flask_cors import CORS
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {sys.path}")
    print("💡 Try: pip install flask flask-cors")
    sys.exit(1)
import cv2
import time
import os
import numpy as np
from src.processing.forms_check import get_exercise_checker, init_state
from src.processing.pose_detector import PoseDetectorFactory, get_mp_drawing_utils
import mediapipe as mp
import sys
import json
import shutil
from typing import Dict, Optional, Tuple


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class VideoStreamManager:
    """Manages video streaming sessions with pose detection and form analysis."""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = get_mp_drawing_utils()
        self.pose_detector = PoseDetectorFactory.create_live_exercise_detector()
        self.session_states: Dict[str, dict] = {}
        self.video_writers: Dict[Tuple[int, str], cv2.VideoWriter] = {}
        self.video_temp_paths: Dict[Tuple[int, str], str] = {}
        # Throttling state: {user_id: {message: last_timestamp}}
        self.feedback_throttle: Dict[int, Dict[str, float]] = {}
    
    def get_session_key(self, user_id: int, exercise: str) -> str:
        """Generate a unique session key."""
        return f"{user_id}_{exercise}"
    
    def get_or_create_session_state(self, user_id: int, exercise: str) -> dict:
        """Get or create session state for a user and exercise."""
        key = self.get_session_key(user_id, exercise)
        if key not in self.session_states:
            self.session_states[key] = init_state(exercise)
        return self.session_states[key]
    
    def set_session_active(self, user_id: int, exercise: str, active: bool):
        """Set session active state."""
        key = self.get_session_key(user_id, exercise)
        if key not in self.session_states:
            self.session_states[key] = init_state(exercise)
        self.session_states[key]["session_active"] = active
    
    def save_reps_to_tempfile(self, user_id: int, reps: int):
        """Save rep count to temporary file."""
        with open(f"/tmp/reps_{user_id}.txt", "w") as f:
            f.write(str(reps))
    
    def append_session_detail(self, user_id: int, detail: dict):
        """Append session detail to temporary JSON file with error handling."""
        details_file = f"/tmp/sessiondetails_{user_id}.json"
        import fcntl
        
        try:
            # Use file locking to prevent concurrent access
            with open(details_file, "a+") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.seek(0)
                content = f.read().strip()
                
                if content:
                    try:
                        details = json.loads(content)
                        if not isinstance(details, list):
                            details = [details]
                    except json.JSONDecodeError:
                        details = []
                else:
                    details = []
                
                details.append(detail)
                
                f.seek(0)
                f.truncate()
                json.dump(details, f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            print(f"Error appending session detail for user {user_id}: {e}")
            # Fallback: write to a separate file with timestamp
            fallback_file = f"/tmp/sessiondetails_{user_id}_{int(time.time())}.json"
            try:
                with open(fallback_file, "w") as f:
                    json.dump([detail], f)
            except Exception as fallback_error:
                print(f"Fallback write also failed: {fallback_error}")
    
    def should_throttle_feedback(self, user_id: int, message: str, current_time: float, throttle_seconds: float = 2.0) -> bool:
        """Check if feedback should be throttled based on message and time."""
        if user_id not in self.feedback_throttle:
            self.feedback_throttle[user_id] = {}
        
        user_throttle = self.feedback_throttle[user_id]
        
        # Check if we've seen this exact message recently
        if message in user_throttle:
            time_since_last = current_time - user_throttle[message]
            return time_since_last < throttle_seconds
        
        return False
    
    def append_feedback(self, user_id: int, feedback: dict):
        """Append feedback to temporary JSON file with throttling and error handling."""
        message = feedback.get("message", "")
        current_time = feedback.get("timestamp", time.time())
        
        # Apply throttling - only save if message hasn't been seen recently
        if self.should_throttle_feedback(user_id, message, current_time):
            return  # Skip this feedback due to throttling
        
        # Update throttle tracking
        if user_id not in self.feedback_throttle:
            self.feedback_throttle[user_id] = {}
        self.feedback_throttle[user_id][message] = current_time
        
        feedback_file = f"/tmp/feedback_{user_id}.json"
        import fcntl
        
        try:
            # Use file locking to prevent concurrent access
            with open(feedback_file, "a+") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.seek(0)
                content = f.read().strip()
                
                if content:
                    try:
                        feedback_list = json.loads(content)
                        if not isinstance(feedback_list, list):
                            feedback_list = [feedback_list]
                    except json.JSONDecodeError:
                        feedback_list = []
                else:
                    feedback_list = []
                
                feedback_list.append(feedback)
                
                f.seek(0)
                f.truncate()
                json.dump(feedback_list, f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            print(f"Error appending feedback for user {user_id}: {e}")
            # Fallback: write to a separate file with timestamp
            fallback_file = f"/tmp/feedback_{user_id}_{int(time.time())}.json"
            try:
                with open(fallback_file, "w") as f:
                    json.dump([feedback], f)
            except Exception as fallback_error:
                print(f"Fallback feedback write also failed: {fallback_error}")
    
    def process_frame(self, frame: np.ndarray, exercise: str, user_id: int) -> np.ndarray:
        """Process a single video frame with pose detection and form analysis."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(image_rgb)

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            state = self.get_or_create_session_state(user_id, exercise)
            
            # Get exercise checker and analyze form
            exercise_checker = get_exercise_checker(exercise)
            feedback, reps = exercise_checker.check_form(frame, landmarks, state)
            
            self.save_reps_to_tempfile(user_id, reps)
            
            # Save session details only when reps > 0 (to satisfy database constraint)
            if reps > 0:
                detail = {
                    "timestamp": time.time(),
                    "rep_num": reps,
                    # Use new schema - no keypoints, focus on form analysis
                    "features_json": {
                        "form_correct": len(feedback) == 0,
                        "feedback_count": len(feedback),
                        "timestamp": time.time()
                    },
                    "is_correct": len(feedback) == 0,
                    "incorrect_duration": 0
                }
                self.append_session_detail(user_id, detail)
            
            # Save feedback
            for msg in feedback:
                self.append_feedback(user_id, {"timestamp": time.time(), "message": msg})
        
        return frame
    
    def setup_video_recording(self, user_id: int, exercise: str, cap: cv2.VideoCapture) -> Optional[cv2.VideoWriter]:
        """Setup video recording for the session with improved codec handling."""
        import tempfile
        import platform
        
        # Use appropriate temp directory for the platform
        system = platform.system().lower()
        if system == 'windows':
            temp_dir = tempfile.gettempdir()
            temp_video_path = os.path.join(temp_dir, f"user{user_id}_session.mp4")
        else:
            temp_video_path = f"/tmp/user{user_id}_session.mp4"
            
        self.video_temp_paths[(user_id, exercise)] = temp_video_path
        
        # Platform-specific codec priority for better Windows compatibility
        
        if system == 'windows':
            fourcc_options = [
                cv2.VideoWriter_fourcc(*'MJPG'),  # Motion JPEG (best Windows support)
                cv2.VideoWriter_fourcc(*'XVID'),  # XVID (good Windows compatibility)
                cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4 (fallback)
                cv2.VideoWriter_fourcc(*'H264'),  # H.264 (if available)
                cv2.VideoWriter_fourcc(*'avc1'),  # H.264 alternative
                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),  # Alternative MJPG format
                cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'),  # DivX codec
                cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),  # Alternative XVID
                -1,  # Default codec (last resort)
            ]
        else:
            # macOS/Linux codec priority
            fourcc_options = [
                cv2.VideoWriter_fourcc(*'H264'),  # H.264 (best browser support)
                cv2.VideoWriter_fourcc(*'avc1'),  # H.264 alternative
                cv2.VideoWriter_fourcc(*'XVID'),  # XVID (good compatibility)
                cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4 (fallback)
                cv2.VideoWriter_fourcc(*'MJPG'),  # Motion JPEG (last resort)
            ]
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Ensure dimensions are even numbers (required for some codecs)
        if width % 2 != 0:
            width -= 1
        if height % 2 != 0:
            height -= 1
        
        # Try each codec until one works
        out = None
        print(f"🎬 Attempting video recording setup:")
        print(f"   Platform: {system}")
        print(f"   Video path: {temp_video_path}")
        print(f"   Dimensions: {width}x{height}")
        print(f"   FPS: {fps}")
        
        for i, fourcc in enumerate(fourcc_options):
            try:
                codec_name = f"codec_{i}" if fourcc == -1 else str(fourcc)
                print(f"   Trying codec {i+1}/{len(fourcc_options)}: {codec_name}")
                
                test_out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
                if test_out.isOpened():
                    out = test_out
                    print(f"✅ Successfully initialized video recording with codec: {codec_name}")
                    break
                else:
                    test_out.release()
                    print(f"   ❌ Codec {codec_name} failed to open")
            except Exception as e:
                print(f"   ⚠️ Exception with codec {codec_name}: {e}")
                continue
        
        if out is None:
            print("❌ All codecs failed. Attempting fallback without video recording...")
            # Create a dummy video writer that doesn't actually record
            print("⚠️ Running in NO-VIDEO mode - session will work but no video will be saved")
            return None  # Signal that video recording is disabled
        
        self.video_writers[(user_id, exercise)] = out
        return out
    
    def generate_stream(self, exercise: str, user_id: int):
        """Generate video stream with pose detection and form analysis."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Camera not available")
        
        self.set_session_active(user_id, exercise, True)
        out = self.setup_video_recording(user_id, exercise, cap)
        
        try:
            time.sleep(1)  # Allow camera to warm up
            while True:
                # Check if session is still active
                key = self.get_session_key(user_id, exercise)
                if not self.session_states.get(key, {}).get("session_active", True):
                    break
                
                success, frame = cap.read()
                if not success:
                    break
                
                processed_frame = self.process_frame(frame, exercise, user_id)
                
                # Only write to video if recording is enabled
                if out is not None:
                    out.write(processed_frame)
                
                _, buffer = cv2.imencode('.jpg', processed_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            # Only release if video writer was successfully created
            if out is not None:
                out.release()
            cap.release()
    
    def stop_session(self, user_id: int, exercise: str) -> Optional[str]:
        """Stop session and move video to final location."""
        self.set_session_active(user_id, exercise, False)
        
        temp_video_path = self.video_temp_paths.get((user_id, exercise))
        if temp_video_path and os.path.exists(temp_video_path):
            videos_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "videos"))
            os.makedirs(videos_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            final_video_name = f"user{user_id}_session_{timestamp}.mp4"
            final_video_path = os.path.join(videos_dir, final_video_name)
            
            shutil.move(temp_video_path, final_video_path)
            return final_video_path
        return None


# Global video stream manager instance
video_manager = VideoStreamManager()

app = Flask(__name__)
CORS(app, supports_credentials=True)


# Legacy function compatibility - delegate to VideoStreamManager


@app.route("/", methods=["GET"])
def index():
    """Serve the main video streaming page."""
    exercise = request.args.get("exercise")
    user_id = request.args.get("user_id")

    if not exercise or not user_id:
        return "Missing parameters. Usage: /?exercise=press&user_id=1", 400

    return f"""
    <html>
        <head><title>Live Feedback</title></head>
        <body style="margin:0; background:black;">
            <img src="/video_feed?exercise={exercise}&user_id={user_id}" width="100%" />
        </body>
    </html>
    """

@app.route("/video_feed", methods=["GET"])
def video_feed():
    """Serve the video stream with pose detection and form analysis."""
    exercise = request.args.get("exercise")
    user_id = request.args.get("user_id")

    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return "Invalid user_id", 400

    return Response(
        video_manager.generate_stream(exercise, user_id),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/stop_session", methods=["POST"])
def stop_session():
    """Stop the current video session and save the recorded video."""
    data = request.get_json()
    user_id = data.get("user_id")
    exercise = data.get("exercise")
    
    final_video_path = video_manager.stop_session(user_id, exercise)
    
    return jsonify({
        "status": "stopped", 
        "video_path": final_video_path
    })

@app.route("/serve_video/<filename>", methods=["GET"])
def serve_video(filename):
    """Serve analyzed video files."""
    try:
        # Get the videos directory (same as used in app.py)
        videos_dir = os.path.join(os.path.dirname(__file__), "..", "..", "videos")
        video_path = os.path.join(videos_dir, filename)
        
        # Security check: ensure the file is within the videos directory
        if not os.path.abspath(video_path).startswith(os.path.abspath(videos_dir)):
            return "Access denied", 403
            
        # Check if file exists
        if not os.path.exists(video_path):
            return "Video not found", 404
            
        # Check if this is a download request
        is_download = request.args.get('download') == 'true'
        
        # Serve the video file
        response = send_file(
            video_path,
            mimetype='video/mp4',
            as_attachment=is_download,
            download_name=filename
        )
        
        # Set headers to support both streaming and downloads
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Range'
        response.headers['Access-Control-Allow-Methods'] = 'GET, HEAD, OPTIONS'
        response.headers['Access-Control-Expose-Headers'] = 'Accept-Ranges, Content-Length, Content-Range'
        
        # Set appropriate Content-Disposition header
        if is_download:
            response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        else:
            response.headers['Content-Disposition'] = f'inline; filename="{filename}"'
        
        return response
        
    except Exception as e:
        import traceback
        error_msg = f"Error serving video {filename}: {e}"
        print(error_msg)
        print(traceback.format_exc())
        return f"Server error: {str(e)}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, threaded=True)
