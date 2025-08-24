import streamlit as st
import os
import sys
import tempfile
import time
import subprocess
import cv2
import base64
import socket
from datetime import datetime
from streamlit_option_menu import option_menu
import mediapipe as mp
import requests
import threading

# Ensure local packages are accessible before any src import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Use absolute imports for src modules
from src.database.db_utils import save_session_to_db
from src.database.users.user import User
from src.database.session import Session
from src.database.session_details import SessionDetails
from src.database.system_feedback import SystemFeedback
from src.database.workout import Workout
from src.utils.temp_paths import (
    get_user_reps_path,
    get_user_session_details_path, 
    get_user_feedback_path,
    get_user_session_video_path,
    get_temp_path,
    TEMP_DIR
)
from src.processing.forms_check import check_form
from src.database.database_connection import get_connection

def format_datetime(dt):
    from datetime import datetime
    if isinstance(dt, datetime):
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    try:
        # Try to convert from timestamp (int or float)
        return datetime.fromtimestamp(dt).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return str(dt)

# Utility to check if Flask server is already running
def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

# Launch video_streamer.py if Flask not already running
if not is_port_open(5050) and "video_streamer_started" not in st.session_state:
    try:
        # Use the same Python interpreter as Streamlit and set PYTHONPATH
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        
        # Try to detect virtual environment python
        python_cmd = sys.executable
        if not python_cmd or 'python' not in python_cmd:
            python_cmd = "python3"
            
        print(f"🐍 Using Python: {python_cmd}")
        subprocess.Popen([python_cmd, "-m", "src.app.video_streamer"], env=env)
        st.session_state["video_streamer_started"] = True
        print("✅ video_streamer.py started.")
    except Exception as e:
        print("❌ Failed to start video_streamer.py:", e)
        print(f"🐍 Python executable: {sys.executable}")
        print("💡 Try running: source venv/bin/activate && streamlit run src/app/app.py")

# Remove duplicate imports - using the ones from db_utils instead
from datetime import datetime
import json
import time

def get_reps_count_from_tempfile(user_id):
    try:
        with open(get_user_reps_path(user_id), "r") as f:
            reps = int(f.read().strip())
            return reps
    except Exception as e:
        print("Could not read reps from temp:", e)
        return 0

def get_session_details_from_temp(user_id):
    details_file = get_user_session_details_path(user_id)
    if not os.path.exists(details_file):
        return []
    
    try:
        with open(details_file, "r") as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
        print(f"Warning: Could not read session details from {details_file}: {e}")
        # Try to salvage partial data by reading line by line
        try:
            details = []
            with open(details_file, "r") as f:
                lines = f.readlines()
                # Try to find valid JSON objects in the file
                current_json = ""
                brace_count = 0
                for line in lines:
                    current_json += line
                    brace_count += line.count('{') - line.count('}')
                    if brace_count == 0 and current_json.strip():
                        try:
                            obj = json.loads(current_json.strip())
                            if isinstance(obj, list):
                                details.extend(obj)
                            else:
                                details.append(obj)
                            current_json = ""
                        except json.JSONDecodeError:
                            current_json = ""
            return details
        except Exception as fallback_error:
            print(f"Could not salvage session details: {fallback_error}")
            
            # Try to read from fallback files
            try:
                import glob
                fallback_files = glob.glob(os.path.join(TEMP_DIR, f"sessiondetails_{user_id}_*.json"))
                details = []
                for fallback_file in fallback_files:
                    try:
                        with open(fallback_file, "r") as f:
                            fallback_data = json.load(f)
                            if isinstance(fallback_data, list):
                                details.extend(fallback_data)
                            else:
                                details.append(fallback_data)
                    except Exception:
                        continue
                return details
            except Exception:
                return []

def get_feedback_from_temp(user_id):
    feedback_file = get_user_feedback_path(user_id)
    if not os.path.exists(feedback_file):
        return []
    
    try:
        with open(feedback_file, "r") as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
        print(f"Warning: Could not read feedback from {feedback_file}: {e}")
        # Try to salvage partial data
        try:
            feedback = []
            with open(feedback_file, "r") as f:
                lines = f.readlines()
                current_json = ""
                brace_count = 0
                for line in lines:
                    current_json += line
                    brace_count += line.count('{') - line.count('}')
                    if brace_count == 0 and current_json.strip():
                        try:
                            obj = json.loads(current_json.strip())
                            if isinstance(obj, list):
                                feedback.extend(obj)
                            else:
                                feedback.append(obj)
                            current_json = ""
                        except json.JSONDecodeError:
                            current_json = ""
            return feedback
        except Exception as fallback_error:
            print(f"Could not salvage feedback: {fallback_error}")
            
            # Try to read from fallback files
            try:
                import glob
                fallback_files = glob.glob(os.path.join(TEMP_DIR, f"feedback_{user_id}_*.json"))
                feedback = []
                for fallback_file in fallback_files:
                    try:
                        with open(fallback_file, "r") as f:
                            fallback_data = json.load(f)
                            if isinstance(fallback_data, list):
                                feedback.extend(fallback_data)
                            else:
                                feedback.append(fallback_data)
                    except Exception:
                        continue
                return feedback
            except Exception:
                return []


# Background image
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

image_path = os.path.join(os.path.dirname(__file__), "../assets/background.jpg")
background_css = f"""
<style>
body::before {{
    content: "";
    background-image: url("data:image/jpeg;base64,{get_base64_image(image_path)}");
    background-size: cover;
    background-position: center;
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: -1;
    opacity: 0.25;
}}
</style>
"""
st.markdown(background_css, unsafe_allow_html=True)

# Load custom CSS
css_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Navigation setup
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None

def set_page(page_name):
    st.session_state.page = page_name

def logout():
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.page = "Home"

def get_exercise_id_by_name(exercise_name):
    from src.database.database_connection import get_connection
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT exercise_id FROM exercises WHERE exercise_name = %s', (exercise_name,))
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()

def move_video_file_async(temp_path, final_path):
    import shutil
    try:
        shutil.move(temp_path, final_path)
    except Exception as e:
        print(f"Error moving video file: {e}")

# ---------------- PAGES ---------------- #

# Home Page
if st.session_state.page == "Home":
    st.title("Welcome to Right Motion")

    if not st.session_state.logged_in:
        st.subheader("Please log in or register to continue.")
        col1, col2 = st.columns(2)
        with col1:
            st.button("🔐 Register", on_click=set_page, args=("Register",))
        with col2:
            st.button("🔑 Log In", on_click=set_page, args=("Login",))
    else:
        st.success(f"Logged in as {st.session_state.username}")
        st.subheader("Choose an option:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("📹 Analyze Video", on_click=set_page, args=("Analyze",))
        with col2:
            st.button("🏃 Live Exercise", on_click=set_page, args=("LiveExercise",))
        with col3:
            st.button("📜 Session History", on_click=set_page, args=("History",))
        
        # Center the Log Out button on its own row
        col_logout = st.columns([1, 1, 1])
        with col_logout[1]:
            st.button("🚪 Log Out", on_click=logout)

# Register Page
elif st.session_state.page == "Register":
    st.title("User Registration")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    # Profile fields
    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    dob = st.date_input("Date of Birth")
    height = st.number_input("Height (cm)", min_value=0)
    weight = st.number_input("Weight (kg)", min_value=0)
    user_type = st.selectbox("Role", ["user", "coach", "admin"])

    if st.button("Register"):
        if not password or not email:
            st.error("Email and password are required.")
        else:
            try:
                profile_data = {
                    "date_of_birth": str(dob),
                    "height": height,
                    "weight": weight
                }
                user = User.register(
                    email=email,
                    password=password,
                    profile_data=profile_data,
                    user_type=user_type,
                    first_name=first_name,
                    last_name=last_name
                )
                st.success(f"User registered with ID: {user.user_id}")
            except Exception as e:
                st.error(f"Registration failed: {e}")

    st.button("⬅️ Back to Home", on_click=set_page, args=("Home",))

# Login Page
elif st.session_state.page == "Login":
    st.title("User Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Log In"):
        user = User.authenticate(email, password)
        if user:
            st.session_state.logged_in = True
            st.session_state.user_id = user.user_id
            st.session_state.username = user.get_full_name()
            st.success(f"Login successful! Welcome, {user.get_full_name()}.")
            st.session_state.page = "Home"
            st.rerun()
        else:
            st.error("Login failed: Invalid credentials.")

    st.button("⬅️ Back to Home", on_click=set_page, args=("Home",))

# Analyze Video Page (OOP refactor)
elif st.session_state.page == "Analyze":
    st.session_state.is_analyze_video_flow = True
    st.title("Right Motion Video Analyzer")
    st.header("Upload a video to analyze")
    video_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    st.header("Select Exercise Type")
    selected_exercise = option_menu(
        menu_title=None,
        options=["lunge", "overhead_press", "plank"],
        icons=["1-circle-fill", "2-circle-fill", "3-circle-fill"],
        orientation="horizontal",
    )

    user_id = st.session_state.get("user_id")
    if not user_id:
        st.error("You must be logged in to analyze a video.")
    else:
        if st.button("Start Analysis"):
            if not video_file:
                st.error("Please select a video file first.")
            elif video_file.size > 200 * 1024 * 1024:  # 200MB limit
                st.error("Video file is too large. Please upload a file under 200MB.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("🔍 Processing video file...")
                
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                        tmp.write(video_file.read())
                        video_path = tmp.name
                    
                    progress_bar.progress(20)
                    status_text.text("📹 Opening video...")

                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        st.error("Error: Unable to open video file. Please ensure it's a valid MP4, AVI, or MOV file.")
                    else:
                        progress_bar.progress(30)
                        status_text.text("🤖 Initializing AI pose detection...")
                        
                        from src.processing.pose_detector import PoseDetectorFactory, get_mp_drawing_utils, get_mp_pose_solutions
                        pose = PoseDetectorFactory.create_video_analysis_detector()
                        mp_drawing = get_mp_drawing_utils()
                        mp_pose = get_mp_pose_solutions()
                        
                        # Create output path in videos directory for proper playback
                        videos_dir = os.path.join(os.path.dirname(__file__), "..", "..", "videos")
                        os.makedirs(videos_dir, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_filename = f"analyzed_video_{timestamp}.mp4"
                        output_path = os.path.join(videos_dir, output_filename)
                        
                        # Try different codecs for better browser compatibility (H.264 first)
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
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        # Ensure dimensions are even numbers (required for some codecs)
                        if width % 2 != 0:
                            width -= 1
                        if height % 2 != 0:
                            height -= 1
                        
                        # Try codecs until one works and log which one succeeded
                        out = None
                        used_codec = None
                        codec_names = ['H264', 'avc1', 'XVID', 'mp4v', 'MJPG']
                        
                        for i, fourcc in enumerate(fourcc_options):
                            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                            if out.isOpened():
                                used_codec = codec_names[i]
                                print(f"✅ Successfully initialized video writer with {used_codec} codec")
                                break
                            out.release()
                        
                        if not out or not out.isOpened():
                            raise Exception("Could not initialize video writer with any supported codec (H264, avc1, XVID, mp4v, MJPG)")
                        frame_count = 0
                        rep_count = 0
                        state = {
                            "ready": False,
                            "direction": None,
                            "count": 0,
                            "last_message": "",
                            "message_timer": 0,
                            "feedback": [],
                            "angle_buffer_l": [],
                            "angle_buffer_r": []
                        }
                        feedback_messages = []
                        
                        progress_bar.progress(40)
                        status_text.text(f"🏃 Analyzing {selected_exercise} form...")
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame_count += 1
                            
                            # Calculate timestamp for this frame
                            current_timestamp = frame_count / fps
                            
                            # Update progress during processing
                            if frame_count % 30 == 0:  # Update every 30 frames
                                progress_percent = min(40 + int((frame_count / total_frames) * 50), 90)
                                progress_bar.progress(progress_percent)
                                status_text.text(f"🏃 Analyzing frame {frame_count}/{total_frames}...")
                            
                            # Resize frame if needed to match expected dimensions
                            if frame.shape[1] != width or frame.shape[0] != height:
                                frame = cv2.resize(frame, (width, height))
                            
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = pose.process(rgb_frame)
                            if results.pose_landmarks:
                                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                                # Use the same check_form logic as live mode
                                from src.processing.forms_check import check_form
                                feedback, reps = check_form(selected_exercise, frame, results.pose_landmarks.landmark, state)
                                rep_count = max(rep_count, reps)
                                if feedback:
                                    # Add timestamp to feedback messages
                                    for msg in feedback:
                                        feedback_messages.append({
                                            "timestamp": current_timestamp,
                                            "message": msg
                                        })
                            
                            # Write frame to output video
                            # Note: OpenCV's write() may not return a boolean value consistently
                            out.write(frame)
                        
                        cap.release()
                        out.release()
                        cv2.destroyAllWindows()
                        
                        progress_bar.progress(95)
                        status_text.text("💾 Saving session data...")
                        
                        # Save session to DB using OOP Session class
                        exercise_id = get_exercise_id_by_name(selected_exercise)
                        # Create workout for this session
                        workout = Workout.create(
                            user_id=user_id,
                            workout_date=datetime.now().date(),
                            start_time=datetime.now()
                        )
                        
                        session = Session(
                            exercise_id=exercise_id,
                            workout_id=workout.workout_id,
                            session_order=1,
                            start_time=datetime.now(),
                            end_time=datetime.now(),
                            duration=0,
                            planned_reps=rep_count,
                            actual_reps=rep_count,
                            session_status='completed',
                            video_path=output_path
                        )
                        session.save()
                        
                        progress_bar.progress(98)
                        status_text.text("🎬 Preparing video for playback...")
                        
                        # Wait for video file to be fully written and accessible
                        video_ready = False
                        max_wait_time = 30  # Maximum 30 seconds
                        wait_interval = 0.5  # Check every 500ms
                        waited_time = 0
                        
                        while not video_ready and waited_time < max_wait_time:
                            if os.path.exists(output_path):
                                file_size = os.path.getsize(output_path)
                                if file_size > 0:
                                    # Try to open the video file to verify it's readable
                                    try:
                                        test_cap = cv2.VideoCapture(output_path)
                                        if test_cap.isOpened():
                                            # Try to read one frame to ensure file integrity
                                            ret, frame = test_cap.read()
                                            if ret and frame is not None:
                                                video_ready = True
                                        test_cap.release()
                                    except:
                                        pass
                            
                            if not video_ready:
                                time.sleep(wait_interval)
                                waited_time += wait_interval
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Analysis complete!")
                        
                        st.success(f"🎉 Analysis complete! Session saved (ID: {session.session_id}) with {rep_count} reps.")
                        if used_codec:
                            st.info(f"📹 Video encoded with {used_codec} codec for optimal browser compatibility.")
                        
                        # Display analyzed video with proper readiness checking
                        st.subheader("📹 Analyzed Video:")
                        if video_ready and os.path.exists(output_path):
                            file_size = os.path.getsize(output_path)
                            st.info(f"Video file size: {file_size / (1024*1024):.1f} MB")
                            try:
                                # Use Flask backend to serve the video
                                video_url = f"http://localhost:5050/serve_video/{output_filename}"
                                
                                # Create HTML video element with enhanced error handling and codec support
                                video_html = f"""
                                <div id="video-container">
                                    <video id="analyzed-video" width="100%" height="auto" controls preload="metadata" crossorigin="anonymous">
                                        <source src="{video_url}" type="video/mp4; codecs=&quot;mp4v.20.9,mp4a.40.2&quot;">
                                        <source src="{video_url}" type="video/mp4">
                                        Your browser does not support the video tag or the video codec.
                                    </video>
                                    <div id="video-status" style="margin-top: 10px; padding: 10px; border-radius: 4px;"></div>
                                    <div id="retry-container" style="margin-top: 10px; display: none;">
                                        <button onclick="retryVideo()" style="padding: 8px 16px; background-color: #ff6b6b; color: white; border: none; border-radius: 4px; cursor: pointer;">
                                            🔄 Retry Loading Video
                                        </button>
                                    </div>
                                    <div id="codec-info" style="margin-top: 10px; padding: 8px; background-color: #f0f0f0; border-radius: 4px; display: none;">
                                        <strong>Technical Info:</strong> <span id="codec-details"></span>
                                    </div>
                                </div>
                                
                                <script>
                                    const video = document.getElementById('analyzed-video');
                                    const status = document.getElementById('video-status');
                                    const retryContainer = document.getElementById('retry-container');
                                    const codecInfo = document.getElementById('codec-info');
                                    const codecDetails = document.getElementById('codec-details');
                                    
                                    function updateStatus(message, color, background) {{
                                        status.style.display = 'block';
                                        status.innerHTML = message;
                                        status.style.color = color;
                                        status.style.backgroundColor = background || 'transparent';
                                    }}
                                    
                                    function showCodecInfo(details) {{
                                        codecDetails.innerHTML = details;
                                        codecInfo.style.display = 'block';
                                    }}
                                    
                                    // Enhanced event listeners with detailed error reporting
                                    video.addEventListener('loadstart', function() {{
                                        updateStatus('⏳ Starting to load video...', '#666', '#f0f8ff');
                                        retryContainer.style.display = 'none';
                                        codecInfo.style.display = 'none';
                                    }});
                                    
                                    video.addEventListener('loadedmetadata', function() {{
                                        updateStatus('📊 Video metadata loaded...', '#666', '#f0f8ff');
                                        showCodecInfo(`Duration: ${{video.duration.toFixed(1)}}s, Dimensions: ${{video.videoWidth}}x${{video.videoHeight}}`);
                                    }});
                                    
                                    video.addEventListener('loadeddata', function() {{
                                        updateStatus('📹 Video data loaded...', '#666', '#f0f8ff');
                                    }});
                                    
                                    video.addEventListener('canplay', function() {{
                                        updateStatus('✅ Video ready to play!', 'white', '#28a745');
                                        retryContainer.style.display = 'none';
                                    }});
                                    
                                    video.addEventListener('canplaythrough', function() {{
                                        updateStatus('✅ Video fully loaded and ready!', 'white', '#28a745');
                                    }});
                                    
                                    video.addEventListener('error', function(e) {{
                                        const error = video.error;
                                        let errorMessage = '❌ Video playback error: ';
                                        
                                        if (error) {{
                                            switch(error.code) {{
                                                case error.MEDIA_ERR_ABORTED:
                                                    errorMessage += 'Playback aborted by user';
                                                    break;
                                                case error.MEDIA_ERR_NETWORK:
                                                    errorMessage += 'Network error while loading video';
                                                    break;
                                                case error.MEDIA_ERR_DECODE:
                                                    errorMessage += 'Video decode error (codec not supported)';
                                                    break;
                                                case error.MEDIA_ERR_SRC_NOT_SUPPORTED:
                                                    errorMessage += 'Video format/codec not supported by browser';
                                                    break;
                                                default:
                                                    errorMessage += 'Unknown error';
                                            }}
                                        }} else {{
                                            errorMessage += 'Unknown playback error';
                                        }}
                                        
                                        updateStatus(errorMessage, 'white', '#dc3545');
                                        retryContainer.style.display = 'block';
                                        showCodecInfo('This video uses MPEG-4 codec which may not be supported by all browsers.');
                                    }});
                                    
                                    video.addEventListener('stalled', function() {{
                                        updateStatus('⚠️ Video playback stalled (buffering...)', '#ff8c00', '#fff3cd');
                                    }});
                                    
                                    video.addEventListener('waiting', function() {{
                                        updateStatus('⏳ Buffering video data...', '#666', '#f0f8ff');
                                    }});
                                    
                                    function retryVideo() {{
                                        updateStatus('🔄 Retrying video load...', '#666', '#f0f8ff');
                                        video.load();
                                    }}
                                    
                                    // Test codec support
                                    const codecSupport = {{
                                        'mp4v': video.canPlayType('video/mp4; codecs="mp4v.20.9"'),
                                        'h264': video.canPlayType('video/mp4; codecs="avc1.42E01E"'),
                                        'basic': video.canPlayType('video/mp4')
                                    }};
                                    
                                    console.log('Codec support:', codecSupport);
                                    
                                    // Test server connectivity
                                    fetch('{video_url}', {{method: 'HEAD'}})
                                        .then(response => {{
                                            if (!response.ok) {{
                                                throw new Error(`Server responded with status: ${{response.status}}`);
                                            }}
                                            console.log('✅ Video server is accessible');
                                        }})
                                        .catch(error => {{
                                            updateStatus('⚠️ Cannot connect to video server. Please ensure the Flask backend is running.', 'white', '#fd7e14');
                                            retryContainer.style.display = 'block';
                                            console.error('Server connectivity error:', error);
                                        }});
                                </script>
                                
                                <p><em>Video URL: <a href="{video_url}" target="_blank">{video_url}</a></em></p>
                                """
                                
                                st.markdown(video_html, unsafe_allow_html=True)
                                st.success("✅ Video loaded and ready for playback!")
                                
                                # Always provide download option as backup using same Flask endpoint
                                download_url = f"http://localhost:5050/serve_video/{output_filename}?download=true"
                                st.markdown(f"""
                                <a href="{download_url}" style="
                                    display: inline-block;
                                    padding: 0.5rem 1rem;
                                    background-color: #ff6b6b;
                                    color: white;
                                    text-decoration: none;
                                    border-radius: 0.25rem;
                                    font-weight: 500;
                                ">📥 Download Analyzed Video</a>
                                """, unsafe_allow_html=True)
                                    
                            except Exception as e:
                                st.error(f"❌ Could not display video: {str(e)}")
                                st.info(f"Video saved at: {output_path}")
                                # Offer download option as fallback using same Flask endpoint
                                try:
                                    download_url = f"http://localhost:5050/serve_video/{output_filename}?download=true"
                                    st.markdown(f"""
                                    <a href="{download_url}" style="
                                        display: inline-block;
                                        padding: 0.5rem 1rem;
                                        background-color: #ff6b6b;
                                        color: white;
                                        text-decoration: none;
                                        border-radius: 0.25rem;
                                        font-weight: 500;
                                    ">📥 Download Analyzed Video</a>
                                    """, unsafe_allow_html=True)
                                except Exception as download_err:
                                    st.error(f"Download failed: {download_err}")
                        elif os.path.exists(output_path):
                            file_size = os.path.getsize(output_path)
                            if file_size > 0:
                                st.warning("⏳ Video file exists but may not be fully ready for playback yet.")
                                st.info(f"Video saved at: {output_path}")
                                # Still offer download option using same Flask endpoint
                                try:
                                    download_url = f"http://localhost:5050/serve_video/{output_filename}?download=true"
                                    st.markdown(f"""
                                    <a href="{download_url}" style="
                                        display: inline-block;
                                        padding: 0.5rem 1rem;
                                        background-color: #ff6b6b;
                                        color: white;
                                        text-decoration: none;
                                        border-radius: 0.25rem;
                                        font-weight: 500;
                                    ">📥 Download Analyzed Video</a>
                                    """, unsafe_allow_html=True)
                                except Exception as download_err:
                                    st.error(f"Download failed: {download_err}")
                            else:
                                st.error("❌ Video file is empty (0 bytes). Analysis may have failed.")
                        else:
                            st.error("❌ Video file was not created. Check file permissions and available disk space.")
                        
                        # Display feedback in a clean, readable format
                        if feedback_messages:
                            st.subheader("📝 Form Feedback:")
                            
                            # Deduplicate feedback before saving to database
                            unique_feedback_for_db = []
                            seen_combinations = set()
                            
                            for fb in feedback_messages:
                                timestamp = fb.get("timestamp")
                                message = fb.get("message")
                                if timestamp and message:
                                    # Create a deduplication key with message and rounded timestamp (to nearest 5 seconds)
                                    rounded_timestamp = round(timestamp / 5) * 5
                                    dedup_key = f"{message}_{rounded_timestamp}"
                                    
                                    if dedup_key not in seen_combinations:
                                        unique_feedback_for_db.append(fb)
                                        seen_combinations.add(dedup_key)
                            
                            # Create formatted feedback with proper timestamps
                            formatted_feedback = []
                            session_start_time = None
                            
                            # Find the earliest timestamp to use as session start
                            valid_timestamps = [fb.get("timestamp") for fb in feedback_messages if fb.get("timestamp")]
                            if valid_timestamps:
                                session_start_time = min(valid_timestamps)
                            
                            for fb in feedback_messages:
                                try:
                                    timestamp = fb.get("timestamp")
                                    message = fb.get("message")
                                    if timestamp and message and session_start_time:
                                        # Calculate relative seconds from session start
                                        relative_seconds = timestamp - session_start_time
                                        # Ensure non-negative timestamps
                                        relative_seconds = max(0, relative_seconds)
                                        # Convert to MM:SS format
                                        minutes = int(relative_seconds // 60)
                                        seconds = int(relative_seconds % 60)
                                        timestamp_str = f"{minutes:02d}:{seconds:02d}"
                                        formatted_feedback.append(f"{timestamp_str} - {message}")
                                    elif message:
                                        # If no valid timestamp, use message without timestamp
                                        formatted_feedback.append(str(message))
                                except Exception as e:
                                    # Fallback for any timestamp conversion issues
                                    if fb.get("message"):
                                        formatted_feedback.append(str(fb.get("message")))
                            
                            # Smart message aggregation instead of simple deduplication
                            def aggregate_feedback_messages(formatted_feedback):
                                """Group similar feedback messages with occurrence counts and time ranges."""
                                message_groups = {}
                                
                                for msg in formatted_feedback:
                                    if ' - ' in msg:
                                        timestamp_str, message = msg.split(' - ', 1)
                                        # Extract just the core message (ignore severity words)
                                        core_message = message.replace("CRITICAL: ", "").replace("Don't ", "").strip()
                                        
                                        if core_message not in message_groups:
                                            message_groups[core_message] = {
                                                'original_message': message,
                                                'timestamps': [],
                                                'count': 0,
                                                'has_critical': False
                                            }
                                        
                                        message_groups[core_message]['timestamps'].append(timestamp_str)
                                        message_groups[core_message]['count'] += 1
                                        if "CRITICAL" in message:
                                            message_groups[core_message]['has_critical'] = True
                                    else:
                                        # Handle messages without timestamps
                                        if msg not in message_groups:
                                            message_groups[msg] = {
                                                'original_message': msg,
                                                'timestamps': [],
                                                'count': 1,
                                                'has_critical': False
                                            }
                                
                                # Create aggregated feedback messages
                                aggregated = []
                                for _, data in message_groups.items():
                                    if data['timestamps']:
                                        first_time = data['timestamps'][0]
                                        last_time = data['timestamps'][-1]
                                        
                                        if data['count'] == 1:
                                            # Single occurrence - use original format
                                            aggregated.append(f"{first_time} - {data['original_message']}")
                                        else:
                                            # Multiple occurrences - show aggregated format
                                            severity_prefix = "⚠️ CRITICAL: " if data['has_critical'] else ""
                                            if first_time == last_time:
                                                aggregated.append(f"{first_time} - {severity_prefix}{data['original_message']} (occurred {data['count']} times)")
                                            else:
                                                aggregated.append(f"{first_time}-{last_time} - {severity_prefix}{data['original_message']} (occurred {data['count']} times)")
                                    else:
                                        # No timestamp - just add the message
                                        aggregated.append(data['original_message'])
                                
                                return aggregated
                            
                            unique_feedback = aggregate_feedback_messages(formatted_feedback)
                            
                            # Display feedback with enhanced styling
                            st.markdown("""
                            <style>
                            .feedback-container {
                                max-height: 300px;
                                overflow-y: auto;
                                border: 1px solid #ddd;
                                border-radius: 8px;
                                padding: 1rem;
                                background-color: #f8f9fa;
                                margin: 10px 0;
                            }
                            .feedback-item {
                                background-color: white;
                                padding: 0.5rem;
                                margin-bottom: 0.5rem;
                                border-left: 4px solid #ff6b6b;
                                border-radius: 4px;
                                font-family: 'Source Code Pro', monospace;
                            }
                            </style>
                            """, unsafe_allow_html=True)
                            
                            if unique_feedback:
                                feedback_html = '<div class="feedback-container">'
                                for i, msg in enumerate(unique_feedback, 1):
                                    if ' - ' in msg:
                                        timestamp, message = msg.split(' - ', 1)
                                        feedback_html += f'<div class="feedback-item"><strong>{timestamp}</strong> – {message}</div>'
                                    else:
                                        feedback_html += f'<div class="feedback-item">{msg}</div>'
                                feedback_html += '</div>'
                                
                                st.markdown(feedback_html, unsafe_allow_html=True)
                                st.caption(f"Generated {len(unique_feedback)} feedback messages during your session.")
                            else:
                                st.info("Great job! No form corrections needed during your session.")
                        else:
                            st.success("✅ Great form! No issues detected.")
                        
                        # Clean up temporary input file only
                        try:
                            os.unlink(video_path)
                        except:
                            pass
                            
                except Exception as e:
                    st.error(f"❌ Error processing video: {str(e)}")
                    st.info("Please ensure your video file is a valid MP4, AVI, or MOV format and try again.")
                    # Clean up temporary files on error
                    try:
                        if 'video_path' in locals():
                            os.unlink(video_path)
                        if 'output_path' in locals() and os.path.exists(output_path):
                            os.unlink(output_path)
                    except:
                        pass
    st.button("⬅️ Back to Home", on_click=set_page, args=("Home",))

# Live Exercise Page
elif st.session_state.page == "LiveExercise":
    if not st.session_state.logged_in:
        st.warning("Please log in to access this page.")
        st.button("🔑 Go to Login", on_click=set_page, args=("Login",))
    else:
        st.title("Live Exercise Tracker")
        exercise = option_menu(None, ["lunge", "overhead_press", "plank"],
                               icons=["1-circle-fill", "2-circle-fill", "3-circle-fill"],
                               orientation="horizontal")

        if st.button("Start Live Tracking"):
            st.session_state["start_streaming"] = True
            st.session_state["selected_exercise"] = exercise
            st.session_state["start_time"] = datetime.now().isoformat()
            st.session_state["reps_count"] = 0

        if st.session_state.get("start_streaming", False):
            user_id = st.session_state.get("user_id")
            url = f"http://localhost:5050?exercise={exercise}&user_id={user_id}"
            
            # Create responsive video container
            st.markdown("""
            <style>
            .responsive-video-container {
                position: relative;
                width: 100%;
                height: 0;
                padding-bottom: 56.25%; /* 16:9 aspect ratio */
                overflow: hidden;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .responsive-video-container iframe {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                border: none;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="responsive-video-container">
                <iframe src="{url}" allowfullscreen></iframe>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🛑 Stop Live Tracking"):
                # Signal backend to stop session immediately
                try:
                    user_id = st.session_state.get("user_id")
                    exercise_name = st.session_state.get("selected_exercise")
                    requests.post("http://localhost:5050/stop_session", json={"user_id": user_id, "exercise": exercise_name})
                except Exception as e:
                    st.warning(f"Could not signal backend to stop session: {e}")
                st.session_state["start_streaming"] = False
                st.session_state["stop_time"] = datetime.now().isoformat()
                try:
                    with open(get_user_reps_path(user_id)) as f:
                        reps = int(f.read().strip())
                        os.remove(get_user_reps_path(user_id))
                        st.session_state["reps_count"] = reps
                except:
                    reps = 234  # fallback if file not found

                start = datetime.fromisoformat(st.session_state["start_time"])
                end = datetime.fromisoformat(st.session_state["stop_time"])
                exercise_id = get_exercise_id_by_name(exercise_name)
                duration_sec = int((end - start).total_seconds())

                # Save video to videos directory with unique name
                videos_dir = os.path.join(os.path.dirname(__file__), "..", "..", "videos")
                os.makedirs(videos_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_filename = f"user{user_id}_session_{timestamp}.mp4"
                video_path = os.path.join(videos_dir, video_filename)
                temp_video_path = get_user_session_video_path(user_id)

                # Move video file in a background thread
                if os.path.exists(temp_video_path):
                    thread = threading.Thread(target=move_video_file_async, args=(temp_video_path, video_path))
                    thread.start()

                # Create workout for this session using comprehensive schema
                workout = Workout.create(
                    user_id=user_id,
                    workout_date=start.date(),
                    start_time=start
                )
                
                # Create session with comprehensive schema
                session = Session(
                    exercise_id=exercise_id,
                    workout_id=workout.workout_id,  # Link to workout
                    session_order=1,  # First exercise in workout
                    start_time=start,
                    end_time=end,
                    duration=duration_sec,  # Use new field name
                    planned_reps=reps,  # Assume planned equals actual for live sessions
                    actual_reps=reps,   # Use new field name
                    session_status='completed',
                    video_path=video_path
                )
                session.save()
                
                # Finish the workout
                workout.finish(end_time=end)
                
                st.success(f"✅ Session saved to database (ID: {session.session_id}) with Workout (ID: {workout.workout_id})")

                # Save SessionDetails to DB using new comprehensive schema
                details = get_session_details_from_temp(user_id)
                details_saved = 0
                for detail in details:
                    try:
                        rep_num = detail.get("rep_num", 0)
                        # Only save if rep_number > 0 to satisfy database constraint
                        if rep_num > 0:
                            SessionDetails.create(
                                session_id=session.session_id,
                                rep_number=rep_num,
                                features_json=detail.get("features_json", {}),
                                is_correct_form=detail.get("is_correct", False),
                                incorrect_duration=detail.get("incorrect_duration", 0),
                                timestamp=datetime.fromtimestamp(detail.get("timestamp", time.time()))
                            )
                            details_saved += 1
                        else:
                            # Skip saving details for rep_num = 0 (before first rep)
                            print(f"Skipping session detail with rep_num=0")
                    except Exception as e:
                        print(f"Error saving session detail: {e}")
                # Clean up temp files after saving
                details_file = get_user_session_details_path(user_id)
                if os.path.exists(details_file):
                    os.remove(details_file)
                    
                # Also clean up any fallback files
                try:
                    import glob
                    fallback_files = glob.glob(os.path.join(TEMP_DIR, f"sessiondetails_{user_id}_*.json"))
                    for fallback_file in fallback_files:
                        try:
                            os.remove(fallback_file)
                        except Exception:
                            pass
                    
                    feedback_fallback_files = glob.glob(os.path.join(TEMP_DIR, f"feedback_{user_id}_*.json"))
                    for fallback_file in feedback_fallback_files:
                        try:
                            os.remove(fallback_file)
                        except Exception:
                            pass
                except Exception:
                    pass
                    
                st.info(f"Saved {details_saved} session details to the database.")

                # Process and display Form Feedback
                feedback = get_feedback_from_temp(user_id)
                if feedback:
                    # Deduplicate feedback before saving to database
                    unique_feedback_for_db = []
                    seen_combinations = set()
                    
                    for fb in feedback:
                        timestamp = fb.get("timestamp")
                        message = fb.get("message")
                        if timestamp and message:
                            # Create a deduplication key with message and rounded timestamp (to nearest 5 seconds)
                            rounded_timestamp = round(timestamp / 5) * 5
                            dedup_key = f"{message}_{rounded_timestamp}"
                            
                            if dedup_key not in seen_combinations:
                                unique_feedback_for_db.append(fb)
                                seen_combinations.add(dedup_key)
                    
                    # Save deduplicated feedback to database
                    feedback_saved = 0
                    for fb in unique_feedback_for_db:
                        try:
                            timestamp = fb.get("timestamp")
                            message = fb.get("message")
                            # Use new SystemFeedback OOP model
                            SystemFeedback.create(
                                session_id=session.session_id,
                                message=message,
                                feedback_type="form_correction",
                                timestamp=datetime.fromtimestamp(timestamp) if isinstance(timestamp, (int, float)) else timestamp
                            )
                            feedback_saved += 1
                        except Exception as e:
                            print(f"Error saving feedback: {e}")
                    
                    if feedback_saved > 0:
                        st.info(f"Saved {feedback_saved} feedback messages to the database.")
                    
                    # Display Form Feedback with same styling as Analyze Video page
                    st.subheader("📝 Form Feedback:")
                    
                    # Create formatted feedback with proper timestamps
                    formatted_feedback = []
                    session_start_time = None
                    
                    # Find the earliest timestamp to use as session start
                    valid_timestamps = [fb.get("timestamp") for fb in feedback if fb.get("timestamp")]
                    if valid_timestamps:
                        session_start_time = min(valid_timestamps)
                    
                    for fb in feedback:
                        try:
                            timestamp = fb.get("timestamp")
                            message = fb.get("message")
                            if timestamp and message and session_start_time:
                                # Calculate relative seconds from session start
                                relative_seconds = timestamp - session_start_time
                                # Ensure non-negative timestamps
                                relative_seconds = max(0, relative_seconds)
                                # Convert to MM:SS format
                                minutes = int(relative_seconds // 60)
                                seconds = int(relative_seconds % 60)
                                timestamp_str = f"{minutes:02d}:{seconds:02d}"
                                formatted_feedback.append(f"{timestamp_str} - {message}")
                            elif message:
                                # If no valid timestamp, use message without timestamp
                                formatted_feedback.append(str(message))
                        except Exception as e:
                            # Fallback for any timestamp conversion issues
                            if fb.get("message"):
                                formatted_feedback.append(str(fb.get("message")))
                    
                    # Smart message aggregation instead of simple deduplication
                    def aggregate_feedback_messages(formatted_feedback):
                        """Group similar feedback messages with occurrence counts and time ranges."""
                        message_groups = {}
                        
                        for msg in formatted_feedback:
                            if ' - ' in msg:
                                timestamp_str, message = msg.split(' - ', 1)
                                # Extract just the core message (ignore severity words)
                                core_message = message.replace("CRITICAL: ", "").replace("Don't ", "").strip()
                                
                                if core_message not in message_groups:
                                    message_groups[core_message] = {
                                        'original_message': message,
                                        'timestamps': [],
                                        'count': 0,
                                        'has_critical': False
                                    }
                                
                                message_groups[core_message]['timestamps'].append(timestamp_str)
                                message_groups[core_message]['count'] += 1
                                if "CRITICAL" in message:
                                    message_groups[core_message]['has_critical'] = True
                            else:
                                # Handle messages without timestamps
                                if msg not in message_groups:
                                    message_groups[msg] = {
                                        'original_message': msg,
                                        'timestamps': [],
                                        'count': 1,
                                        'has_critical': False
                                    }
                        
                        # Create aggregated feedback messages
                        aggregated = []
                        for _, data in message_groups.items():
                            if data['timestamps']:
                                first_time = data['timestamps'][0]
                                last_time = data['timestamps'][-1]
                                
                                if data['count'] == 1:
                                    # Single occurrence - use original format
                                    aggregated.append(f"{first_time} - {data['original_message']}")
                                else:
                                    # Multiple occurrences - show aggregated format
                                    severity_prefix = "⚠️ CRITICAL: " if data['has_critical'] else ""
                                    if first_time == last_time:
                                        aggregated.append(f"{first_time} - {severity_prefix}{data['original_message']} (occurred {data['count']} times)")
                                    else:
                                        aggregated.append(f"{first_time}-{last_time} - {severity_prefix}{data['original_message']} (occurred {data['count']} times)")
                            else:
                                # No timestamp - just add the message
                                aggregated.append(data['original_message'])
                        
                        return aggregated
                    
                    unique_feedback = aggregate_feedback_messages(formatted_feedback)
                    
                    # Display feedback with enhanced styling (same as Analyze Video page)
                    st.markdown("""
                    <style>
                    .feedback-container {
                        max-height: 300px;
                        overflow-y: auto;
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        padding: 1rem;
                        background-color: #f8f9fa;
                        margin: 10px 0;
                    }
                    .feedback-item {
                        background-color: white;
                        padding: 0.5rem;
                        margin-bottom: 0.5rem;
                        border-left: 4px solid #ff6b6b;
                        border-radius: 4px;
                        font-family: 'Source Code Pro', monospace;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    if unique_feedback:
                        feedback_html = '<div class="feedback-container">'
                        for i, msg in enumerate(unique_feedback, 1):
                            if ' - ' in msg:
                                timestamp, message = msg.split(' - ', 1)
                                feedback_html += f'<div class="feedback-item"><strong>{timestamp}</strong> – {message}</div>'
                            else:
                                feedback_html += f'<div class="feedback-item">{msg}</div>'
                        feedback_html += '</div>'
                        
                        st.markdown(feedback_html, unsafe_allow_html=True)
                        st.caption(f"Generated {len(unique_feedback)} feedback messages during your live session.")
                    else:
                        st.info("Great job! No form corrections needed during your session.")
                else:
                    st.info("No feedback data available for this session.")

                # Wait for video file to be fully written (poll for up to 10 seconds)
                max_wait = 60
                waited = 0
                while not (os.path.exists(video_path) and os.path.getsize(video_path) > 0):
                    if waited >= max_wait:
                        break
                    with st.spinner("Processing video, please wait..."):
                        time.sleep(1)
                    waited += 1

                if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                    st.subheader("🎬 Session Recording:")
                    
                    # Use Flask serving for better compatibility (same as analyze page)
                    video_url = f"http://localhost:5050/serve_video/{video_filename}"
                    
                    # Create HTML video element with error handling
                    video_html = f"""
                    <div id="live-video-container">
                        <video id="live-session-video" width="100%" height="auto" controls preload="metadata" crossorigin="anonymous">
                            <source src="{video_url}" type="video/mp4; codecs=&quot;mp4v.20.9,mp4a.40.2&quot;">
                            <source src="{video_url}" type="video/mp4">
                            Your browser does not support the video tag or the video codec.
                        </video>
                        <div id="live-video-status" style="margin-top: 10px; padding: 10px; border-radius: 4px;"></div>
                        <div id="live-retry-container" style="margin-top: 10px; display: none;">
                            <button onclick="retryLiveVideo()" style="padding: 8px 16px; background-color: #ff6b6b; color: white; border: none; border-radius: 4px; cursor: pointer;">
                                🔄 Retry Loading Video
                            </button>
                        </div>
                    </div>
                    
                    <script>
                        const liveVideo = document.getElementById('live-session-video');
                        const liveStatus = document.getElementById('live-video-status');
                        const liveRetryContainer = document.getElementById('live-retry-container');
                        
                        function updateLiveStatus(message, color, background) {{
                            liveStatus.style.display = 'block';
                            liveStatus.innerHTML = message;
                            liveStatus.style.color = color;
                            liveStatus.style.backgroundColor = background || 'transparent';
                        }}
                        
                        // Enhanced event listeners for live session video
                        liveVideo.addEventListener('loadstart', function() {{
                            updateLiveStatus('⏳ Loading session video...', '#666', '#f0f8ff');
                            liveRetryContainer.style.display = 'none';
                        }});
                        
                        liveVideo.addEventListener('loadedmetadata', function() {{
                            updateLiveStatus('📊 Session video metadata loaded...', '#666', '#f0f8ff');
                        }});
                        
                        liveVideo.addEventListener('canplay', function() {{
                            updateLiveStatus('✅ Session video ready to play!', 'white', '#28a745');
                            liveRetryContainer.style.display = 'none';
                        }});
                        
                        liveVideo.addEventListener('error', function(e) {{
                            const error = liveVideo.error;
                            let errorMessage = '❌ Session video playback error: ';
                            
                            if (error) {{
                                switch(error.code) {{
                                    case error.MEDIA_ERR_DECODE:
                                        errorMessage += 'Video decode error (codec not supported)';
                                        break;
                                    case error.MEDIA_ERR_NETWORK:
                                        errorMessage += 'Network error while loading video';
                                        break;
                                    case error.MEDIA_ERR_SRC_NOT_SUPPORTED:
                                        errorMessage += 'Video format/codec not supported by browser';
                                        break;
                                    default:
                                        errorMessage += 'Unknown error';
                                }}
                            }} else {{
                                errorMessage += 'Unknown playback error';
                            }}
                            
                            updateLiveStatus(errorMessage, 'white', '#dc3545');
                            liveRetryContainer.style.display = 'block';
                        }});
                        
                        function retryLiveVideo() {{
                            updateLiveStatus('🔄 Retrying session video load...', '#666', '#f0f8ff');
                            liveVideo.load();
                        }}
                        
                        // Test server connectivity for live session video
                        fetch('{video_url}', {{method: 'HEAD'}})
                            .then(response => {{
                                if (!response.ok) {{
                                    throw new Error(`Server responded with status: ${{response.status}}`);
                                }}
                                console.log('✅ Live session video server is accessible');
                            }})
                            .catch(error => {{
                                updateLiveStatus('⚠️ Cannot connect to video server. Video may not be ready yet.', 'white', '#fd7e14');
                                liveRetryContainer.style.display = 'block';
                                console.error('Live video server connectivity error:', error);
                            }});
                    </script>
                    
                    <p><em>Video URL: <a href="{video_url}" target="_blank">{video_url}</a></em></p>
                    """
                    
                    st.markdown(video_html, unsafe_allow_html=True)
                    
                    # Provide download option using Flask endpoint
                    download_url = f"http://localhost:5050/serve_video/{video_filename}?download=true"
                    st.markdown(f"""
                    <a href="{download_url}" style="
                        display: inline-block;
                        padding: 0.5rem 1rem;
                        background-color: #28a745;
                        color: white;
                        text-decoration: none;
                        border-radius: 0.25rem;
                        font-weight: 500;
                        margin-top: 10px;
                    ">📥 Download Session Video</a>
                    """, unsafe_allow_html=True)
                    
                elif os.path.exists(video_path) and os.path.getsize(video_path) == 0:
                    st.warning("⚠️ Session video file exists but is empty. The recording may have failed.")
                    st.info("💡 Try starting a new session or check your camera permissions.")
                else:
                    st.info("🔄 Session video is still processing or failed to save.")
                    st.info("📜 You can check your session history later to view the recording.")
                    
                    # Show helpful troubleshooting info
                    with st.expander("🛠️ Troubleshooting"):
                        st.write("**Possible reasons for missing video:**")
                        st.write("• Camera permissions not granted")
                        st.write("• Video processing still in progress")
                        st.write("• Insufficient disk space")
                        st.write("• Session was too short to record")
                        st.write("• Flask backend connection issues")

        if st.button("⬅️ Back to Home"):
            st.session_state.page = "Home"
            st.session_state["start_streaming"] = False
            for key in ["selected_exercise", "start_time", "stop_time", "reps_count"]:
                st.session_state.pop(key, None)
            st.rerun()

# Session History Page
elif st.session_state.page == "History":
    st.title("Session History")
    user_id = st.session_state.get("user_id")
    if not user_id:
        st.warning("You must be logged in to view your session history.")
    else:
        try:
            from src.database.session import Session
            sessions = Session.load_by_user(user_id)
            if not sessions:
                st.info("No sessions found.")
            else:
                for sess in sessions:
                    st.markdown(f"**Session ID:** {sess.session_id}  ")
                    st.markdown(f"**Start Time:** {format_datetime(sess.start_time)}  ")
                    st.markdown(f"**End Time:** {format_datetime(sess.end_time)}  ")
                    st.markdown(f"**Exercise ID:** {sess.exercise_id}  ")
                    st.markdown(f"**Planned Reps:** {sess.planned_reps}  ")
                    st.markdown(f"**Actual Reps:** {sess.actual_reps}  ")
                    st.markdown(f"**Duration (sec):** {sess.duration}  ")
                    st.markdown(f"**Status:** {sess.session_status}  ")
                    if sess.video_path and os.path.exists(sess.video_path):
                        st.video(sess.video_path)
                    else:
                        st.info("Video not available or still processing.")
                    st.markdown("---")
        except Exception as e:
            st.error(f"Error loading session history: {e}")
    st.button("⬅️ Back to Home", on_click=set_page, args=("Home",))
