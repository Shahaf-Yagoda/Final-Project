# main.py (refactored)
import cv2
import mediapipe as mp
import numpy as np
import time
import json
from datetime import datetime
import os
import shutil
from dotenv import load_dotenv
from src.processing.forms_check import check_form, init_state
from src.database.database_connection import get_connection
from src.database.session import Session

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
from src.database.db_utils import (
    save_session_to_db,
    save_session_details_to_db,
    save_system_feedback_to_db,
    get_exercise_id_by_name,
)


class ExerciseSession:
    """Manages a complete exercise session with pose detection and form analysis."""
    
    def __init__(self, exercise_name, user_id, show_window=True, save_keypoints=False, save_video=True):
        self.exercise_name = exercise_name
        self.user_id = user_id
        self.show_window = show_window
        self.save_keypoints = save_keypoints
        self.save_video = save_video
        self.state = {
            "exercise": {
                exercise_name: init_state(exercise_name)
            }
        }
        self.pose_detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.start_time = None
        self.end_time = None
        self.video_writer = None
        self.temp_video_path = None
        self.final_video_path = None
        self.session_details = []
        self.feedback_data = []
    
    def setup_video_recording(self, cap):
        """Setup video recording for the session."""
        if not self.save_video:
            return None
            
        self.temp_video_path = f"/tmp/user{self.user_id}_{self.exercise_name}_session.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.video_writer = cv2.VideoWriter(self.temp_video_path, fourcc, fps, (width, height))
        print(f"📹 Video recording started: {self.temp_video_path}")
        return self.video_writer
    
    def finalize_video_recording(self):
        """Finalize video recording and move to permanent location."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            
        if self.temp_video_path and os.path.exists(self.temp_video_path):
            # Create videos directory
            videos_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "videos"))
            os.makedirs(videos_dir, exist_ok=True)
            
            # Generate final video filename with timestamp
            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            final_video_name = f"user{self.user_id}_{self.exercise_name}_session_{timestamp}.mp4"
            self.final_video_path = os.path.join(videos_dir, final_video_name)
            
            # Move video to final location
            shutil.move(self.temp_video_path, self.final_video_path)
            print(f"💾 Video saved: {self.final_video_path}")
            return self.final_video_path
        return None
    
    def save_session_detail(self, landmarks, feedback, reps, current_time):
        """Save session detail for database storage."""
        detail = {
            "timestamp": current_time,
            "rep_num": reps,
            "keypoints_json": [dict(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility) for lm in landmarks],
            "features_json": {},
            "is_correct": len(feedback) == 0,
            "incorrect_duration": 0
        }
        self.session_details.append(detail)
        
        # Save feedback
        for msg in feedback:
            self.feedback_data.append({"timestamp": current_time, "message": msg})
    
    def save_keypoints_to_db(self, keypoints_data, workout_id):
        """Save keypoints data to database (placeholder implementation)."""
        pass
    
    def process_frame(self, frame, landmarks):
        """Process a single frame with pose landmarks."""
        exercise_state = self.state["exercise"][self.exercise_name]
        feedback, reps_count = check_form(self.exercise_name, frame, landmarks, exercise_state)
        
        # Update rep count (accumulated total)
        current_count = exercise_state.get("count", 0)
        if reps_count > 0:
            exercise_state["count"] = current_count + reps_count
        
        # Save session data for database
        current_time = time.time()
        self.save_session_detail(landmarks, feedback, exercise_state["count"], current_time)
        
        if self.save_keypoints:
            keypoints_data = [{"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility} for lm in landmarks]
            self.save_keypoints_to_db(keypoints_data, workout_id=1)
        
        return feedback, reps_count
    
    def run(self):
        """Execute the complete exercise session with video recording and database saving."""
        print(f"🎬 Starting session | User: {self.user_id}, Exercise: {self.exercise_name}")
        print(f"💡 Press ESC to stop and save session")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return None
            
        self.start_time = datetime.now()
        
        # Setup video recording
        self.setup_video_recording(cap)
        
        session_interrupted = False
        
        try:
            with self.pose_detector as pose:
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break

                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(image_rgb)

                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        self.process_frame(frame, landmarks)
                    
                    # Write frame to video if recording
                    if self.video_writer:
                        self.video_writer.write(frame)

                    if self.show_window:
                        cv2.imshow("Pose Tracker - Press ESC to stop and save", frame)
                        key = cv2.waitKey(5) & 0xFF
                        if key == 27:  # ESC key
                            print("\n🛑 ESC pressed - stopping and saving session...")
                            break
                        elif key == ord('q'):  # Also allow 'q' to quit
                            print("\n🛑 'Q' pressed - stopping and saving session...")
                            break
        except KeyboardInterrupt:
            print("\n🛑 Ctrl+C pressed - stopping and saving session...")
            session_interrupted = True
        finally:
            # Always cleanup and save, regardless of how session ended
            cap.release()
            self.end_time = datetime.now()
            if self.show_window:
                cv2.destroyAllWindows()
            
            # Finalize video recording
            self.finalize_video_recording()
            
            # Save to database
            exercise_state = self.state["exercise"][self.exercise_name]
            final_reps = exercise_state.get("count", 0)
            
            print(f"🏁 Session completed. Total reps: {final_reps}")
            
            # Create and save session to database
            self.save_session_to_database(final_reps)
            
            return {
                "user_id": self.user_id,
                "exercise": self.exercise_name,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "reps": final_reps,
                "video_path": self.final_video_path,
                "interrupted": session_interrupted
            }
    
    def save_session_to_database(self, reps_count):
        """Save complete session data to database."""
        try:
            # Get exercise ID
            exercise_id = get_exercise_id_by_name(self.exercise_name)
            duration_sec = int((self.end_time - self.start_time).total_seconds())
            
            # Create and save session using Session class
            session = Session(
                user_id=self.user_id,
                exercise_id=exercise_id,
                start_time=self.start_time,
                end_time=self.end_time,
                duration_sec=duration_sec,
                video_path=self.final_video_path or "",
                reps_count=reps_count
            )
            session.save()
            session_id = session.session_id
            
            print(f"💾 Session saved to database (ID: {session_id})")
            
            # Save session details
            if self.session_details:
                print(f"💾 Saving {len(self.session_details)} session details...")
                for detail in self.session_details:
                    save_session_details_to_db(
                        session_id=session_id,
                        timestamp=detail["timestamp"],
                        rep_num=detail["rep_num"],
                        keypoints_json=json.dumps(detail["keypoints_json"]),
                        features_json=json.dumps(detail["features_json"]),
                        is_correct=detail["is_correct"],
                        incorrect_duration=detail["incorrect_duration"]
                    )
            
            # Save feedback data
            if self.feedback_data:
                print(f"💾 Saving {len(self.feedback_data)} feedback entries...")
                for feedback in self.feedback_data:
                    save_system_feedback_to_db(
                        session_id=session_id,
                        timestamp=feedback["timestamp"],
                        feedback_type="form_correction",
                        message=feedback["message"]
                    )
            
            print(f"✅ Complete session data saved successfully!")
            
        except Exception as e:
            print(f"❌ Error saving session to database: {e}")
            import traceback
            traceback.print_exc()


def run_session(exercise_name, user_id, show_window=True, save_keypoints=False, save_video=True):
    """Backward compatibility function that uses the new ExerciseSession class."""
    session = ExerciseSession(exercise_name, user_id, show_window, save_keypoints, save_video)
    return session.run()

def main():
    """Main function to run exercise session from command line."""
    import argparse
    parser = argparse.ArgumentParser(description="Run an exercise session with pose analysis and automatic saving")
    parser.add_argument("--exercise", required=True, help="Exercise type (lunge, press, plank)")
    parser.add_argument("--user_id", type=int, required=True, help="User ID")
    parser.add_argument("--no-window", action="store_true", help="Run without display window")
    parser.add_argument("--save-keypoints", action="store_true", help="Save keypoints to database")
    parser.add_argument("--no-video", action="store_true", help="Disable video recording")
    args = parser.parse_args()

    session = ExerciseSession(
        exercise_name=args.exercise,
        user_id=args.user_id,
        show_window=not args.no_window,
        save_keypoints=args.save_keypoints,
        save_video=not args.no_video
    )
    
    session_data = session.run()
    
    if session_data:
        print(f"🎉 Session completed successfully!")
        if session_data.get("video_path"):
            print(f"📹 Video saved: {session_data['video_path']}")
        print(f"📊 Final stats: {session_data['reps']} reps in {(session_data['end_time'] - session_data['start_time']).total_seconds():.1f} seconds")
    else:
        print(f"❌ Session failed to complete properly")


if __name__ == "__main__":
    main()

