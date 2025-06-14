# Right Motion - Installation Guide

A comprehensive fitness tracking application using computer vision and MediaPipe for real-time exercise form analysis.

## 📋 Prerequisites

### All Platforms
- **Python 3.9-3.11** (recommended: Python 3.9)
- **PostgreSQL Database** (local or cloud)
- **Webcam/Camera** for live exercise tracking
- **Microphone** (optional, for audio feedback)

### Platform-Specific Requirements

#### Windows
- **Visual Studio Build Tools** or **Visual Studio Community** (for compiling packages)
- **Windows Camera permissions** enabled
- **PostgreSQL Server** installed or cloud database access

#### macOS
- **Xcode Command Line Tools**: `xcode-select --install`
- **Homebrew** (recommended): `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3-dev postgresql postgresql-contrib libpq-dev
sudo apt install build-essential cmake pkg-config
sudo apt install libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev
```

## 🚀 Installation Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Final_project
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

#### For Windows:
```bash
pip install -r requirements_windows.txt
```

#### For macOS/Linux:
```bash
pip install -r requirements.txt
```

#### For Minimal Installation (any platform):
```bash
pip install -r requirements_clean.txt
```

### 4. Database Setup

#### Option A: Local PostgreSQL
```bash
# Install PostgreSQL (if not already installed)
# Windows: Download from https://www.postgresql.org/download/windows/
# macOS: brew install postgresql
# Linux: sudo apt install postgresql postgresql-contrib

# Create database
createdb rightmotion_db

# Set environment variables
export DATABASE_URL="postgresql://username:password@localhost:5432/rightmotion_db"
```

#### Option B: Cloud Database (recommended)
Set your database URL in `.env` file:
```bash
DATABASE_URL="postgresql://username:password@host:port/database"
```

### 5. Environment Configuration
Create a `.env` file in the project root:
```bash
# Database
DATABASE_URL="your_database_url_here"

# Optional: Set protocol buffers implementation for Streamlit compatibility
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

### 6. Initialize Database
```bash
# Run database migrations
python src/database/migrations/run_comprehensive_migration.py

# Add seed data for exercises
python src/database/migrations/seed_exercises.py
```

## 🎯 Running the Application

### Web Interface (Streamlit)
```bash
streamlit run src/app/app.py
```
Access at: `http://localhost:8501`

### Command Line Interface
```bash
# Run live exercise session
python main.py --exercise lunge --user_id 1

# Run without GUI (headless)
python main.py --exercise lunge --user_id 1 --no-window

# Save keypoints data
python main.py --exercise lunge --user_id 1 --save-keypoints
```

### Video Analysis Server (Backend)
```bash
# Start video streaming server (runs automatically with Streamlit)
python -m src.app.video_streamer
```

## 🔧 Troubleshooting

### Common Issues

#### MediaPipe Installation Issues
```bash
# If MediaPipe fails to install:
pip install --no-cache-dir mediapipe

# Windows alternative:
pip install mediapipe-silicon
```

#### OpenCV Camera Issues
```bash
# Test camera access:
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera:', cap.isOpened()); cap.release()"

# Windows: Ensure camera permissions are enabled in Privacy settings
# macOS: Grant camera permissions when prompted
# Linux: Add user to video group: sudo usermod -a -G video $USER
```

#### PostgreSQL Connection Issues
```bash
# Test database connection:
python src/database/test_db.py

# Check connection string format:
# postgresql://username:password@host:port/database_name
```

#### Audio/TTS Issues
```bash
# Windows: Install additional audio dependencies if needed
pip install pywin32

# macOS: TTS should work out of the box
# Linux: Install espeak: sudo apt install espeak espeak-data
```

### Performance Optimization

#### For Low-End Hardware:
- Reduce MediaPipe model complexity in `main.py`:
  ```python
  pose = mp_pose.Pose(
      static_image_mode=False,
      model_complexity=0,  # Change from 1 to 0
      min_detection_confidence=0.3,  # Lower threshold
      min_tracking_confidence=0.3
  )
  ```

#### For Better Accuracy:
- Increase model complexity:
  ```python
  model_complexity=2,  # Highest accuracy
  min_detection_confidence=0.7,
  min_tracking_confidence=0.7
  ```

## 📊 Supported Exercises

- **Lunge**: Front and back leg detection with form analysis
- **Overhead Press**: Ready position detection and rep counting
- **Plank**: Duration tracking with posture validation

## 🛡️ Security Notes

- Never commit database credentials to version control
- Use environment variables for sensitive configuration
- Ensure camera permissions are properly configured
- Use HTTPS in production deployments

## 📖 Additional Documentation

- **CLAUDE.md**: Development guidelines and architecture overview
- **src/database/**: Database schema and migration scripts
- **tests/**: Comprehensive test suite with 43 tests
- **videos/**: Processed exercise videos (created during runtime)

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Test individual components (database, camera, etc.)
4. Check the logs for detailed error messages

For development questions, refer to the code documentation and CLAUDE.md file.