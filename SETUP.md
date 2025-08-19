# Right Motion - Setup Guide for New Computers

This guide helps you set up the Right Motion fitness tracking application on a new computer.

## Prerequisites

### 1. Install Python 3.9+
```bash
# macOS (with Homebrew)
brew install python@3.9

# Ubuntu/Debian
sudo apt update && sudo apt install python3.9 python3.9-venv python3.9-pip

# Windows
# Download from https://python.org and install
```

### 2. Install PostgreSQL
```bash
# macOS (with Homebrew)
brew install postgresql@16
brew services start postgresql@16

# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql

# Windows
# Download from https://www.postgresql.org/download/windows/
```

### 3. Create PostgreSQL Database
```bash
# Create database user (replace 'yourusername' with your actual username)
createuser --interactive yourusername
# Answer: y (superuser), y (create databases), y (create roles)

# Create the database
createdb rightmotion_db
```

## Project Setup

### 1. Clone and Navigate
```bash
git clone <your-repo-url>
cd Final-Project
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Make sure virtual environment is activated first!
pip install -r requirements.txt

# If you get SSL warnings, also install:
pip install --upgrade urllib3
```

### 4. Environment Configuration
```bash
# Copy and edit environment file
cp .env.example .env  # If .env.example exists, otherwise create new .env

# Edit .env file with your settings:
# USE_CLOUD_DB=false
# DB_NAME_LOCAL=rightmotion_db
# DB_USER_LOCAL=yourusername
# DB_PASSWORD_LOCAL=yourpassword  # or leave empty if no password
# DB_ADDRESS_LOCAL=localhost
```

### 5. Database Setup
```bash
# Set up database schema
python3 setup_database.py

# OR create tables manually:
python3 create_db_schema.py
```

### 6. Test Database Connection
```bash
python3 src/database/test_db.py
```

## Running the Application

### Option 1: Use Startup Script (Recommended)
```bash
# Make script executable
chmod +x start_app.sh

# Run the application
./start_app.sh
```

### Option 2: Manual Start
```bash
# Activate virtual environment
source venv/bin/activate

# Set environment variable for MediaPipe compatibility
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Start the application
streamlit run src/app/app.py
```

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError: No module named 'flask_cors'"**
   ```bash
   # Make sure virtual environment is activated
   source venv/bin/activate
   pip install flask-cors playsound
   ```

2. **Database Connection Errors**
   ```bash
   # Check PostgreSQL is running
   brew services list | grep postgresql  # macOS
   sudo systemctl status postgresql      # Linux
   
   # Test connection manually
   psql -h localhost -U yourusername -d rightmotion_db
   ```

3. **Permission Errors**
   ```bash
   # Make sure you have write permissions
   chmod 755 start_app.sh
   chmod -R 755 videos/  # If videos directory exists
   ```

4. **Python Version Issues**
   ```bash
   # Check Python version
   python3 --version
   # Should be 3.9 or higher
   
   # If using wrong Python, recreate virtual environment
   rm -rf venv
   python3.9 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

5. **Windows Video Codec Issues**
   ```bash
   # If you see "Failed to initialize video writer" on Windows:
   # The app will automatically fallback to NO-VIDEO mode
   # Session will work but videos won't be saved
   
   # To fix video recording on Windows:
   # Option 1: Install K-Lite Codec Pack
   # Download from: https://codecguide.com/download_kl.htm
   
   # Option 2: Install VLC Media Player (includes codecs)
   # Download from: https://www.videolan.org/vlc/
   
   # Option 3: Install Windows Media Feature Pack (Windows N/KN editions)
   # Run in PowerShell as Administrator:
   # Get-WindowsCapability -Online | Where-Object Name -like "*Media*"
   # Add-WindowsCapability -Online -Name "Media.MediaFeaturePack~~~~0.0.1.0"
   ```

### Environment Variables

Create a `.env` file with these settings:

```bash
# Database Configuration
USE_CLOUD_DB=false
DB_NAME_LOCAL=rightmotion_db
DB_USER_LOCAL=yourusername
DB_PASSWORD_LOCAL=
DB_ADDRESS_LOCAL=localhost

# Protocol buffers for Streamlit compatibility
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

## Development Notes

- **Always use the virtual environment** when running the application
- **Database tables use lowercase names**: `users`, `sessions`, `exercises`, etc.
- **Video files are saved** to `videos/` directory during sessions
- **Temporary files** are created in `/tmp/` during processing

## Testing

```bash
# Run all tests
python3 tests/run_tests.py

# Test specific components
python3 tests/run_tests.py test_forms_check
python3 tests/run_tests.py test_video_streamer
```

## Accessing the Application

Once running, access the application at:
- **Local**: http://localhost:8501
- **Network**: Check terminal output for network URL

## Additional Tools

### PostgreSQL Administration
```bash
# Install pgAdmin (optional)
brew install --cask pgadmin4

# OR use command line
psql -h localhost -U yourusername -d rightmotion_db
```

### Performance Monitoring
```bash
# Install watchdog for better performance
pip install watchdog
```

---

## Quick Start Checklist

- [ ] Install Python 3.9+
- [ ] Install PostgreSQL
- [ ] Create database and user
- [ ] Clone repository
- [ ] Create virtual environment
- [ ] Install requirements
- [ ] Configure .env file
- [ ] Set up database schema
- [ ] Test database connection
- [ ] Run application

**Need help?** Check the troubleshooting section or run `./start_app.sh` for the easiest setup.