@echo off
REM Right Motion - Application Starter Script for Windows

echo 🏃 Starting Right Motion Fitness App...
echo 📱 Opening Streamlit web interface at http://localhost:8501
echo ⏹️  Press Ctrl+C to stop the application
echo.

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo ❌ Virtual environment not found! Please run: python -m venv venv
    echo Then install requirements: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Set protocol buffers environment variable
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

REM Start Streamlit app
streamlit run src/app/app.py

pause