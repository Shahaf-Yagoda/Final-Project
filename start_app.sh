#!/bin/bash

# Right Motion Fitness App Startup Script
# Starts all necessary components for the application

set -e  # Exit on any error

echo "🚀 Starting Right Motion Fitness App"
echo "===================================="
echo

# Check if we're in the right directory
if [[ ! -f "src/app/app.py" ]]; then
    echo "❌ src/app/app.py not found. Make sure you're in the project root directory."
    exit 1
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found. Please install Python 3."
    exit 1
fi

# Check if .env file exists
if [[ ! -f ".env" ]]; then
    echo "❌ .env file not found. Please run setup_db.sh first or create .env manually."
    exit 1
fi

# Activate virtual environment if it exists
if [[ -d "venv" ]]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Set required environment variables
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "🔧 Setting up environment..."
echo "   Protocol Buffers: $PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"
echo

# Test database connection
echo "🗄️  Testing database connection..."
python3 -c "
import sys
sys.path.append('src')
try:
    from database.database_connection import get_connection
    conn = get_connection()
    if conn:
        print('✅ Database connection successful')
        conn.close()
    else:
        print('❌ Database connection failed')
        exit(1)
except Exception as e:
    print(f'❌ Database connection error: {e}')
    exit(1)
" || {
    echo "❌ Database connection failed. Please check your .env configuration."
    exit 1
}

echo
echo "🎬 Starting video streaming server..."
echo "   Server will start on http://localhost:5050"
echo

# Start the video streaming server in the background
python3 -c "
import sys
sys.path.append('src/app')
from video_streamer import app
app.run(host='0.0.0.0', port=5050, debug=False)
" &

# Store the PID of the video server
VIDEO_SERVER_PID=$!

# Give the video server time to start
sleep 3

echo
echo "🌐 Starting Streamlit application..."
echo "   App will open at http://localhost:8501"
echo

# Function to cleanup background processes on exit
cleanup() {
    echo
    echo "🛑 Shutting down services..."
    if kill -0 $VIDEO_SERVER_PID 2>/dev/null; then
        echo "   Stopping video server (PID: $VIDEO_SERVER_PID)..."
        kill $VIDEO_SERVER_PID
    fi
    echo "✅ Shutdown complete"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM EXIT

# Start Streamlit application
streamlit run src/app/app.py --server.port=8501 --server.address=0.0.0.0

# This line should not be reached, but just in case
cleanup