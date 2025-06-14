#!/bin/bash

echo "🔧 Cleaning up camera sessions and services..."

# Kill Streamlit and video_streamer processes
echo "Stopping Streamlit and video streaming processes..."
pkill -f "streamlit run src/app/app.py"
pkill -f "src.app.video_streamer"
pkill -f "video_streamer.py"

# Wait a moment for graceful shutdown
sleep 2

# Force kill if still running
pids=$(ps aux | grep -E "(streamlit.*app\.py|video_streamer)" | grep -v grep | awk '{print $2}')
if [ ! -z "$pids" ]; then
    echo "Force killing remaining processes: $pids"
    kill -9 $pids 2>/dev/null
fi

# Clean up temporary session files
echo "Cleaning up temporary files..."
rm -f /tmp/user*_session.mp4
rm -f /tmp/reps_*.txt
rm -f /tmp/feedback_*.json
rm -f /tmp/sessiondetails_*.json

# Check for processes using port 5050
port_pids=$(lsof -ti :5050 2>/dev/null)
if [ ! -z "$port_pids" ]; then
    echo "Killing processes using port 5050: $port_pids"
    kill -9 $port_pids
fi

echo "✅ Camera cleanup complete!"
echo ""
echo "You can now restart the application:"
echo "  streamlit run src/app/app.py"