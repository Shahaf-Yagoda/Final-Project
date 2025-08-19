#!/bin/bash

# Right Motion - Application Starter Script

# Activate virtual environment
source venv/bin/activate

# Add PostgreSQL to PATH
export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"

# Set protocol buffers environment variable
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "🏃 Starting Right Motion Fitness App..."
echo "📱 Opening Streamlit web interface at http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the application"
echo ""

# Start Streamlit app
streamlit run src/app/app.py