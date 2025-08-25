#!/bin/bash

# Database Setup Script for Right Motion Fitness App
# This script creates the database schema according to your specifications

set -e  # Exit on any error

echo "🏗️  Right Motion Database Setup"
echo "================================"
echo

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found. Please install Python 3."
    exit 1
fi

# Check if we're in the right directory
if [[ ! -f "create_db_schema.py" ]]; then
    echo "❌ create_db_schema.py not found. Make sure you're in the project root directory."
    exit 1
fi

# Check if .env file exists
if [[ ! -f ".env" ]]; then
    if [[ -f ".env.example" ]]; then
        echo "📋 No .env file found. Copying from .env.example..."
        cp .env.example .env
        echo "⚠️  Please edit .env file with your database credentials before continuing."
        echo "   Press Enter after you've configured your database settings..."
        read
    else
        echo "❌ No .env file found and no .env.example to copy from."
        echo "   Please create a .env file with your database configuration."
        exit 1
    fi
fi

echo "🔧 Setting up database schema..."
echo

# Ask user if they want to drop existing tables
read -p "⚠️  Do you want to drop existing tables? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  Dropping existing tables and recreating schema..."
    python3 create_db_schema.py --drop
else
    echo "📝 Creating new schema (existing tables will be preserved)..."
    python3 create_db_schema.py
fi

echo
echo "✅ Database setup complete!"
echo
echo "🧪 Running verification tests..."
echo

# Run a quick test to verify everything works
python3 -c "
import sys
sys.path.append('src')
from database.database_connection import get_connection
from database.db_utils import get_exercise_id_by_name

print('Testing database connection...')
conn = get_connection()
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM exercise')
exercise_count = cursor.fetchone()[0]
cursor.close()
conn.close()

print(f'✅ Found {exercise_count} exercises in database')

for exercise in ['lunge', 'overhead_press', 'plank']:
    try:
        ex_id = get_exercise_id_by_name(exercise)
        print(f'✅ {exercise}: ID = {ex_id}')
    except Exception as e:
        print(f'❌ {exercise}: {e}')
        
print()
print('🎉 Database setup verification complete!')
"

echo
echo "🚀 Your database is ready to use!"
echo
echo "Next steps:"
echo "  1. Run the application: streamlit run src/app/app.py"
echo "  2. Or run a test session: python3 main.py --exercise lunge --user_id 1"
echo