#!/bin/bash
# ============================================================================
# 🎭 VISION LAB v3.2 - Quick Start Script
# ============================================================================
# Title:      Vision Lab Launcher
# Author:     ajax
# Date:       2026-01-20
# Version:    3.2.0
# License:    MIT
#
# Description:
#   One-click launcher for Vision Lab. Starts both the FastAPI backend
#   and React frontend development servers.
#
# Usage:
#   ./start.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🎭 Vision Lab v3.2"
echo "===================="

# Check if venv exists
if [ ! -d "$SCRIPT_DIR/backend/venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv "$SCRIPT_DIR/backend/venv"
fi

# Activate venv and install dependencies
source "$SCRIPT_DIR/backend/venv/bin/activate"

echo "📦 Installing backend dependencies..."
pip install -q -r "$SCRIPT_DIR/backend/requirements.txt"

# Check if node_modules exists
if [ ! -d "$SCRIPT_DIR/frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd "$SCRIPT_DIR/frontend"
    npm install
    cd "$SCRIPT_DIR"
fi

# Start backend in background
echo "🚀 Starting backend server on port 8080..."
cd "$SCRIPT_DIR/backend"
python main.py &
BACKEND_PID=$!

# Give backend time to start
sleep 2

# Start frontend
echo "🚀 Starting frontend server on port 5173..."
cd "$SCRIPT_DIR/frontend"
npm run dev -- --host &
FRONTEND_PID=$!

echo ""
echo "✅ Vision Lab is running!"
echo "   Frontend: http://localhost:5173"
echo "   Backend:  http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop both servers"

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for both processes
wait
