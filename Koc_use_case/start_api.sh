#!/bin/bash
# Start the Flask API server for score computation

cd "$(dirname "$0")"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if Flask and Gurobi are installed
python3 -c "import flask, gurobipy" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install flask gurobipy
}

echo "Starting API server on port 5000..."
python3 api.py
