#!/bin/bash

echo "Setting up Movie Review Sentiment Analysis project..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Initialize models
echo "Initializing models (this may take a while on first run)..."
python -m src.models --size 0.5B
python -m src.models --size 1.5B

# Start the Streamlit UI
echo "Starting the UI..."
streamlit run main.py 