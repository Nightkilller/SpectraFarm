#!/bin/bash
# SpectraFarm — Quick Setup & Run Script
# Run this after moving/cloning the project to a new location.
#
# Usage:
#   chmod +x setup_and_run.sh
#   ./setup_and_run.sh

set -e

echo ""
echo "=========================================="
echo "  SpectraFarm — AgriN Setup & Run"
echo "=========================================="
echo ""

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 is not installed. Install Python 3.11+ first."
    exit 1
fi
echo "[OK] Python found: $(python3 --version)"

# 2. Create venv if it doesn't exist or is broken
if [ ! -f "venv/bin/activate" ]; then
    echo "[..] Creating virtual environment..."
    python3 -m venv venv
    echo "[OK] Virtual environment created."
else
    echo "[OK] Virtual environment exists."
fi

# 3. Activate venv
source venv/bin/activate
echo "[OK] Virtual environment activated."

# 4. Install dependencies
echo "[..] Installing dependencies (this may take a minute)..."
pip install -r requirements.txt --quiet
echo "[OK] All dependencies installed."

# 5. Check .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "[WARNING] No .env file found!"
    echo "  Copy .env.example to .env and add your API keys:"
    echo ""
    echo "    cp .env.example .env"
    echo "    # Then edit .env and add:"
    echo "    #   GEE_PROJECT=your-gee-project-id"
    echo "    #   GROQ_API_KEY=your-groq-key"
    echo "    #   GEMINI_API_KEY=your-gemini-key"
    echo ""
    echo "  The app will still run in DEMO mode without keys."
    echo ""
fi

# 6. Run the app
echo ""
echo "=========================================="
echo "  Starting SpectraFarm Dashboard..."
echo "  Open http://localhost:8501 in your browser"
echo "  Press Ctrl+C to stop"
echo "=========================================="
echo ""

streamlit run app.py
