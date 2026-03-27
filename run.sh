#!/bin/bash
# run.sh — One-command launcher for ML Stock Predictor
# Usage: ./run.sh [TICKER]

set -e

TICKER=${1:-AAPL}
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║          ML STOCK PREDICTOR — LAUNCHER              ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  Ticker:  $TICKER"
echo "  Project: $PROJECT_DIR"
echo ""

cd "$PROJECT_DIR"

# Create virtual environment if missing
if [ ! -d "venv" ]; then
    echo "→ Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install/upgrade dependencies
echo "→ Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Create output dirs
mkdir -p output logs

# Run pipeline
echo ""
echo "→ Running ML pipeline for $TICKER..."
python pipeline.py --ticker "$TICKER"

# Launch dashboard
echo ""
echo "→ Launching Streamlit dashboard..."
echo "   Open http://localhost:8501 in your browser"
echo ""
streamlit run src/dashboard/app.py \
    --server.headless true \
    --server.port 8501 \
    --browser.gatherUsageStats false
