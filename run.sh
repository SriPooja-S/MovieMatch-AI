#!/usr/bin/env bash
# MovieMatch AI — launch script

set -e
cd "$(dirname "$0")"

echo ""
echo "╔═══════════════════════════════════════╗"
echo "║      MovieMatch AI — Starting Up      ║"
echo "╚═══════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "Error: Python 3 not found. Please install Python 3.10+."
  exit 1
fi

# Create venv if missing
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
  source venv/bin/activate
  pip install --quiet --upgrade pip
  echo "Installing dependencies (first run, may take a few minutes)..."
  pip install --quiet -r requirements.txt
else
  source venv/bin/activate
fi

# Check for data
if [ ! -f "app/data/movies.json" ]; then
  echo "Warning: No movie database found."
  echo "    Run this to fetch movies: python3 -c \"from app.ingestion.ingestion_manager import IngestionManager; IngestionManager().incremental_update(days=30)\""
fi

echo ""
echo "Launching MovieMatch AI at http://localhost:8501"
echo ""

streamlit run cinematch_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.maxUploadSize 200 \
  --browser.gatherUsageStats false
