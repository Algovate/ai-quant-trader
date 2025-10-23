#!/bin/bash
# Simple launcher for the Stock Analysis UI

echo "🚀 Starting Stock Analysis Results Viewer..."
echo "📊 Loading Gradio interface..."

cd "$(dirname "$0")"

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "Using uv to run the UI..."
    uv run python scripts/stock_ui.py
else
    echo "Using python directly..."
    python scripts/stock_ui.py
fi
