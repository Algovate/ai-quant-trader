#!/bin/bash
# Enhanced Stock Analysis UI with Portfolio Management

echo "🚀 Starting AI Trading System - Professional Dashboard..."
echo "📊 Loading Professional Trading Interface..."
echo ""
echo "🎯 Professional Features:"
echo "   📊 Market Data & Charts - Real-time price analysis with technical indicators"
echo "   🤖 AI Analysis & Predictions - LLM-powered quantitative analysis"
echo "   📈 Trading Signals & Strategy - AI-generated trading recommendations"
echo "   💼 Portfolio & Risk Management - Professional portfolio tracking and order management"
echo ""
echo "💼 Desktop Professional Interface:"
echo "   🖥️  Full-width layout optimized for desktop trading workstations"
echo "   📊 Professional metrics and real-time statistics"
echo "   🎯 Organized workflow for quantitative analysts"
echo "   🔧 Advanced risk management and portfolio controls"
echo "   📏 Maximized screen utilization for 1920x1080+ displays"
echo ""

cd "$(dirname "$0")"

# Check if results exist
if [ ! -d "data/results" ]; then
    echo "⚠️  No results found. Please run the analysis pipeline first:"
    echo "   uv run python scripts/trading_pipeline.py"
    echo ""
fi

# Check if portfolio exists
if [ ! -f "data/portfolio.json" ]; then
    echo "💡 No portfolio file found. Portfolio features will show empty state."
    echo "   Run the demo to generate portfolio data."
    echo ""
fi

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "Using uv to run the UI..."
    uv run python scripts/trading_dashboard.py
else
    echo "Using python directly..."
    python scripts/trading_dashboard.py
fi
