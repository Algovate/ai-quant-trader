# 🤖 AI Quant Trader

AI-powered quantitative trading system with intelligent market analysis and portfolio management.

## 📚 Documentation

### 📖 Documentation Index

- **[docs/INDEX.md](docs/INDEX.md)** - Complete documentation index and navigation
- **[docs/TESTING.md](docs/TESTING.md)** - Testing guide and documentation
- **[docs/GLOSSARY.md](docs/GLOSSARY.md)** - System terminology, core data and algorithm details
- **[docs/QUANTITATIVE_METHODS.md](docs/QUANTITATIVE_METHODS.md)** - Comprehensive quantitative trading methods overview
- **[docs/SUMMARY.md](docs/SUMMARY.md)** - Documentation system summary and quick navigation

## ✨ Core Features

- **🧠 LLM Integration**: Integrate Claude, GPT-4, Gemini and other advanced LLMs via OpenRouter API
- **📊 Intelligent Analysis**: AI provides detailed market analysis with confidence scoring
- **🔄 Data Processing**: Automated fetching, cleaning, and processing of historical stock data
- **📈 Technical Indicators**: Calculate 20+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **🎯 Signal Generation**: Generate trading signals based on AI predictions
- **💼 Portfolio Management**: Position tracking, risk assessment, and order generation
- **⚙️ Flexible Configuration**: YAML configuration files for easy customization
- **🌐 Web Dashboard**: Modern, interactive trading dashboard with real-time data visualization
- **📊 Multiple Trading Methods**: Support for traditional technical analysis, machine learning, statistical arbitrage, and more
- **🔧 Flexible Deployment**: Command-line options for local and public sharing

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy template
cp config/env_template.txt .env

# Edit .env file with your API keys
# OPENROUTER_API_KEY=your-api-key-here
```

### 3. Run the System

```bash
# Run complete trading pipeline
uv run python scripts/trading_pipeline.py

# Launch trading dashboard
uv run python scripts/trading_dashboard.py

# Quick test
uv run python scripts/trading_pipeline.py test
```

## 📦 Installation Details

### Dependencies

The system requires Python 3.12+ and the following key packages:

- **Data Processing**: pandas, numpy
- **Visualization**: plotly, matplotlib, seaborn
- **Web UI**: gradio
- **Machine Learning**: scikit-learn, scipy, xgboost
- **Data Sources**: yfinance, requests
- **Utilities**: click, python-dotenv, pyyaml
- **Testing**: pytest and related tools

### Environment Setup

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd ai-quant-trader
   ```

2. **Install dependencies**:
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp config/env_template.txt .env
   # Edit .env with your API keys
   ```

## 📁 Project Structure

```
ai-quant-trader/
├── src/                          # Source code
│   ├── core/                     # Core modules (constants, utilities)
│   ├── data/                     # Data processing (fetching, cleaning, feature engineering)
│   ├── models/                   # AI models (LLM predictor, model management)
│   └── strategy/                 # Trading strategies (signal generation, portfolio management)
├── config/                       # Configuration files
├── scripts/                      # Executable scripts
│   ├── trading_pipeline.py      # Trading pipeline script
│   └── trading_dashboard.py     # Trading dashboard
├── start_dashboard.sh           # Dashboard startup script
├── data/                         # Data storage
│   ├── cache/                   # Cached data
│   ├── results/                 # Intermediate results (for verification)
│   └── portfolio.json           # Portfolio state
├── tests/                        # Test modules
├── docs/                         # Documentation
│   ├── INDEX.md                  # Documentation index
│   ├── TESTING.md                # Testing guide
│   ├── GLOSSARY.md               # System terminology
│   ├── QUANTITATIVE_METHODS.md   # Trading methods overview
│   └── SUMMARY.md                # Documentation summary
├── requirements.txt              # Python dependencies
└── pyproject.toml               # Project configuration
```

## 🤖 Supported LLM Models

| Model | Features | Use Cases |
|-------|----------|-----------|
| Claude 3.5 Sonnet | Advanced reasoning | Complex market analysis |
| GPT-4o | Latest capabilities | General analysis |
| GPT-4o Mini | Fast and economical | High-frequency predictions |
| DeepSeek Chat V3 | Balanced performance | Standard analysis |

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
# Data settings
data:
  symbols: ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
  start_date: "2023-01-01"
  end_date: "2024-01-31"

# LLM model settings
model:
  default_model: "deepseek/deepseek-chat-v3-0324"
  min_confidence: 0.6

# Trading strategy
strategy:
  position_size: 0.1
  stop_loss: 0.05
  take_profit: 0.15

# Portfolio management
portfolio:
  initial_cash: 100000
  max_position_size: 0.15  # Maximum 15% per position
  max_total_positions: 10
  min_trade_size: 1000    # Minimum trade amount $1000
  risk_per_trade: 0.02    # 2% risk per trade
```

## 🖥️ Command Line Options

### Trading Dashboard Options

The dashboard supports flexible command-line configuration:

```bash
# Basic usage
uv run python scripts/trading_dashboard.py

# Available options
--share          # Enable public sharing (creates public URL)
--port PORT      # Set port number (default: 7860)
--host HOST      # Set host address (default: 0.0.0.0)
--no-browser     # Don't automatically open browser
--help           # Show help message
```

### Examples

```bash
# Local access only
uv run python scripts/trading_dashboard.py

# Public sharing for remote access
uv run python scripts/trading_dashboard.py --share

# Custom configuration
uv run python scripts/trading_dashboard.py --port 8080 --host 127.0.0.1

# Production deployment
uv run python scripts/trading_dashboard.py --share --port 80 --no-browser
```

## 💡 Usage Examples

### Run Complete Demo
```bash
# Run complete demo
uv run python scripts/trading_pipeline.py

# Quick test
uv run python scripts/trading_pipeline.py test
```

### Launch Web Dashboard
```bash
# Start professional trading dashboard (local access)
uv run python scripts/trading_dashboard.py

# Start with public sharing (accessible via public URL)
uv run python scripts/trading_dashboard.py --share

# Custom port and host
uv run python scripts/trading_dashboard.py --port 8080 --host 0.0.0.0

# Public sharing with custom port
uv run python scripts/trading_dashboard.py --share --port 8080

# Start without opening browser
uv run python scripts/trading_dashboard.py --no-browser

# View all available options
uv run python scripts/trading_dashboard.py --help
```

### Portfolio Management
```bash
# Run complete demo (including portfolio management)
uv run python scripts/trading_pipeline.py

# View portfolio status
cat data/portfolio.json

# View generated orders
ls data/results/step6_orders_*.json
```

### Programming Interface
```python
from src.models.llm_predictor import create_llm_predictor
from src.data.data_fetcher import DataFetcher
from src.strategy.portfolio_manager import Portfolio, PortfolioManager
from src.strategy.order_generator import create_order_generator

# Initialize components
llm_predictor = create_llm_predictor()
data_fetcher = DataFetcher()
portfolio = Portfolio(initial_cash=100000)
portfolio_manager = PortfolioManager(portfolio)

# Fetch data and predictions
data = data_fetcher.fetch_data(["AAPL"], "2024-01-01", "2024-12-31")
predictions = llm_predictor.predict(data)

# Generate trading signals
from src.strategy.llm_signal_generator import create_llm_signal_generator
signal_generator = create_llm_signal_generator(llm_predictor)
signals = signal_generator.generate_signals_from_predictions(predictions)

# Generate portfolio orders
order_generator = create_order_generator(portfolio_manager, {
    'max_position_size': 0.15,
    'min_trade_size': 1000,
    'risk_per_trade': 0.02
})
orders = order_generator.generate_orders(signals, data)

# View results
for symbol, pred in predictions.items():
    print(f"{symbol}: {pred['prediction']:.2%} (confidence: {pred['confidence']:.2%})")

for order in orders:
    print(f"Order: {order['action']} {order['symbol']} {order['quantity']} shares")
```

## 🔧 Extending the System

- **Add new models**: Configure in `config/config.yaml`
- **Custom indicators**: Modify `src/data/technical_indicators.py`
- **Custom strategies**: Extend `src/strategy/llm_signal_generator.py`
- **Portfolio management**: Modify `src/strategy/portfolio_manager.py` and `src/strategy/order_generator.py`
- **Risk control**: Adjust portfolio parameters in `config/config.yaml`
- **Alternative methods**: Implement traditional technical analysis, machine learning, or statistical arbitrage strategies
- **Dashboard customization**: Modify `scripts/trading_dashboard.py` for custom UI components

## 📊 Quantitative Trading Methods

### Beyond LLM: Comprehensive Trading Strategies

The system supports multiple quantitative trading approaches beyond LLM predictions:

#### 🎯 Traditional Technical Analysis
- **Trend Following**: Moving averages, MACD, Bollinger Bands
- **Mean Reversion**: RSI, Stochastic Oscillator, Williams %R
- **Volume Analysis**: Volume-price trend, on-balance volume

#### 🤖 Machine Learning Approaches
- **Supervised Learning**: Random Forest, SVM, Neural Networks
- **Deep Learning**: LSTM, CNN for time series prediction
- **Unsupervised Learning**: K-means clustering for market regimes

#### 📈 Statistical Arbitrage
- **Pairs Trading**: Cointegration-based strategies
- **Mean Reversion**: Price deviation strategies
- **Cross-Asset Arbitrage**: Multi-asset correlation strategies

#### 🏗️ Factor Investing
- **Value Factors**: P/E, P/B, P/S ratios
- **Momentum Factors**: Price momentum, earnings momentum
- **Quality Factors**: ROE, ROA, debt-to-equity ratios

#### ⚡ High-Frequency Trading
- **Microstructure Strategies**: Order flow imbalance
- **Short-term Momentum**: Tick-level price movements
- **Spread Arbitrage**: Bid-ask spread opportunities

#### 🛡️ Risk Management
- **Dynamic Hedging**: Delta hedging strategies
- **VaR Calculation**: Historical and Monte Carlo methods
- **Stress Testing**: Scenario analysis and backtesting

For detailed implementation examples and code, see **[docs/QUANTITATIVE_METHODS.md](docs/QUANTITATIVE_METHODS.md)**.

## 💼 Portfolio Management Features

### Core Features

- **📊 Position Tracking**: Real-time monitoring of cash, positions, and P&L
- **🎯 Smart Orders**: Generate buy/sell orders based on LLM signals and risk assessment
- **⚖️ Risk Control**: Dynamic position management to prevent over-concentration
- **💾 State Persistence**: Save portfolio state in JSON format
- **📈 Performance Analysis**: Calculate unrealized P&L and portfolio value

### Workflow

1. **Data Fetching** → 2. **Feature Engineering** → 3. **LLM Prediction** → 4. **Signal Generation** → 5. **Portfolio Analysis** → 6. **Order Generation**

### Order Types

- **🟢 Buy Orders**: Based on bullish signals and available cash
- **🔴 Sell Orders**: Based on bearish signals and existing positions
- **🟡 Hold**: Insufficient signal strength or high risk

### Risk Control Parameters

```yaml
portfolio:
  max_position_size: 0.15    # Maximum 15% per position
  min_trade_size: 1000       # Minimum trade amount
  risk_per_trade: 0.02       # 2% risk per trade
  max_total_positions: 10    # Maximum number of positions
```

## ⚠️ Disclaimer

This system is for educational and research purposes only. Past performance does not guarantee future results. Trading involves risk, please make informed decisions.

---

**Happy Trading!** 🚀