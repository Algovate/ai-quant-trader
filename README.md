# 🤖 LLM-Powered AI Trading System

AI驱动的量化交易系统，使用大语言模型进行智能市场分析和交易信号生成。

## 📚 文档导航

### 📖 文档目录
- **[docs/INDEX.md](docs/INDEX.md)** - 完整文档索引和导航
- **[docs/TESTING.md](docs/TESTING.md)** - 测试指南和文档
- **[docs/GLOSSARY.md](docs/GLOSSARY.md)** - 系统术语、核心数据和算法详解
- **[docs/SUMMARY.md](docs/SUMMARY.md)** - 文档体系总结和快速导航

## ✨ 核心功能

- **🧠 LLM集成**: 通过OpenRouter API集成Claude、GPT-4、Gemini等先进LLM
- **📊 智能分析**: AI提供详细的市场分析和置信度评分
- **🔄 数据处理**: 自动获取、清理和处理历史股票数据
- **📈 技术指标**: 计算20+技术指标（RSI、MACD、布林带等）
- **🎯 信号生成**: 基于AI预测生成交易信号
- **💼 投资组合管理**: 持仓跟踪、风险评估、订单生成
- **⚙️ 灵活配置**: YAML配置文件，易于自定义

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install uv
uv init
uv add pyyaml pandas numpy matplotlib seaborn scipy yfinance scikit-learn xgboost click requests python-dotenv
```

### 2. 配置API密钥

```bash
# 复制模板
cp config/env_template.txt .env

# 编辑.env文件，添加你的API密钥
# OPENROUTER_API_KEY=your-api-key-here
```

### 3. 运行系统

```bash
# 运行完整演示
uv run python scripts/trading_pipeline.py

# 快速测试
uv run python scripts/trading_pipeline.py test
```

## 📁 项目结构

```
stock/
├── src/                          # 源代码
│   ├── core/                     # 核心模块（常量、工具）
│   ├── data/                     # 数据处理（获取、清理、特征工程）
│   ├── models/                   # AI模型（LLM预测器、模型管理）
│   └── strategy/                 # 交易策略（信号生成、投资组合管理）
├── config/                       # 配置文件
├── scripts/                      # 可执行脚本
│   ├── trading_pipeline.py      # 交易流水线脚本
│   ├── trading_dashboard.py     # 交易仪表板
│   └── validate_docs.py         # 文档验证工具
├── start_dashboard.sh           # 仪表板启动脚本
├── data/                         # 数据存储
│   ├── cache/                   # 缓存数据
│   ├── results/                 # 中间结果（用于核查）
│   └── portfolio.json           # 投资组合状态
└── tests/                        # 测试模块
```

## 🤖 支持的LLM模型

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| Claude 3.5 Sonnet | 高级推理能力 | 复杂市场分析 |
| GPT-4o | 最新能力 | 通用分析 |
| GPT-4o Mini | 快速、经济 | 高频预测 |
| DeepSeek Chat V3 | 平衡性能 | 标准分析 |

## ⚙️ 配置

编辑 `config/config.yaml` 自定义：

```yaml
# 数据设置
data:
  symbols: ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
  start_date: "2023-01-01"
  end_date: "2024-01-31"

# LLM模型设置
model:
  default_model: "deepseek/deepseek-chat-v3-0324"
  min_confidence: 0.6

# 交易策略
strategy:
  position_size: 0.1
  stop_loss: 0.05
  take_profit: 0.15

# 投资组合管理
portfolio:
  initial_cash: 100000
  max_position_size: 0.15  # 每个仓位最大15%
  max_total_positions: 10
  min_trade_size: 1000    # 最小交易金额$1000
  risk_per_trade: 0.02    # 每笔交易风险2%
```

## 💡 使用示例

### 运行完整演示
```bash
# 运行完整演示
uv run python scripts/trading_pipeline.py

# 快速测试
uv run python scripts/trading_pipeline.py test
```

### 启动Web仪表板
```bash
# 启动专业交易仪表板
./start_dashboard.sh

# 或者直接运行
uv run python scripts/trading_dashboard.py
```

### 投资组合管理
```bash
# 运行完整演示（包含投资组合管理）
uv run python scripts/trading_pipeline.py

# 查看投资组合状态
cat data/portfolio.json

# 查看生成的订单
ls data/results/step6_orders_*.json
```

### 编程接口
```python
from src.models.llm_predictor import create_llm_predictor
from src.data.data_fetcher import DataFetcher
from src.strategy.portfolio_manager import Portfolio, PortfolioManager
from src.strategy.order_generator import create_order_generator

# 初始化组件
llm_predictor = create_llm_predictor()
data_fetcher = DataFetcher()
portfolio = Portfolio(initial_cash=100000)
portfolio_manager = PortfolioManager(portfolio)

# 获取数据和预测
data = data_fetcher.fetch_data(["AAPL"], "2024-01-01", "2024-12-31")
predictions = llm_predictor.predict(data)

# 生成交易信号
from src.strategy.llm_signal_generator import create_llm_signal_generator
signal_generator = create_llm_signal_generator(llm_predictor)
signals = signal_generator.generate_signals_from_predictions(predictions)

# 生成投资组合订单
order_generator = create_order_generator(portfolio_manager, {
    'max_position_size': 0.15,
    'min_trade_size': 1000,
    'risk_per_trade': 0.02
})
orders = order_generator.generate_orders(signals, data)

# 查看结果
for symbol, pred in predictions.items():
    print(f"{symbol}: {pred['prediction']:.2%} (置信度: {pred['confidence']:.2%})")

for order in orders:
    print(f"订单: {order['action']} {order['symbol']} {order['quantity']}股")
```

## 🔧 扩展系统

- **添加新模型**: 在 `config/config.yaml` 中配置
- **自定义指标**: 修改 `src/data/technical_indicators.py`
- **自定义策略**: 扩展 `src/strategy/llm_signal_generator.py`
- **投资组合管理**: 修改 `src/strategy/portfolio_manager.py` 和 `src/strategy/order_generator.py`
- **风险控制**: 调整 `config/config.yaml` 中的投资组合参数

## 💼 投资组合管理功能

### 核心特性

- **📊 持仓跟踪**: 实时监控现金、持仓、盈亏状况
- **🎯 智能订单**: 基于LLM信号和风险评估生成买卖订单
- **⚖️ 风险控制**: 动态仓位管理，防止过度集中
- **💾 状态持久化**: JSON格式保存投资组合状态
- **📈 绩效分析**: 计算未实现盈亏和投资组合价值

### 工作流程

1. **数据获取** → 2. **特征工程** → 3. **LLM预测** → 4. **信号生成** → 5. **投资组合分析** → 6. **订单生成**

### 订单类型

- **🟢 买入订单**: 基于看涨信号和可用资金
- **🔴 卖出订单**: 基于看跌信号和现有持仓
- **🟡 持有**: 信号强度不足或风险过高

### 风险控制参数

```yaml
portfolio:
  max_position_size: 0.15    # 单个仓位最大15%
  min_trade_size: 1000       # 最小交易金额
  risk_per_trade: 0.02       # 每笔交易风险2%
  max_total_positions: 10    # 最大持仓数量
```

## ⚠️ 免责声明

本系统仅供教育和研究使用。过往表现不代表未来结果，投资有风险，请谨慎决策。

---

**Happy Trading!** 🚀