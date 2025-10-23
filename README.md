# 🤖 LLM-Powered AI Trading System

AI驱动的量化交易系统，使用大语言模型进行智能市场分析和交易信号生成。

## ✨ 核心功能

- **🧠 LLM集成**: 通过OpenRouter API集成Claude、GPT-4、Gemini等先进LLM
- **📊 智能分析**: AI提供详细的市场分析和置信度评分
- **🔄 数据处理**: 自动获取、清理和处理历史股票数据
- **📈 技术指标**: 计算20+技术指标（RSI、MACD、布林带等）
- **🎯 信号生成**: 基于AI预测生成交易信号
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
uv run python scripts/llm_demo.py

# 快速测试
uv run python scripts/llm_demo.py test
```

## 📁 项目结构

```
stock/
├── src/                          # 源代码
│   ├── core/                     # 核心模块（常量、工具）
│   ├── data/                     # 数据处理（获取、清理、特征工程）
│   ├── models/                   # AI模型（LLM预测器、模型管理）
│   └── strategy/                 # 交易策略（信号生成）
├── config/                       # 配置文件
├── scripts/                      # 可执行脚本
├── data/                         # 数据存储
│   ├── cache/                   # 缓存数据
│   └── results/                 # 中间结果（用于核查）
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
```

## 💡 使用示例

### 运行完整演示
```bash
# 运行完整演示
uv run python scripts/llm_demo.py

# 快速测试
uv run python scripts/llm_demo.py test
```

### 查看中间结果
```bash
# 列出所有结果文件
uv run python scripts/view_results.py --list

# 查看最新结果摘要
uv run python scripts/view_results.py

# 查看特定文件
uv run python scripts/view_results.py --file data/results/step4_predictions_*.json
```

### 编程接口
```python
from src.models.llm_predictor import create_llm_predictor
from src.data.data_fetcher import DataFetcher

# 初始化组件
llm_predictor = create_llm_predictor()
data_fetcher = DataFetcher()

# 获取数据和预测
data = data_fetcher.fetch_data(["AAPL"], "2024-01-01", "2024-12-31")
predictions = llm_predictor.predict(data)

# 查看结果
for symbol, pred in predictions.items():
    print(f"{symbol}: {pred['prediction']:.2%} (置信度: {pred['confidence']:.2%})")
```

## 🔧 扩展系统

- **添加新模型**: 在 `config/config.yaml` 中配置
- **自定义指标**: 修改 `src/data/technical_indicators.py`
- **自定义策略**: 扩展 `src/strategy/llm_signal_generator.py`

## ⚠️ 免责声明

本系统仅供教育和研究使用。过往表现不代表未来结果，投资有风险，请谨慎决策。

---

**Happy Trading!** 🚀