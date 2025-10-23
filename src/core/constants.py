"""
常量定义模块
"""

# API配置
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_RATE_LIMIT_DELAY = 1.0

# 模型配置
DEFAULT_MODEL = "deepseek/deepseek-chat-v3-0324"
FALLBACK_MODEL = "openai/gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 500

# 交易配置
DEFAULT_MIN_CONFIDENCE = 0.6
DEFAULT_POSITION_SIZE = 0.1
DEFAULT_STOP_LOSS = 0.05
DEFAULT_TAKE_PROFIT = 0.15

# 技术指标配置
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
WILLIAMS_R_PERIOD = 14
ATR_PERIOD = 14

# 移动平均线配置
MA_PERIODS = [5, 10, 20, 50, 200]

# 缓存配置
CACHE_DIR = "data/cache"
CACHE_EXTENSION = ".pkl"

# 日志配置
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 信号类型
SIGNAL_BUY = "BUY"
SIGNAL_SELL = "SELL"
SIGNAL_HOLD = "HOLD"

# 系统提示词
DEFAULT_SYSTEM_PROMPT = "You are a professional financial analyst."
FINANCIAL_ANALYST_PROMPT = "You are an expert quantitative analyst specializing in stock market analysis."
CONCISE_ANALYST_PROMPT = "Provide quick, concise market analysis."
TRADING_EXPERT_PROMPT = "You are a professional financial analyst with expertise in quantitative trading."
