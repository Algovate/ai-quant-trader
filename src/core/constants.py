"""
AI Trading System - 核心常量定义

这个模块定义了系统中使用的所有常量，包括API配置、模型参数、
技术指标设置、交易参数等。常量按功能分类组织，便于维护和修改。
"""

from typing import List, Dict, Any
from enum import Enum


class SignalType(Enum):
    """交易信号类型枚举"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderType(Enum):
    """订单类型枚举"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


# =============================================================================
# API 配置
# =============================================================================

class APIConfig:
    """API相关配置"""
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_TIMEOUT = 30
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RATE_LIMIT_DELAY = 1.0
    REQUEST_HEADERS = {
        "Content-Type": "application/json",
        "User-Agent": "AI-Trading-System/1.0"
    }


# =============================================================================
# 模型配置
# =============================================================================

class ModelConfig:
    """LLM模型相关配置"""
    DEFAULT_MODEL = "deepseek/deepseek-chat-v3-0324"
    FALLBACK_MODEL = "openai/gpt-4o-mini"
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_MAX_TOKENS = 500
    MAX_CONTEXT_LENGTH = 4096

    # 支持的模型列表
    SUPPORTED_MODELS = [
        "deepseek/deepseek-chat-v3-0324",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-pro-1.5"
    ]


# =============================================================================
# 交易配置
# =============================================================================

class TradingConfig:
    """交易相关配置"""
    DEFAULT_MIN_CONFIDENCE = 0.6
    DEFAULT_POSITION_SIZE = 0.1
    DEFAULT_STOP_LOSS = 0.05
    DEFAULT_TAKE_PROFIT = 0.15
    DEFAULT_SIGNAL_THRESHOLD = 0.5

    # 风险控制
    MAX_PORTFOLIO_RISK = 0.20  # 最大投资组合风险
    MAX_POSITION_RISK = 0.05   # 最大单仓位风险
    MIN_TRADE_SIZE = 1000      # 最小交易金额


# =============================================================================
# 技术指标配置
# =============================================================================

class TechnicalIndicators:
    """技术指标参数配置"""

    # RSI配置
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30

    # MACD配置
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9

    # 布林带配置
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2

    # 随机指标配置
    STOCH_K_PERIOD = 14
    STOCH_D_PERIOD = 3

    # 威廉指标配置
    WILLIAMS_R_PERIOD = 14

    # ATR配置
    ATR_PERIOD = 14

    # 移动平均线配置
    MA_PERIODS = [5, 10, 20, 50, 200]

    # 成交量指标配置
    VOLUME_MA_PERIOD = 20
    VOLUME_RATIO_THRESHOLD = 1.5


# =============================================================================
# 投资组合配置
# =============================================================================

class PortfolioConfig:
    """投资组合相关配置"""
    DEFAULT_INITIAL_CASH = 100000
    DEFAULT_MAX_POSITION_SIZE = 0.15
    DEFAULT_RISK_PER_TRADE = 0.02
    DEFAULT_MAX_TOTAL_POSITIONS = 10
    DEFAULT_MIN_TRADE_SIZE = 1000
    PORTFOLIO_FILE = "data/portfolio.json"

    # 仓位管理
    MAX_POSITION_WEIGHT = 0.20  # 最大单仓位权重
    MIN_POSITION_WEIGHT = 0.01  # 最小单仓位权重

    # 风险控制
    MAX_DRAWDOWN = 0.15         # 最大回撤
    MAX_CORRELATION = 0.7       # 最大相关性


# =============================================================================
# 数据配置
# =============================================================================

class DataConfig:
    """数据相关配置"""
    CACHE_DIR = "data/cache"
    RESULTS_DIR = "data/results"
    CACHE_EXTENSION = ".pkl"
    JSON_EXTENSION = ".json"

    # 数据获取配置
    DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    DEFAULT_START_DATE = "2023-01-01"
    DEFAULT_END_DATE = "2024-01-31"
    DEFAULT_INTERVAL = "1d"

    # 数据源配置
    POLYGON_API_BASE = "https://api.polygon.io"
    YAHOO_FINANCE_BASE = "https://query1.finance.yahoo.com"


# =============================================================================
# 日志配置
# =============================================================================

class LogConfig:
    """日志相关配置"""
    DEFAULT_LOG_LEVEL = "INFO"
    DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOGS_DIR = "logs"
    LOG_FILE = "logs/trading_system.log"
    ERROR_LOG_FILE = "logs/error.log"
    DEBUG_LOG_FILE = "logs/debug.log"
    
    # 日志轮转配置
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 5


# =============================================================================
# 系统提示词
# =============================================================================

class SystemPrompts:
    """系统提示词配置"""
    DEFAULT_SYSTEM_PROMPT = "You are a professional financial analyst."

    FINANCIAL_ANALYST_PROMPT = (
        "You are an expert quantitative analyst specializing in stock market analysis. "
        "Provide detailed, data-driven insights with clear reasoning."
    )

    CONCISE_ANALYST_PROMPT = (
        "Provide quick, concise market analysis. Focus on key insights and actionable recommendations."
    )

    TRADING_EXPERT_PROMPT = (
        "You are a professional financial analyst with expertise in quantitative trading. "
        "Provide precise trading recommendations with risk assessment."
    )

    PORTFOLIO_MANAGER_PROMPT = (
        "You are an expert portfolio manager. Analyze market conditions and provide "
        "investment recommendations with proper risk management."
    )


# =============================================================================
# 文件路径配置
# =============================================================================

class FilePaths:
    """文件路径配置"""
    CONFIG_FILE = "config/config.yaml"
    ENV_TEMPLATE = "config/env_template.txt"
    PORTFOLIO_FILE = "data/portfolio.json"
    CACHE_DIR = "data/cache"
    RESULTS_DIR = "data/results"
    LOGS_DIR = "logs"
    TESTS_DIR = "tests"


# =============================================================================
# 向后兼容性 - 保持原有常量
# =============================================================================

# API配置
OPENROUTER_BASE_URL = APIConfig.OPENROUTER_BASE_URL
DEFAULT_TIMEOUT = APIConfig.DEFAULT_TIMEOUT
DEFAULT_MAX_RETRIES = APIConfig.DEFAULT_MAX_RETRIES
DEFAULT_RATE_LIMIT_DELAY = APIConfig.DEFAULT_RATE_LIMIT_DELAY

# 模型配置
DEFAULT_MODEL = ModelConfig.DEFAULT_MODEL
FALLBACK_MODEL = ModelConfig.FALLBACK_MODEL
DEFAULT_TEMPERATURE = ModelConfig.DEFAULT_TEMPERATURE
DEFAULT_MAX_TOKENS = ModelConfig.DEFAULT_MAX_TOKENS

# 交易配置
DEFAULT_MIN_CONFIDENCE = TradingConfig.DEFAULT_MIN_CONFIDENCE
DEFAULT_POSITION_SIZE = TradingConfig.DEFAULT_POSITION_SIZE
DEFAULT_STOP_LOSS = TradingConfig.DEFAULT_STOP_LOSS
DEFAULT_TAKE_PROFIT = TradingConfig.DEFAULT_TAKE_PROFIT

# 技术指标配置
RSI_PERIOD = TechnicalIndicators.RSI_PERIOD
MACD_FAST = TechnicalIndicators.MACD_FAST
MACD_SLOW = TechnicalIndicators.MACD_SLOW
MACD_SIGNAL = TechnicalIndicators.MACD_SIGNAL
BOLLINGER_PERIOD = TechnicalIndicators.BOLLINGER_PERIOD
BOLLINGER_STD = TechnicalIndicators.BOLLINGER_STD
STOCH_K_PERIOD = TechnicalIndicators.STOCH_K_PERIOD
STOCH_D_PERIOD = TechnicalIndicators.STOCH_D_PERIOD
WILLIAMS_R_PERIOD = TechnicalIndicators.WILLIAMS_R_PERIOD
ATR_PERIOD = TechnicalIndicators.ATR_PERIOD
MA_PERIODS = TechnicalIndicators.MA_PERIODS

# 缓存配置
CACHE_DIR = DataConfig.CACHE_DIR
CACHE_EXTENSION = DataConfig.CACHE_EXTENSION

# 日志配置
DEFAULT_LOG_LEVEL = LogConfig.DEFAULT_LOG_LEVEL
DEFAULT_LOG_FORMAT = LogConfig.DEFAULT_LOG_FORMAT

# 信号类型
SIGNAL_BUY = SignalType.BUY.value
SIGNAL_SELL = SignalType.SELL.value
SIGNAL_HOLD = SignalType.HOLD.value

# 系统提示词
DEFAULT_SYSTEM_PROMPT = SystemPrompts.DEFAULT_SYSTEM_PROMPT
FINANCIAL_ANALYST_PROMPT = SystemPrompts.FINANCIAL_ANALYST_PROMPT
CONCISE_ANALYST_PROMPT = SystemPrompts.CONCISE_ANALYST_PROMPT
TRADING_EXPERT_PROMPT = SystemPrompts.TRADING_EXPERT_PROMPT

# Portfolio defaults
DEFAULT_INITIAL_CASH = PortfolioConfig.DEFAULT_INITIAL_CASH
DEFAULT_MAX_POSITION_SIZE = PortfolioConfig.DEFAULT_MAX_POSITION_SIZE
DEFAULT_RISK_PER_TRADE = PortfolioConfig.DEFAULT_RISK_PER_TRADE
PORTFOLIO_FILE = PortfolioConfig.PORTFOLIO_FILE
