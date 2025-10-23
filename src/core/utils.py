"""
AI Trading System - 核心工具函数模块

这个模块提供了系统中使用的各种工具函数，包括配置管理、
日志设置、格式化函数、投资组合管理等。所有函数都经过优化，
提供了更好的错误处理和类型安全。
"""

import os
import yaml
import json
import logging
import logging.handlers
from datetime import datetime
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
from dotenv import load_dotenv

from .constants import (
    LogConfig, FilePaths, ModelConfig, PortfolioConfig,
    DEFAULT_LOG_LEVEL, DEFAULT_LOG_FORMAT
)


# =============================================================================
# 配置管理
# =============================================================================

class ConfigManager:
    """配置管理器"""

    def __init__(self, config_file: str = FilePaths.CONFIG_FILE):
        self.config_file = config_file
        self._config = None

    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self._config is None:
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f)
            except FileNotFoundError:
                print(f"Config file {self.config_file} not found, using defaults")
                self._config = self.get_default_config()
            except yaml.YAMLError as e:
                print(f"Error parsing YAML file: {e}")
                self._config = self.get_default_config()
        return self._config

    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'data': {
                'symbols': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
                'start_date': '2023-01-01',
                'end_date': '2024-01-31',
                'interval': '1d',
                'cache_data': True
            },
            'model': {
                'llm_model': ModelConfig.DEFAULT_MODEL,
                'min_confidence': 0.6,
                'temperature': ModelConfig.DEFAULT_TEMPERATURE,
                'max_tokens': ModelConfig.DEFAULT_MAX_TOKENS
            },
            'portfolio': {
                'initial_cash': PortfolioConfig.DEFAULT_INITIAL_CASH,
                'max_position_size': PortfolioConfig.DEFAULT_MAX_POSITION_SIZE,
                'risk_per_trade': PortfolioConfig.DEFAULT_RISK_PER_TRADE,
                'max_total_positions': PortfolioConfig.DEFAULT_MAX_TOTAL_POSITIONS,
                'min_trade_size': PortfolioConfig.DEFAULT_MIN_TRADE_SIZE
            },
            'logging': {
                'level': DEFAULT_LOG_LEVEL,
                'format': DEFAULT_LOG_FORMAT
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        config = self.load_config()
        keys = key.split('.')
        value = config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


# =============================================================================
# 日志管理
# =============================================================================

class LogManager:
    """日志管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._setup_logging()

    def _setup_logging(self) -> None:
        """设置日志配置"""
        log_config = self.config.get('logging', {})
        log_level = log_config.get('level', DEFAULT_LOG_LEVEL)
        log_format = log_config.get('format', DEFAULT_LOG_FORMAT)

        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }

        # 创建日志目录
        os.makedirs(LogConfig.LOGS_DIR, exist_ok=True)

        # 设置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(level_map.get(log_level.upper(), logging.INFO))

        # 清除现有处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(log_format)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # 文件处理器
        file_handler = logging.handlers.RotatingFileHandler(
            LogConfig.LOG_FILE,
            maxBytes=LogConfig.MAX_LOG_SIZE,
            backupCount=LogConfig.BACKUP_COUNT
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        # 错误日志处理器
        error_handler = logging.handlers.RotatingFileHandler(
            LogConfig.ERROR_LOG_FILE,
            maxBytes=LogConfig.MAX_LOG_SIZE,
            backupCount=LogConfig.BACKUP_COUNT
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)


# =============================================================================
# 环境管理
# =============================================================================

class EnvironmentManager:
    """环境管理器"""

    def __init__(self):
        load_dotenv()

    def get_api_key(self, key_name: str = 'OPENROUTER_API_KEY') -> Optional[str]:
        """获取API密钥"""
        api_key = os.getenv(key_name)
        if not api_key:
            self._print_api_key_warning(key_name)
        return api_key

    def _print_api_key_warning(self, key_name: str) -> None:
        """打印API密钥警告"""
        print_warning(f"{key_name} not found!")
        print("Please set your API key in one of these ways:")
        print(f"1. Create a .env file with: {key_name}='your-api-key-here'")
        print(f"2. Set environment variable: export {key_name}='your-api-key-here'")
        print("\nYou can copy the template: cp config/env_template.txt .env")

    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            'model': os.getenv('DEFAULT_LLM_MODEL', ModelConfig.DEFAULT_MODEL),
            'temperature': float(os.getenv('MODEL_TEMPERATURE', ModelConfig.DEFAULT_TEMPERATURE)),
            'max_tokens': int(os.getenv('MODEL_MAX_TOKENS', ModelConfig.DEFAULT_MAX_TOKENS))
        }


# =============================================================================
# 格式化工具
# =============================================================================

class Formatter:
    """格式化工具类"""

    @staticmethod
    def format_percentage(value: float, decimals: int = 2) -> str:
        """格式化百分比"""
        return f"{value:.{decimals}%}"

    @staticmethod
    def format_number(value: float, decimals: int = 2) -> str:
        """格式化数字"""
        return f"{value:.{decimals}f}"

    @staticmethod
    def format_currency(value: float, currency: str = "$") -> str:
        """格式化货币"""
        return f"{currency}{value:,.2f}"

    @staticmethod
    def format_large_number(value: float) -> str:
        """格式化大数字（K, M, B）"""
        if value >= 1e9:
            return f"{value/1e9:.1f}B"
        elif value >= 1e6:
            return f"{value/1e6:.1f}M"
        elif value >= 1e3:
            return f"{value/1e3:.1f}K"
        else:
            return f"{value:.0f}"

    @staticmethod
    def format_timestamp(timestamp: Union[str, datetime]) -> str:
        """格式化时间戳"""
        if isinstance(timestamp, str):
            return timestamp
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")


# =============================================================================
# 投资组合工具
# =============================================================================

class PortfolioUtils:
    """投资组合工具类"""

    @staticmethod
    def load_portfolio(portfolio_file: str = PortfolioConfig.PORTFOLIO_FILE):
        """加载投资组合"""
        try:
            from ..strategy.portfolio_manager import Portfolio
            return Portfolio(portfolio_file=portfolio_file)
        except ImportError as e:
            print_error(f"Failed to import Portfolio: {e}")
            return None

    @staticmethod
    def save_portfolio(portfolio, portfolio_file: str = PortfolioConfig.PORTFOLIO_FILE) -> bool:
        """保存投资组合"""
        try:
            return portfolio.save_to_file()
        except Exception as e:
            print_error(f"Failed to save portfolio: {e}")
            return False

    @staticmethod
    def calculate_position_size(
        available_cash: float,
        price: float,
        max_position_size: float,
        total_value: float,
        min_trade_size: float = PortfolioConfig.DEFAULT_MIN_TRADE_SIZE
    ) -> int:
        """计算仓位大小"""
        if price <= 0:
            return 0

        # 基于可用现金计算
        shares_by_cash = int(available_cash / price)

        # 基于最大仓位比例计算
        max_position_value = total_value * max_position_size
        shares_by_position = int(max_position_value / price)

        # 取较小值
        shares = min(shares_by_cash, shares_by_position)

        # 检查最小交易规模
        trade_value = shares * price
        if trade_value < min_trade_size:
            return 0

        return shares


# =============================================================================
# 文件工具
# =============================================================================

class FileUtils:
    """文件工具类"""

    @staticmethod
    def ensure_directory(path: str) -> None:
        """确保目录存在"""
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def safe_json_load(file_path: str, default: Any = None) -> Any:
        """安全加载JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    @staticmethod
    def safe_json_save(data: Any, file_path: str) -> bool:
        """安全保存JSON文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 创建备份
            backup_path = file_path + '.backup'
            if os.path.exists(file_path):
                os.rename(file_path, backup_path)

            # 保存新文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            # 删除备份
            if os.path.exists(backup_path):
                os.remove(backup_path)

            return True
        except Exception as e:
            print_error(f"Failed to save JSON file {file_path}: {e}")
            # 恢复备份
            if os.path.exists(backup_path):
                os.rename(backup_path, file_path)
            return False


# =============================================================================
# 显示工具
# =============================================================================

class DisplayUtils:
    """显示工具类"""

    @staticmethod
    def print_section_header(title: str, width: int = 80) -> None:
        """打印章节标题"""
        print("=" * width)
        print(title)
        print("=" * width)

    @staticmethod
    def print_step_header(step: str, description: str, width: int = 40) -> None:
        """打印步骤标题"""
        print(f"\n{step}: {description}")
        print("-" * width)

    @staticmethod
    def print_success(message: str) -> None:
        """打印成功消息"""
        print(f"✓ {message}")

    @staticmethod
    def print_error(message: str) -> None:
        """打印错误消息"""
        print(f"❌ {message}")

    @staticmethod
    def print_warning(message: str) -> None:
        """打印警告消息"""
        print(f"⚠️  {message}")

    @staticmethod
    def print_info(message: str) -> None:
        """打印信息消息"""
        print(f"ℹ️  {message}")


# =============================================================================
# 向后兼容性函数
# =============================================================================

# 配置管理
def load_config(config_file: str = FilePaths.CONFIG_FILE) -> Dict[str, Any]:
    """加载配置文件（向后兼容）"""
    manager = ConfigManager(config_file)
    return manager.load_config()

def get_default_config() -> Dict[str, Any]:
    """获取默认配置（向后兼容）"""
    manager = ConfigManager()
    return manager.get_default_config()

def setup_logging(config: Dict[str, Any]) -> None:
    """设置日志配置（向后兼容）"""
    log_manager = LogManager(config)
    # 日志管理器会自动设置日志

def get_default_model() -> str:
    """获取默认模型（向后兼容）"""
    env_manager = EnvironmentManager()
    model_config = env_manager.get_model_config()
    return model_config['model']

# 显示函数
def print_section_header(title: str, width: int = 80) -> None:
    """打印章节标题（向后兼容）"""
    DisplayUtils.print_section_header(title, width)

def print_step_header(step: str, description: str, width: int = 40) -> None:
    """打印步骤标题（向后兼容）"""
    DisplayUtils.print_step_header(step, description, width)

def print_success(message: str) -> None:
    """打印成功消息（向后兼容）"""
    DisplayUtils.print_success(message)

def print_error(message: str) -> None:
    """打印错误消息（向后兼容）"""
    DisplayUtils.print_error(message)

def print_warning(message: str) -> None:
    """打印警告消息（向后兼容）"""
    DisplayUtils.print_warning(message)

# 环境管理
def ensure_api_key() -> Optional[str]:
    """确保API密钥存在（向后兼容）"""
    env_manager = EnvironmentManager()
    return env_manager.get_api_key()

# 格式化函数
def format_percentage(value: float) -> str:
    """格式化百分比（向后兼容）"""
    return Formatter.format_percentage(value)

def format_number(value: float, decimals: int = 2) -> str:
    """格式化数字（向后兼容）"""
    return Formatter.format_number(value, decimals)

def format_currency(value: float, currency: str = "$") -> str:
    """格式化货币（向后兼容）"""
    return Formatter.format_currency(value, currency)

# 投资组合函数
def load_portfolio(portfolio_file: str = PortfolioConfig.PORTFOLIO_FILE):
    """加载投资组合（向后兼容）"""
    return PortfolioUtils.load_portfolio(portfolio_file)

def save_portfolio(portfolio, portfolio_file: str = PortfolioConfig.PORTFOLIO_FILE) -> bool:
    """保存投资组合（向后兼容）"""
    return PortfolioUtils.save_portfolio(portfolio, portfolio_file)

def calculate_position_size(available_cash: float, price: float,
                           max_position_size: float, total_value: float) -> int:
    """计算仓位大小（向后兼容）"""
    return PortfolioUtils.calculate_position_size(
        available_cash, price, max_position_size, total_value
    )
