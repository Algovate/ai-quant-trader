"""
通用工具函数模块
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv


def load_config(config_file: str = "config/config.yaml") -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Config file {config_file} not found, using defaults")
        return get_default_config()
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
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
            'llm_model': 'deepseek/deepseek-chat-v3-0324',
            'min_confidence': 0.6
        },
        'strategy': {
            'initial_capital': 100000,
            'position_size': 0.1
        }
    }


def setup_logging(config: Dict[str, Any]) -> None:
    """设置日志配置"""
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    logging.basicConfig(
        level=level_map.get(log_level.upper(), logging.INFO),
        format=log_format
    )


def get_default_model() -> str:
    """获取默认模型"""
    try:
        config = load_config()
        return config.get('model', {}).get('default_model', 'deepseek/deepseek-chat-v3-0324')
    except:
        return os.getenv('DEFAULT_LLM_MODEL', 'deepseek/deepseek-chat-v3-0324')


def print_section_header(title: str, width: int = 80) -> None:
    """打印章节标题"""
    print("=" * width)
    print(title)
    print("=" * width)


def print_step_header(step: str, description: str, width: int = 40) -> None:
    """打印步骤标题"""
    print(f"\n{step}: {description}")
    print("-" * width)


def print_success(message: str) -> None:
    """打印成功消息"""
    print(f"✓ {message}")


def print_error(message: str) -> None:
    """打印错误消息"""
    print(f"❌ {message}")


def print_warning(message: str) -> None:
    """打印警告消息"""
    print(f"⚠️  {message}")


def ensure_api_key() -> Optional[str]:
    """确保API密钥存在"""
    load_dotenv()
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print_warning("OpenRouter API key not found!")
        print("Please set your API key in one of these ways:")
        print("1. Create a .env file with: OPENROUTER_API_KEY='your-api-key-here'")
        print("2. Set environment variable: export OPENROUTER_API_KEY='your-api-key-here'")
        print("\nYou can copy the template: cp env_template.txt .env")
    return api_key


def format_percentage(value: float) -> str:
    """格式化百分比"""
    return f"{value:.2%}"


def format_number(value: float, decimals: int = 2) -> str:
    """格式化数字"""
    return f"{value:.{decimals}f}"
