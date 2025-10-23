"""
AI Trading System - pytest配置文件

提供测试用的fixtures和配置。
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path


@pytest.fixture
def temp_dir():
    """临时目录fixture"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_ohlcv_data():
    """样本OHLCV数据"""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)  # 确保可重复性
    
    # 生成模拟价格数据
    base_price = 100
    returns = np.random.normal(0, 0.02, 100)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # 确保OHLC逻辑正确
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    return data


@pytest.fixture
def sample_market_data():
    """样本市场数据"""
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    data = {}
    
    for symbol in symbols:
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        np.random.seed(hash(symbol) % 2**32)  # 每个符号不同的随机种子
        
        base_price = 100 + hash(symbol) % 200
        returns = np.random.normal(0, 0.02, 50)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        df = pd.DataFrame({
            'open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 50),
            'rsi': np.random.uniform(20, 80, 50),
            'macd': np.random.uniform(-1, 1, 50),
            'bb_upper': [p * 1.02 for p in prices],
            'bb_lower': [p * 0.98 for p in prices]
        }, index=dates)
        
        # 确保OHLC逻辑正确
        df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
        df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
        
        data[symbol] = df
    
    return data


@pytest.fixture
def sample_predictions():
    """样本预测结果"""
    from src.models.llm_predictor import PredictionResult
    
    return {
        'AAPL': PredictionResult(
            symbol='AAPL',
            prediction='BUY',
            confidence=0.8,
            reasoning='Strong bullish momentum',
            price_target=200.0
        ),
        'MSFT': PredictionResult(
            symbol='MSFT',
            prediction='SELL',
            confidence=0.7,
            reasoning='Bearish divergence',
            price_target=300.0
        ),
        'GOOGL': PredictionResult(
            symbol='GOOGL',
            prediction='HOLD',
            confidence=0.6,
            reasoning='Mixed signals',
            price_target=None
        ),
        'TSLA': PredictionResult(
            symbol='TSLA',
            prediction='BUY',
            confidence=0.75,
            reasoning='Strong technical indicators',
            price_target=250.0
        ),
        'NVDA': PredictionResult(
            symbol='NVDA',
            prediction='HOLD',
            confidence=0.65,
            reasoning='Neutral market conditions',
            price_target=None
        )
    }


@pytest.fixture
def sample_signals():
    """样本交易信号"""
    return {
        'AAPL': {
            'symbol': 'AAPL',
            'signal': 'BUY',
            'confidence': 0.8,
            'strength': 0.6,
            'reasoning': 'Strong bullish momentum',
            'timestamp': datetime.now().isoformat()
        },
        'MSFT': {
            'symbol': 'MSFT',
            'signal': 'SELL',
            'confidence': 0.7,
            'strength': -0.5,
            'reasoning': 'Bearish divergence',
            'timestamp': datetime.now().isoformat()
        },
        'GOOGL': {
            'symbol': 'GOOGL',
            'signal': 'HOLD',
            'confidence': 0.6,
            'strength': 0.1,
            'reasoning': 'Mixed signals',
            'timestamp': datetime.now().isoformat()
        }
    }


@pytest.fixture
def sample_portfolio_data():
    """样本投资组合数据"""
    return {
        'cash': 100000.0,
        'positions': {
            'AAPL': {
                'shares': 100,
                'avg_price': 180.0,
                'current_price': 185.0,
                'market_value': 18500.0,
                'unrealized_pnl': 500.0
            },
            'MSFT': {
                'shares': 50,
                'avg_price': 300.0,
                'current_price': 295.0,
                'market_value': 14750.0,
                'unrealized_pnl': -250.0
            }
        },
        'transactions': [
            {
                'timestamp': '2023-01-01T10:00:00',
                'symbol': 'AAPL',
                'action': 'BUY',
                'shares': 100,
                'price': 180.0,
                'value': 18000.0
            },
            {
                'timestamp': '2023-01-02T10:00:00',
                'symbol': 'MSFT',
                'action': 'BUY',
                'shares': 50,
                'price': 300.0,
                'value': 15000.0
            }
        ],
        'metadata': {
            'created_at': '2023-01-01T00:00:00',
            'updated_at': datetime.now().isoformat(),
            'total_value': 133250.0,
            'total_pnl': 250.0
        }
    }


@pytest.fixture
def sample_orders():
    """样本订单数据"""
    return [
        {
            'symbol': 'AAPL',
            'action': 'BUY',
            'order_type': 'MARKET',
            'quantity': 50,
            'reason': 'LLM signal: BUY, confidence: 0.8',
            'estimated_price': 185.0,
            'risk_score': 0.3,
            'timestamp': datetime.now().isoformat()
        },
        {
            'symbol': 'MSFT',
            'action': 'SELL',
            'order_type': 'MARKET',
            'quantity': 25,
            'reason': 'LLM signal: SELL, confidence: 0.7',
            'estimated_price': 295.0,
            'risk_score': 0.4,
            'timestamp': datetime.now().isoformat()
        }
    ]


@pytest.fixture
def mock_api_key():
    """模拟API密钥"""
    return 'test_api_key_12345'


@pytest.fixture
def mock_llm_response():
    """模拟LLM响应"""
    return {
        'prediction': 'BUY',
        'confidence': 0.8,
        'reasoning': 'Strong bullish momentum with technical indicators supporting upward movement',
        'price_target': 200.0
    }


@pytest.fixture
def mock_llm_response_json():
    """模拟LLM JSON响应"""
    import json
    return json.dumps({
        'prediction': 'BUY',
        'confidence': 0.8,
        'reasoning': 'Strong bullish momentum',
        'price_target': 200.0
    })


@pytest.fixture
def mock_llm_response_text():
    """模拟LLM文本响应"""
    return """
    Prediction: BUY
    Confidence: 80%
    Reasoning: Strong bullish momentum with technical indicators supporting upward movement
    Price Target: $200
    """


@pytest.fixture
def sample_config():
    """样本配置"""
    return {
        'data': {
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'interval': '1d',
            'cache_data': True
        },
        'model': {
            'llm_model': 'deepseek/deepseek-chat-v3-0324',
            'min_confidence': 0.6,
            'temperature': 0.3,
            'max_tokens': 500
        },
        'portfolio': {
            'initial_cash': 100000,
            'max_position_size': 0.15,
            'risk_per_trade': 0.02,
            'max_total_positions': 10,
            'min_trade_size': 1000
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }


# 测试标记
pytestmark = pytest.mark.unit
