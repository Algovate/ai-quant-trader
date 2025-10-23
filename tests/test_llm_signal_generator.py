"""
AI Trading System - LLM信号生成器模块单元测试

测试LLMSignalGenerator类的各种功能，包括信号生成、强度计算、置信度处理等。
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from src.strategy.llm_signal_generator import LLMSignalGenerator
from src.models.llm_predictor import PredictionResult


class TestLLMSignalGenerator:
    """测试LLM信号生成器"""
    
    @pytest.fixture
    def signal_generator(self):
        """信号生成器实例"""
        return LLMSignalGenerator()
    
    @pytest.fixture
    def sample_predictions(self):
        """样本预测结果"""
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
            )
        }
    
    @pytest.fixture
    def sample_market_data(self):
        """样本市场数据"""
        return {
            'AAPL': pd.DataFrame({
                'close': [180, 182, 185, 183, 186],
                'volume': [1000, 1100, 1200, 1300, 1400],
                'rsi': [50, 55, 60, 58, 62],
                'macd': [0.1, 0.2, 0.3, 0.25, 0.35]
            }),
            'MSFT': pd.DataFrame({
                'close': [300, 298, 295, 297, 294],
                'volume': [2000, 2100, 2200, 2300, 2400],
                'rsi': [70, 68, 65, 67, 63],
                'macd': [-0.1, -0.2, -0.3, -0.25, -0.35]
            }),
            'GOOGL': pd.DataFrame({
                'close': [2500, 2510, 2505, 2515, 2520],
                'volume': [500, 550, 600, 650, 700],
                'rsi': [45, 47, 49, 48, 50],
                'macd': [0.05, 0.1, 0.08, 0.12, 0.15]
            })
        }
    
    def test_init(self, signal_generator):
        """测试初始化"""
        assert signal_generator.min_confidence == 0.6
        assert signal_generator.signal_threshold == 0.5
        assert signal_generator.volatility_adjustment == True
    
    def test_init_with_config(self):
        """测试带配置初始化"""
        config = {
            'min_confidence': 0.7,
            'signal_threshold': 0.6,
            'volatility_adjustment': False
        }
        
        generator = LLMSignalGenerator(config)
        
        assert generator.min_confidence == 0.7
        assert generator.signal_threshold == 0.6
        assert generator.volatility_adjustment == False
    
    def test_calculate_signal_strength_buy(self, signal_generator):
        """测试计算买入信号强度"""
        prediction = PredictionResult(
            symbol='AAPL',
            prediction='BUY',
            confidence=0.8,
            reasoning='Strong signal'
        )
        
        market_data = pd.DataFrame({
            'close': [180, 182, 185, 183, 186],
            'rsi': [50, 55, 60, 58, 62],
            'macd': [0.1, 0.2, 0.3, 0.25, 0.35]
        })
        
        strength = signal_generator._calculate_signal_strength(prediction, market_data)
        
        assert strength > 0  # 买入信号应该为正
        assert strength <= 1.0  # 强度应该在0-1之间
    
    def test_calculate_signal_strength_sell(self, signal_generator):
        """测试计算卖出信号强度"""
        prediction = PredictionResult(
            symbol='AAPL',
            prediction='SELL',
            confidence=0.7,
            reasoning='Bearish signal'
        )
        
        market_data = pd.DataFrame({
            'close': [186, 185, 183, 185, 180],
            'rsi': [70, 68, 65, 67, 60],
            'macd': [-0.1, -0.2, -0.3, -0.25, -0.35]
        })
        
        strength = signal_generator._calculate_signal_strength(prediction, market_data)
        
        assert strength < 0  # 卖出信号应该为负
        assert strength >= -1.0  # 强度应该在-1到0之间
    
    def test_calculate_signal_strength_hold(self, signal_generator):
        """测试计算持有信号强度"""
        prediction = PredictionResult(
            symbol='AAPL',
            prediction='HOLD',
            confidence=0.6,
            reasoning='Neutral signal'
        )
        
        market_data = pd.DataFrame({
            'close': [180, 181, 180, 182, 181],
            'rsi': [50, 51, 49, 52, 50],
            'macd': [0.0, 0.01, -0.01, 0.02, 0.0]
        })
        
        strength = signal_generator._calculate_signal_strength(prediction, market_data)
        
        assert abs(strength) < 0.1  # 持有信号强度应该接近0
    
    def test_calculate_volatility_adjustment(self, signal_generator):
        """测试计算波动率调整"""
        market_data = pd.DataFrame({
            'close': [100, 110, 90, 120, 80],  # 高波动率
            'returns': [0.1, -0.18, 0.33, -0.33, 0.5]
        })
        
        adjustment = signal_generator._calculate_volatility_adjustment(market_data)
        
        assert adjustment < 1.0  # 高波动率应该降低信号强度
        assert adjustment > 0.0
    
    def test_calculate_volatility_adjustment_low_volatility(self, signal_generator):
        """测试计算低波动率调整"""
        market_data = pd.DataFrame({
            'close': [100, 101, 100, 102, 101],  # 低波动率
            'returns': [0.01, -0.01, 0.02, -0.01, 0.01]
        })
        
        adjustment = signal_generator._calculate_volatility_adjustment(market_data)
        
        assert adjustment >= 1.0  # 低波动率应该保持或增强信号强度
    
    def test_generate_signal_single_symbol(self, signal_generator, sample_predictions, sample_market_data):
        """测试生成单个符号信号"""
        prediction = sample_predictions['AAPL']
        market_data = sample_market_data['AAPL']
        
        signal = signal_generator.generate_signal('AAPL', prediction, market_data)
        
        assert signal['symbol'] == 'AAPL'
        assert signal['signal'] == 'BUY'
        assert signal['confidence'] == 0.8
        assert signal['strength'] > 0
        assert 'reasoning' in signal
        assert 'timestamp' in signal
    
    def test_generate_signal_low_confidence(self, signal_generator, sample_market_data):
        """测试低置信度信号生成"""
        prediction = PredictionResult(
            symbol='AAPL',
            prediction='BUY',
            confidence=0.5,  # 低于最小置信度
            reasoning='Weak signal'
        )
        market_data = sample_market_data['AAPL']
        
        signal = signal_generator.generate_signal('AAPL', prediction, market_data)
        
        assert signal['signal'] == 'HOLD'  # 应该被转换为HOLD
        assert signal['confidence'] == 0.5
    
    def test_generate_signal_weak_strength(self, signal_generator, sample_market_data):
        """测试弱信号强度处理"""
        prediction = PredictionResult(
            symbol='AAPL',
            prediction='BUY',
            confidence=0.8,
            reasoning='Strong signal'
        )
        market_data = sample_market_data['AAPL']
        
        # 模拟弱信号强度
        with patch.object(signal_generator, '_calculate_signal_strength', return_value=0.1):
            signal = signal_generator.generate_signal('AAPL', prediction, market_data)
            
            assert signal['signal'] == 'HOLD'  # 弱信号应该被转换为HOLD
    
    def test_generate_signals_batch(self, signal_generator, sample_predictions, sample_market_data):
        """测试批量生成信号"""
        signals = signal_generator.generate_signals(sample_predictions, sample_market_data)
        
        assert len(signals) == 3
        assert 'AAPL' in signals
        assert 'MSFT' in signals
        assert 'GOOGL' in signals
        
        # 检查每个信号的基本属性
        for symbol, signal in signals.items():
            assert signal['symbol'] == symbol
            assert signal['signal'] in ['BUY', 'SELL', 'HOLD']
            assert 0 <= signal['confidence'] <= 1
            assert 'strength' in signal
            assert 'reasoning' in signal
            assert 'timestamp' in signal
    
    def test_generate_signals_missing_market_data(self, signal_generator, sample_predictions):
        """测试缺少市场数据的信号生成"""
        market_data = {'AAPL': sample_predictions['AAPL']}  # 只有预测，没有市场数据
        
        signals = signal_generator.generate_signals(sample_predictions, market_data)
        
        assert len(signals) == 1  # 只处理有市场数据的符号
        assert 'AAPL' in signals
    
    def test_generate_signals_empty_predictions(self, signal_generator, sample_market_data):
        """测试空预测的信号生成"""
        signals = signal_generator.generate_signals({}, sample_market_data)
        
        assert len(signals) == 0
    
    def test_get_signal_summary(self, signal_generator):
        """测试获取信号摘要"""
        signals = {
            'AAPL': {
                'symbol': 'AAPL',
                'signal': 'BUY',
                'confidence': 0.8,
                'strength': 0.6
            },
            'MSFT': {
                'symbol': 'MSFT',
                'signal': 'SELL',
                'confidence': 0.7,
                'strength': -0.5
            },
            'GOOGL': {
                'symbol': 'GOOGL',
                'signal': 'HOLD',
                'confidence': 0.6,
                'strength': 0.1
            }
        }
        
        summary = signal_generator.get_signal_summary(signals)
        
        assert summary['total_signals'] == 3
        assert summary['buy_signals'] == 1
        assert summary['sell_signals'] == 1
        assert summary['hold_signals'] == 1
        assert summary['avg_confidence'] == (0.8 + 0.7 + 0.6) / 3
        assert summary['avg_strength'] == (0.6 - 0.5 + 0.1) / 3
    
    def test_get_signal_summary_empty(self, signal_generator):
        """测试空信号的摘要"""
        summary = signal_generator.get_signal_summary({})
        
        assert summary['total_signals'] == 0
        assert summary['buy_signals'] == 0
        assert summary['sell_signals'] == 0
        assert summary['hold_signals'] == 0
        assert summary['avg_confidence'] == 0.0
        assert summary['avg_strength'] == 0.0
    
    def test_filter_signals_by_confidence(self, signal_generator):
        """测试按置信度过滤信号"""
        signals = {
            'AAPL': {'signal': 'BUY', 'confidence': 0.8},
            'MSFT': {'signal': 'SELL', 'confidence': 0.5},
            'GOOGL': {'signal': 'HOLD', 'confidence': 0.7}
        }
        
        filtered = signal_generator.filter_signals_by_confidence(signals, min_confidence=0.6)
        
        assert len(filtered) == 2
        assert 'AAPL' in filtered
        assert 'GOOGL' in filtered
        assert 'MSFT' not in filtered
    
    def test_filter_signals_by_strength(self, signal_generator):
        """测试按强度过滤信号"""
        signals = {
            'AAPL': {'signal': 'BUY', 'strength': 0.8},
            'MSFT': {'signal': 'SELL', 'strength': -0.3},
            'GOOGL': {'signal': 'HOLD', 'strength': 0.1}
        }
        
        filtered = signal_generator.filter_signals_by_strength(signals, min_strength=0.5)
        
        assert len(filtered) == 1
        assert 'AAPL' in filtered
        assert 'MSFT' not in filtered
        assert 'GOOGL' not in filtered
    
    def test_validate_signal(self, signal_generator):
        """测试信号验证"""
        valid_signal = {
            'symbol': 'AAPL',
            'signal': 'BUY',
            'confidence': 0.8,
            'strength': 0.6,
            'reasoning': 'Strong signal',
            'timestamp': datetime.now().isoformat()
        }
        
        assert signal_generator._validate_signal(valid_signal) == True
        
        # 测试无效信号
        invalid_signal = {
            'symbol': 'AAPL',
            'signal': 'INVALID',  # 无效信号类型
            'confidence': 0.8
        }
        
        assert signal_generator._validate_signal(invalid_signal) == False
    
    def test_export_signals(self, signal_generator, tmp_path):
        """测试导出信号"""
        signals = {
            'AAPL': {
                'symbol': 'AAPL',
                'signal': 'BUY',
                'confidence': 0.8,
                'strength': 0.6,
                'reasoning': 'Strong signal',
                'timestamp': datetime.now().isoformat()
            }
        }
        
        output_file = tmp_path / 'test_signals.json'
        result = signal_generator.export_signals(signals, str(output_file))
        
        assert result == str(output_file)
        assert output_file.exists()
        
        # 验证文件内容
        import json
        with open(output_file) as f:
            data = json.load(f)
            assert 'AAPL' in data
            assert data['AAPL']['signal'] == 'BUY'
