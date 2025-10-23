"""
AI Trading System - LLM预测器模块单元测试

测试LLMPredictor、LLMResponseParser、PromptGenerator类的各种功能。
"""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.models.llm_predictor import (
    LLMPredictor, LLMResponseParser, PromptGenerator, 
    PredictionResult, PredictionType
)


class TestPredictionResult:
    """测试预测结果数据类"""
    
    def test_init(self):
        """测试初始化"""
        result = PredictionResult(
            symbol='AAPL',
            prediction='BUY',
            confidence=0.8,
            reasoning='Strong bullish signal'
        )
        
        assert result.symbol == 'AAPL'
        assert result.prediction == 'BUY'
        assert result.confidence == 0.8
        assert result.reasoning == 'Strong bullish signal'
        assert result.price_target is None
        assert isinstance(result.timestamp, datetime)
    
    def test_init_with_price_target(self):
        """测试带价格目标的初始化"""
        result = PredictionResult(
            symbol='AAPL',
            prediction='BUY',
            confidence=0.8,
            reasoning='Strong bullish signal',
            price_target=200.0
        )
        
        assert result.price_target == 200.0


class TestLLMResponseParser:
    """测试LLM响应解析器"""
    
    def test_parse_json_response(self):
        """测试解析JSON格式响应"""
        response = json.dumps({
            'prediction': 'BUY',
            'confidence': 0.75,
            'reasoning': 'Strong technical indicators',
            'price_target': 200.0
        })
        
        result = LLMResponseParser.parse_prediction_response(response, 'AAPL')
        
        assert result.symbol == 'AAPL'
        assert result.prediction == 'BUY'
        assert result.confidence == 0.75
        assert result.reasoning == 'Strong technical indicators'
        assert result.price_target == 200.0
    
    def test_parse_text_response(self):
        """测试解析文本格式响应"""
        response = """
        Prediction: BUY
        Confidence: 80%
        Reasoning: Strong bullish momentum
        Price Target: $200
        """
        
        result = LLMResponseParser.parse_prediction_response(response, 'AAPL')
        
        assert result.symbol == 'AAPL'
        assert result.prediction == 'BUY'
        assert result.confidence == 0.8
        assert result.reasoning == 'Strong bullish momentum'
        assert result.price_target == 200.0
    
    def test_parse_text_response_sell(self):
        """测试解析卖出信号"""
        response = """
        Signal: SELL
        Confidence: 70%
        Reasoning: Bearish divergence
        """
        
        result = LLMResponseParser.parse_prediction_response(response, 'AAPL')
        
        assert result.prediction == 'SELL'
        assert result.confidence == 0.7
    
    def test_parse_text_response_hold(self):
        """测试解析持有信号"""
        response = """
        Signal: HOLD
        Confidence: 60%
        Reasoning: Mixed signals
        """
        
        result = LLMResponseParser.parse_prediction_response(response, 'AAPL')
        
        assert result.prediction == 'HOLD'
        assert result.confidence == 0.6
    
    def test_parse_invalid_json(self):
        """测试解析无效JSON"""
        response = "Invalid JSON response"
        
        result = LLMResponseParser.parse_prediction_response(response, 'AAPL')
        
        assert result.symbol == 'AAPL'
        assert result.prediction == 'HOLD'  # 默认值
        assert result.confidence == 0.5  # 默认值
    
    def test_parse_error_response(self):
        """测试解析错误响应"""
        response = None
        
        result = LLMResponseParser.parse_prediction_response(response, 'AAPL')
        
        assert result.symbol == 'AAPL'
        assert result.prediction == 'HOLD'
        assert result.confidence == 0.0


class TestPromptGenerator:
    """测试提示词生成器"""
    
    def test_init(self):
        """测试初始化"""
        generator = PromptGenerator()
        assert generator.system_prompt is not None
    
    def test_init_custom_system_prompt(self):
        """测试自定义系统提示词初始化"""
        custom_prompt = "You are a custom analyst."
        generator = PromptGenerator(custom_prompt)
        assert generator.system_prompt == custom_prompt
    
    def test_generate_prediction_prompt(self):
        """测试生成预测提示词"""
        import pandas as pd
        
        data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [99, 100, 101, 102, 103],
            'close': [104, 105, 106, 107, 108],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'rsi': [50, 55, 60, 65, 70],
            'macd': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        generator = PromptGenerator()
        prompt = generator.generate_prediction_prompt('AAPL', data)
        
        assert 'AAPL' in prompt
        assert 'Current Price' in prompt
        assert 'RSI:' in prompt
        assert 'MACD:' in prompt
        assert 'JSON format' in prompt
    
    def test_generate_prediction_prompt_with_context(self):
        """测试带上下文的预测提示词生成"""
        import pandas as pd
        
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        context = {'market_sentiment': 'bullish', 'news': 'positive earnings'}
        
        generator = PromptGenerator()
        prompt = generator.generate_prediction_prompt('AAPL', data, context)
        
        assert 'market_sentiment' in prompt
        assert 'positive earnings' in prompt
    
    def test_generate_market_analysis_prompt(self):
        """测试生成市场分析提示词"""
        import pandas as pd
        
        symbols = ['AAPL', 'MSFT']
        market_data = {
            'AAPL': pd.DataFrame({
                'close': [100, 101, 102],
                'volume': [1000, 1100, 1200]
            }),
            'MSFT': pd.DataFrame({
                'close': [200, 201, 202],
                'volume': [2000, 2100, 2200]
            })
        }
        
        generator = PromptGenerator()
        prompt = generator.generate_market_analysis_prompt(symbols, market_data)
        
        assert 'AAPL' in prompt
        assert 'MSFT' in prompt
        assert 'Current Price' in prompt
        assert '5-day Change' in prompt
        assert 'market_sentiment' in prompt


class TestLLMPredictor:
    """测试LLM预测器"""
    
    @pytest.fixture
    def mock_predictor(self):
        """模拟预测器"""
        with patch('src.models.llm_predictor.EnvironmentManager') as mock_env:
            mock_env.return_value.get_api_key.return_value = 'test_api_key'
            predictor = LLMPredictor(api_key='test_api_key')
            return predictor
    
    def test_init(self):
        """测试初始化"""
        with patch('src.models.llm_predictor.EnvironmentManager') as mock_env:
            mock_env.return_value.get_api_key.return_value = 'test_api_key'
            predictor = LLMPredictor(api_key='test_api_key')
            
            assert predictor.api_key == 'test_api_key'
            assert predictor.model is not None
            assert predictor.base_url is not None
            assert predictor.parser is not None
            assert predictor.prompt_generator is not None
    
    def test_init_no_api_key(self):
        """测试无API密钥初始化"""
        with patch('src.models.llm_predictor.EnvironmentManager') as mock_env:
            mock_env.return_value.get_api_key.return_value = None
            
            with pytest.raises(ValueError, match="API key not provided"):
                LLMPredictor()
    
    def test_init_with_config(self):
        """测试带配置初始化"""
        config = {
            'model': 'custom-model',
            'temperature': 0.5,
            'max_tokens': 1000
        }
        
        with patch('src.models.llm_predictor.EnvironmentManager') as mock_env:
            mock_env.return_value.get_api_key.return_value = 'test_api_key'
            predictor = LLMPredictor(api_key='test_api_key', config=config)
            
            assert predictor.model == 'custom-model'
            assert predictor.temperature == 0.5
            assert predictor.max_tokens == 1000
    
    @patch('requests.post')
    def test_call_llm_success(self, mock_post, mock_predictor):
        """测试成功调用LLM"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'Test response'}}]
        }
        mock_post.return_value = mock_response
        
        result = mock_predictor._call_llm('Test prompt', 'AAPL')
        
        assert result == 'Test response'
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_call_llm_api_error(self, mock_post, mock_predictor):
        """测试LLM API错误"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = 'Bad Request'
        mock_post.return_value = mock_response
        
        with pytest.raises(Exception, match="Failed to get LLM response"):
            mock_predictor._call_llm('Test prompt', 'AAPL')
    
    @patch('requests.post')
    def test_call_llm_network_error(self, mock_post, mock_predictor):
        """测试网络错误"""
        mock_post.side_effect = Exception("Network error")
        
        with pytest.raises(Exception, match="Failed to get LLM response"):
            mock_predictor._call_llm('Test prompt', 'AAPL')
    
    @patch.object(LLMPredictor, '_call_llm')
    def test_predict_single_symbol(self, mock_call_llm, mock_predictor):
        """测试预测单个符号"""
        import pandas as pd
        
        mock_call_llm.return_value = json.dumps({
            'prediction': 'BUY',
            'confidence': 0.8,
            'reasoning': 'Strong signal'
        })
        
        data = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })
        
        result = mock_predictor.predict('AAPL', data)
        
        assert result.symbol == 'AAPL'
        assert result.prediction == 'BUY'
        assert result.confidence == 0.8
        assert result.reasoning == 'Strong signal'
    
    @patch.object(LLMPredictor, '_call_llm')
    def test_predict_multiple_symbols(self, mock_call_llm, mock_predictor):
        """测试预测多个符号"""
        import pandas as pd
        
        mock_call_llm.return_value = json.dumps({
            'prediction': 'BUY',
            'confidence': 0.8,
            'reasoning': 'Strong signal'
        })
        
        data = {
            'AAPL': pd.DataFrame({'close': [100, 101, 102]}),
            'MSFT': pd.DataFrame({'close': [200, 201, 202]})
        }
        
        results = mock_predictor.predict_batch(data)
        
        assert len(results) == 2
        assert 'AAPL' in results
        assert 'MSFT' in results
        assert results['AAPL'].prediction == 'BUY'
        assert results['MSFT'].prediction == 'BUY'
    
    @patch.object(LLMPredictor, '_call_llm')
    def test_predict_error_handling(self, mock_call_llm, mock_predictor):
        """测试预测错误处理"""
        import pandas as pd
        
        mock_call_llm.side_effect = Exception("API Error")
        
        data = pd.DataFrame({'close': [100, 101, 102]})
        
        result = mock_predictor.predict('AAPL', data)
        
        assert result.symbol == 'AAPL'
        assert result.prediction == 'HOLD'  # 默认值
        assert result.confidence == 0.0
        assert 'Error' in result.reasoning
    
    def test_format_market_data(self, mock_predictor):
        """测试格式化市场数据"""
        import pandas as pd
        
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [104, 105, 106],
            'volume': [1000, 1100, 1200],
            'rsi': [50, 55, 60],
            'macd': [0.1, 0.2, 0.3]
        })
        
        result = mock_predictor._format_market_data('AAPL', data)
        
        assert 'AAPL' in result
        assert 'Current Price' in result
        assert 'RSI:' in result
        assert 'MACD:' in result
    
    def test_format_market_data_missing_indicators(self, mock_predictor):
        """测试格式化缺少技术指标的数据"""
        import pandas as pd
        
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [104, 105, 106],
            'volume': [1000, 1100, 1200]
        })
        
        result = mock_predictor._format_market_data('AAPL', data)
        
        assert 'AAPL' in result
        assert 'Current Price' in result
        # 应该不包含技术指标信息
    
    def test_get_prediction_summary(self, mock_predictor):
        """测试获取预测摘要"""
        predictions = {
            'AAPL': PredictionResult('AAPL', 'BUY', 0.8, 'Strong signal'),
            'MSFT': PredictionResult('MSFT', 'SELL', 0.7, 'Weak signal'),
            'GOOGL': PredictionResult('GOOGL', 'HOLD', 0.6, 'Neutral signal')
        }
        
        summary = mock_predictor.get_prediction_summary(predictions)
        
        assert summary['total_symbols'] == 3
        assert summary['buy_signals'] == 1
        assert summary['sell_signals'] == 1
        assert summary['hold_signals'] == 1
        assert summary['avg_confidence'] == (0.8 + 0.7 + 0.6) / 3
    
    def test_get_prediction_summary_empty(self, mock_predictor):
        """测试获取空预测摘要"""
        summary = mock_predictor.get_prediction_summary({})
        
        assert summary['total_symbols'] == 0
        assert summary['buy_signals'] == 0
        assert summary['sell_signals'] == 0
        assert summary['hold_signals'] == 0
        assert summary['avg_confidence'] == 0.0
