"""
AI Trading System - 模型管理器模块单元测试

测试ModelManager类的各种功能，包括模型管理、预测协调、结果聚合等。
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.models.model_manager import ModelManager
from src.models.llm_predictor import PredictionResult


class TestModelManager:
    """测试模型管理器"""
    
    @pytest.fixture
    def model_manager(self):
        """模型管理器实例"""
        return ModelManager()
    
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
    
    def test_init(self, model_manager):
        """测试初始化"""
        assert model_manager.models == {}
        assert model_manager.active_model is None
        assert model_manager.config == {}
        assert model_manager.predictions == {}
        assert model_manager.performance_metrics == {}
    
    def test_init_with_config(self):
        """测试带配置初始化"""
        config = {
            'default_model': 'llm',
            'models': {
                'llm': {'type': 'llm', 'config': {}},
                'ml': {'type': 'ml', 'config': {}}
            }
        }
        
        manager = ModelManager(config)
        
        assert manager.config == config
        assert manager.active_model == 'llm'
    
    def test_register_model(self, model_manager):
        """测试注册模型"""
        mock_model = Mock()
        mock_model.name = 'test_model'
        mock_model.model_type = 'llm'
        
        model_manager.register_model('test_model', mock_model)
        
        assert 'test_model' in model_manager.models
        assert model_manager.models['test_model'] == mock_model
    
    def test_register_model_duplicate(self, model_manager):
        """测试注册重复模型"""
        mock_model = Mock()
        mock_model.name = 'test_model'
        
        model_manager.register_model('test_model', mock_model)
        
        with pytest.raises(ValueError, match="Model 'test_model' already registered"):
            model_manager.register_model('test_model', mock_model)
    
    def test_set_active_model(self, model_manager):
        """测试设置活动模型"""
        mock_model = Mock()
        model_manager.register_model('test_model', mock_model)
        
        model_manager.set_active_model('test_model')
        
        assert model_manager.active_model == 'test_model'
    
    def test_set_active_model_not_found(self, model_manager):
        """测试设置不存在的活动模型"""
        with pytest.raises(ValueError, match="Model 'nonexistent' not found"):
            model_manager.set_active_model('nonexistent')
    
    def test_get_model(self, model_manager):
        """测试获取模型"""
        mock_model = Mock()
        model_manager.register_model('test_model', mock_model)
        
        retrieved_model = model_manager.get_model('test_model')
        
        assert retrieved_model == mock_model
    
    def test_get_model_not_found(self, model_manager):
        """测试获取不存在的模型"""
        with pytest.raises(ValueError, match="Model 'nonexistent' not found"):
            model_manager.get_model('nonexistent')
    
    def test_list_models(self, model_manager):
        """测试列出模型"""
        mock_model1 = Mock()
        mock_model1.name = 'model1'
        mock_model1.model_type = 'llm'
        
        mock_model2 = Mock()
        mock_model2.name = 'model2'
        mock_model2.model_type = 'ml'
        
        model_manager.register_model('model1', mock_model1)
        model_manager.register_model('model2', mock_model2)
        
        models = model_manager.list_models()
        
        assert len(models) == 2
        assert 'model1' in models
        assert 'model2' in models
        assert models['model1']['type'] == 'llm'
        assert models['model2']['type'] == 'ml'
    
    @patch('src.models.model_manager.LLMPredictor')
    def test_create_llm_predictor(self, mock_llm_class, model_manager):
        """测试创建LLM预测器"""
        mock_predictor = Mock()
        mock_llm_class.return_value = mock_predictor
        
        config = {'model': 'test-model', 'temperature': 0.5}
        predictor = model_manager.create_llm_predictor(config)
        
        assert predictor == mock_predictor
        mock_llm_class.assert_called_once_with(config=config)
    
    @patch('src.models.model_manager.MLPredictor')
    def test_create_ml_predictor(self, mock_ml_class, model_manager):
        """测试创建ML预测器"""
        mock_predictor = Mock()
        mock_ml_class.return_value = mock_predictor
        
        config = {'model_type': 'random_forest', 'n_estimators': 100}
        predictor = model_manager.create_ml_predictor(config)
        
        assert predictor == mock_predictor
        mock_ml_class.assert_called_once_with(config=config)
    
    def test_predict_single_symbol(self, model_manager, sample_market_data):
        """测试预测单个符号"""
        mock_model = Mock()
        mock_prediction = PredictionResult('AAPL', 'BUY', 0.8, 'Strong signal')
        mock_model.predict.return_value = mock_prediction
        
        model_manager.register_model('test_model', mock_model)
        model_manager.set_active_model('test_model')
        
        result = model_manager.predict('AAPL', sample_market_data['AAPL'])
        
        assert result.symbol == 'AAPL'
        assert result.prediction == 'BUY'
        assert result.confidence == 0.8
        mock_model.predict.assert_called_once_with('AAPL', sample_market_data['AAPL'])
    
    def test_predict_single_symbol_no_active_model(self, model_manager, sample_market_data):
        """测试无活动模型时预测"""
        with pytest.raises(ValueError, match="No active model set"):
            model_manager.predict('AAPL', sample_market_data['AAPL'])
    
    def test_predict_batch(self, model_manager, sample_market_data):
        """测试批量预测"""
        mock_model = Mock()
        mock_predictions = {
            'AAPL': PredictionResult('AAPL', 'BUY', 0.8, 'Strong signal'),
            'MSFT': PredictionResult('MSFT', 'SELL', 0.7, 'Weak signal')
        }
        mock_model.predict_batch.return_value = mock_predictions
        
        model_manager.register_model('test_model', mock_model)
        model_manager.set_active_model('test_model')
        
        result = model_manager.predict_batch(sample_market_data)
        
        assert len(result) == 2
        assert 'AAPL' in result
        assert 'MSFT' in result
        assert result['AAPL'].prediction == 'BUY'
        assert result['MSFT'].prediction == 'SELL'
        mock_model.predict_batch.assert_called_once_with(sample_market_data)
    
    def test_predict_batch_error_handling(self, model_manager, sample_market_data):
        """测试批量预测错误处理"""
        mock_model = Mock()
        mock_model.predict_batch.side_effect = Exception("Prediction error")
        
        model_manager.register_model('test_model', mock_model)
        model_manager.set_active_model('test_model')
        
        result = model_manager.predict_batch(sample_market_data)
        
        assert len(result) == 0  # 错误时应该返回空字典
    
    def test_get_prediction_summary(self, model_manager, sample_predictions):
        """测试获取预测摘要"""
        model_manager.predictions = sample_predictions
        
        summary = model_manager.get_prediction_summary()
        
        assert summary['total_symbols'] == 3
        assert summary['buy_signals'] == 1
        assert summary['sell_signals'] == 1
        assert summary['hold_signals'] == 1
        assert summary['avg_confidence'] == (0.8 + 0.7 + 0.6) / 3
    
    def test_get_prediction_summary_empty(self, model_manager):
        """测试获取空预测摘要"""
        summary = model_manager.get_prediction_summary()
        
        assert summary['total_symbols'] == 0
        assert summary['buy_signals'] == 0
        assert summary['sell_signals'] == 0
        assert summary['hold_signals'] == 0
        assert summary['avg_confidence'] == 0.0
    
    def test_update_performance_metrics(self, model_manager):
        """测试更新性能指标"""
        metrics = {
            'accuracy': 0.75,
            'precision': 0.80,
            'recall': 0.70,
            'f1_score': 0.75
        }
        
        model_manager.update_performance_metrics('test_model', metrics)
        
        assert 'test_model' in model_manager.performance_metrics
        assert model_manager.performance_metrics['test_model'] == metrics
    
    def test_get_performance_metrics(self, model_manager):
        """测试获取性能指标"""
        metrics = {
            'accuracy': 0.75,
            'precision': 0.80,
            'recall': 0.70,
            'f1_score': 0.75
        }
        
        model_manager.update_performance_metrics('test_model', metrics)
        
        retrieved_metrics = model_manager.get_performance_metrics('test_model')
        
        assert retrieved_metrics == metrics
    
    def test_get_performance_metrics_not_found(self, model_manager):
        """测试获取不存在的性能指标"""
        with pytest.raises(ValueError, match="No performance metrics found for model 'nonexistent'"):
            model_manager.get_performance_metrics('nonexistent')
    
    def test_compare_models(self, model_manager):
        """测试比较模型"""
        metrics1 = {'accuracy': 0.75, 'precision': 0.80, 'recall': 0.70}
        metrics2 = {'accuracy': 0.80, 'precision': 0.85, 'recall': 0.75}
        
        model_manager.update_performance_metrics('model1', metrics1)
        model_manager.update_performance_metrics('model2', metrics2)
        
        comparison = model_manager.compare_models(['model1', 'model2'])
        
        assert 'model1' in comparison
        assert 'model2' in comparison
        assert comparison['model1']['accuracy'] == 0.75
        assert comparison['model2']['accuracy'] == 0.80
    
    def test_export_predictions(self, model_manager, sample_predictions, tmp_path):
        """测试导出预测结果"""
        model_manager.predictions = sample_predictions
        
        output_file = tmp_path / 'test_predictions.json'
        result = model_manager.export_predictions(str(output_file))
        
        assert result == str(output_file)
        assert output_file.exists()
        
        # 验证文件内容
        import json
        with open(output_file) as f:
            data = json.load(f)
            assert 'AAPL' in data
            assert data['AAPL']['prediction'] == 'BUY'
    
    def test_export_performance_metrics(self, model_manager, tmp_path):
        """测试导出性能指标"""
        metrics = {
            'model1': {'accuracy': 0.75, 'precision': 0.80},
            'model2': {'accuracy': 0.80, 'precision': 0.85}
        }
        
        for model, metric in metrics.items():
            model_manager.update_performance_metrics(model, metric)
        
        output_file = tmp_path / 'test_metrics.json'
        result = model_manager.export_performance_metrics(str(output_file))
        
        assert result == str(output_file)
        assert output_file.exists()
        
        # 验证文件内容
        import json
        with open(output_file) as f:
            data = json.load(f)
            assert 'model1' in data
            assert 'model2' in data
            assert data['model1']['accuracy'] == 0.75
    
    def test_get_model_status(self, model_manager):
        """测试获取模型状态"""
        mock_model = Mock()
        mock_model.is_trained = True
        mock_model.model_type = 'llm'
        
        model_manager.register_model('test_model', mock_model)
        model_manager.set_active_model('test_model')
        
        status = model_manager.get_model_status()
        
        assert status['active_model'] == 'test_model'
        assert status['total_models'] == 1
        assert status['models']['test_model']['type'] == 'llm'
        assert status['models']['test_model']['trained'] == True
    
    def test_validate_model_config(self, model_manager):
        """测试验证模型配置"""
        valid_config = {
            'model_type': 'llm',
            'model': 'test-model',
            'temperature': 0.5,
            'max_tokens': 1000
        }
        
        is_valid, errors = model_manager.validate_model_config(valid_config)
        
        assert is_valid == True
        assert len(errors) == 0
    
    def test_validate_model_config_invalid(self, model_manager):
        """测试验证无效模型配置"""
        invalid_config = {
            'model_type': 'invalid_type',
            'temperature': 2.0,  # 超出范围
            'max_tokens': -1  # 无效值
        }
        
        is_valid, errors = model_manager.validate_model_config(invalid_config)
        
        assert is_valid == False
        assert len(errors) > 0
        assert any('temperature' in error for error in errors)
        assert any('max_tokens' in error for error in errors)
    
    def test_cleanup_models(self, model_manager):
        """测试清理模型"""
        mock_model1 = Mock()
        mock_model2 = Mock()
        
        model_manager.register_model('model1', mock_model1)
        model_manager.register_model('model2', mock_model2)
        
        model_manager.cleanup_models()
        
        assert len(model_manager.models) == 0
        assert model_manager.active_model is None
        assert len(model_manager.predictions) == 0
        assert len(model_manager.performance_metrics) == 0
