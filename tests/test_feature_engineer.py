"""
AI Trading System - 特征工程模块单元测试

测试FeatureEngineer类的各种功能，包括特征生成、数据转换、特征选择等。
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.data.feature_engineer import FeatureEngineer


class TestFeatureEngineer:
    """测试特征工程器"""
    
    @pytest.fixture
    def sample_data(self):
        """样本数据"""
        return pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'close': [104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        }, index=pd.date_range('2023-01-01', periods=10))
    
    @pytest.fixture
    def feature_engineer(self):
        """特征工程器实例"""
        return FeatureEngineer()
    
    def test_init(self, feature_engineer):
        """测试初始化"""
        assert feature_engineer.config == {}
        assert feature_engineer.feature_columns == []
        assert feature_engineer.target_columns == []
    
    def test_init_with_config(self):
        """测试带配置初始化"""
        config = {
            'feature_columns': ['close', 'volume'],
            'target_columns': ['returns'],
            'lookback_periods': [5, 10, 20]
        }
        
        engineer = FeatureEngineer(config)
        
        assert engineer.config == config
        assert engineer.feature_columns == ['close', 'volume']
        assert engineer.target_columns == ['returns']
    
    def test_add_price_features(self, feature_engineer, sample_data):
        """测试添加价格特征"""
        result = feature_engineer.add_price_features(sample_data)
        
        assert 'returns' in result.columns
        assert 'log_returns' in result.columns
        assert 'price_change' in result.columns
        assert 'price_change_pct' in result.columns
        
        # 检查返回率计算
        expected_returns = sample_data['close'].pct_change()
        pd.testing.assert_series_equal(result['returns'], expected_returns)
    
    def test_add_volume_features(self, feature_engineer, sample_data):
        """测试添加成交量特征"""
        result = feature_engineer.add_volume_features(sample_data)
        
        assert 'volume_change' in result.columns
        assert 'volume_change_pct' in result.columns
        assert 'volume_ma_5' in result.columns
        assert 'volume_ma_10' in result.columns
        assert 'volume_ratio' in result.columns
    
    def test_add_volatility_features(self, feature_engineer, sample_data):
        """测试添加波动率特征"""
        result = feature_engineer.add_volatility_features(sample_data)
        
        assert 'volatility_5' in result.columns
        assert 'volatility_10' in result.columns
        assert 'volatility_20' in result.columns
        assert 'volatility_ratio' in result.columns
        
        # 检查波动率计算
        assert result['volatility_5'].iloc[-1] > 0
        assert result['volatility_10'].iloc[-1] > 0
    
    def test_add_momentum_features(self, feature_engineer, sample_data):
        """测试添加动量特征"""
        result = feature_engineer.add_momentum_features(sample_data)
        
        assert 'momentum_5' in result.columns
        assert 'momentum_10' in result.columns
        assert 'momentum_20' in result.columns
        assert 'momentum_ratio' in result.columns
        
        # 检查动量计算
        expected_momentum_5 = (sample_data['close'] / sample_data['close'].shift(5) - 1) * 100
        pd.testing.assert_series_equal(result['momentum_5'], expected_momentum_5)
    
    def test_add_technical_features(self, feature_engineer, sample_data):
        """测试添加技术指标特征"""
        # 添加技术指标列
        sample_data['rsi'] = np.random.uniform(0, 100, len(sample_data))
        sample_data['macd'] = np.random.uniform(-1, 1, len(sample_data))
        sample_data['bb_upper'] = sample_data['close'] * 1.02
        sample_data['bb_lower'] = sample_data['close'] * 0.98
        
        result = feature_engineer.add_technical_features(sample_data)
        
        assert 'rsi_signal' in result.columns
        assert 'macd_signal' in result.columns
        assert 'bb_position' in result.columns
        assert 'bb_squeeze' in result.columns
        
        # 检查RSI信号
        assert all(result['rsi_signal'].isin([-1, 0, 1]))
    
    def test_add_time_features(self, feature_engineer, sample_data):
        """测试添加时间特征"""
        result = feature_engineer.add_time_features(sample_data)
        
        assert 'day_of_week' in result.columns
        assert 'month' in result.columns
        assert 'quarter' in result.columns
        assert 'is_month_end' in result.columns
        assert 'is_quarter_end' in result.columns
        
        # 检查时间特征值
        assert all(result['day_of_week'].isin(range(7)))
        assert all(result['month'].isin(range(1, 13)))
        assert all(result['quarter'].isin(range(1, 5)))
        assert all(result['is_month_end'].isin([True, False]))
    
    def test_add_lag_features(self, feature_engineer, sample_data):
        """测试添加滞后特征"""
        result = feature_engineer.add_lag_features(sample_data, ['close', 'volume'], lags=[1, 2, 3])
        
        assert 'close_lag_1' in result.columns
        assert 'close_lag_2' in result.columns
        assert 'close_lag_3' in result.columns
        assert 'volume_lag_1' in result.columns
        assert 'volume_lag_2' in result.columns
        assert 'volume_lag_3' in result.columns
        
        # 检查滞后特征值
        assert result['close_lag_1'].iloc[1] == sample_data['close'].iloc[0]
        assert result['close_lag_2'].iloc[2] == sample_data['close'].iloc[0]
    
    def test_add_rolling_features(self, feature_engineer, sample_data):
        """测试添加滚动特征"""
        result = feature_engineer.add_rolling_features(sample_data, ['close'], windows=[5, 10])
        
        assert 'close_rolling_mean_5' in result.columns
        assert 'close_rolling_mean_10' in result.columns
        assert 'close_rolling_std_5' in result.columns
        assert 'close_rolling_std_10' in result.columns
        assert 'close_rolling_max_5' in result.columns
        assert 'close_rolling_min_5' in result.columns
        
        # 检查滚动特征计算
        expected_mean_5 = sample_data['close'].rolling(5).mean()
        pd.testing.assert_series_equal(result['close_rolling_mean_5'], expected_mean_5)
    
    def test_add_correlation_features(self, feature_engineer, sample_data):
        """测试添加相关性特征"""
        # 添加多个价格列
        sample_data['close_2'] = sample_data['close'] * 1.1
        sample_data['close_3'] = sample_data['close'] * 0.9
        
        result = feature_engineer.add_correlation_features(sample_data, ['close', 'close_2', 'close_3'], window=5)
        
        assert 'close_close_2_corr_5' in result.columns
        assert 'close_close_3_corr_5' in result.columns
        assert 'close_2_close_3_corr_5' in result.columns
        
        # 检查相关性特征值
        assert all(result['close_close_2_corr_5'].dropna().between(-1, 1))
    
    def test_engineer_features_comprehensive(self, feature_engineer, sample_data):
        """测试综合特征工程"""
        # 添加技术指标
        sample_data['rsi'] = np.random.uniform(0, 100, len(sample_data))
        sample_data['macd'] = np.random.uniform(-1, 1, len(sample_data))
        
        result = feature_engineer.engineer_features(sample_data)
        
        # 检查各种特征类型
        assert 'returns' in result.columns
        assert 'volume_change' in result.columns
        assert 'volatility_5' in result.columns
        assert 'momentum_5' in result.columns
        assert 'day_of_week' in result.columns
        assert 'close_lag_1' in result.columns
        assert 'close_rolling_mean_5' in result.columns
        
        # 检查数据完整性
        assert not result.empty
        assert len(result) == len(sample_data)
    
    def test_engineer_features_with_config(self, sample_data):
        """测试带配置的特征工程"""
        config = {
            'price_features': True,
            'volume_features': True,
            'volatility_features': True,
            'momentum_features': True,
            'technical_features': False,
            'time_features': True,
            'lag_features': {'columns': ['close'], 'lags': [1, 2]},
            'rolling_features': {'columns': ['close'], 'windows': [5, 10]}
        }
        
        engineer = FeatureEngineer(config)
        result = engineer.engineer_features(sample_data)
        
        assert 'returns' in result.columns
        assert 'volume_change' in result.columns
        assert 'volatility_5' in result.columns
        assert 'momentum_5' in result.columns
        assert 'day_of_week' in result.columns
        assert 'close_lag_1' in result.columns
        assert 'close_rolling_mean_5' in result.columns
    
    def test_select_features(self, feature_engineer, sample_data):
        """测试特征选择"""
        # 添加一些特征
        sample_data['returns'] = sample_data['close'].pct_change()
        sample_data['volume_change'] = sample_data['volume'].pct_change()
        sample_data['volatility_5'] = sample_data['returns'].rolling(5).std()
        sample_data['momentum_5'] = (sample_data['close'] / sample_data['close'].shift(5) - 1) * 100
        
        # 选择特征
        selected_features = ['close', 'volume', 'returns', 'volume_change']
        result = feature_engineer.select_features(sample_data, selected_features)
        
        assert set(result.columns) == set(selected_features)
        assert len(result) == len(sample_data)
    
    def test_select_features_missing_columns(self, feature_engineer, sample_data):
        """测试选择缺失列的特征"""
        selected_features = ['close', 'volume', 'missing_column']
        
        with pytest.raises(ValueError, match="Missing columns"):
            feature_engineer.select_features(sample_data, selected_features)
    
    def test_get_feature_importance(self, feature_engineer, sample_data):
        """测试获取特征重要性"""
        # 添加特征
        sample_data['returns'] = sample_data['close'].pct_change()
        sample_data['volume_change'] = sample_data['volume'].pct_change()
        sample_data['volatility_5'] = sample_data['returns'].rolling(5).std()
        
        # 模拟特征重要性
        importance = feature_engineer.get_feature_importance(sample_data, ['close', 'volume', 'returns'])
        
        assert len(importance) == 3
        assert all(importance.values >= 0)
        assert all(importance.values <= 1)
    
    def test_export_features(self, feature_engineer, sample_data, tmp_path):
        """测试导出特征"""
        # 添加特征
        sample_data['returns'] = sample_data['close'].pct_change()
        sample_data['volume_change'] = sample_data['volume'].pct_change()
        
        output_file = tmp_path / 'test_features.csv'
        result = feature_engineer.export_features(sample_data, str(output_file))
        
        assert result == str(output_file)
        assert output_file.exists()
        
        # 验证文件内容
        loaded_data = pd.read_csv(output_file, index_col=0, parse_dates=True)
        assert len(loaded_data) == len(sample_data)
        assert 'returns' in loaded_data.columns
        assert 'volume_change' in loaded_data.columns
    
    def test_get_feature_summary(self, feature_engineer, sample_data):
        """测试获取特征摘要"""
        # 添加特征
        sample_data['returns'] = sample_data['close'].pct_change()
        sample_data['volume_change'] = sample_data['volume'].pct_change()
        sample_data['volatility_5'] = sample_data['returns'].rolling(5).std()
        
        summary = feature_engineer.get_feature_summary(sample_data)
        
        assert 'total_features' in summary
        assert 'feature_types' in summary
        assert 'missing_values' in summary
        assert 'feature_correlation' in summary
        
        assert summary['total_features'] == len(sample_data.columns)
        assert summary['missing_values']['returns'] > 0  # 第一个值应该是NaN
    
    def test_validate_features(self, feature_engineer, sample_data):
        """测试特征验证"""
        # 添加特征
        sample_data['returns'] = sample_data['close'].pct_change()
        sample_data['volume_change'] = sample_data['volume'].pct_change()
        
        is_valid, errors = feature_engineer.validate_features(sample_data)
        
        assert is_valid == True
        assert len(errors) == 0
    
    def test_validate_features_with_errors(self, feature_engineer, sample_data):
        """测试特征验证错误"""
        # 添加有问题的特征
        sample_data['returns'] = sample_data['close'].pct_change()
        sample_data['invalid_feature'] = [np.nan] * len(sample_data)  # 全NaN列
        
        is_valid, errors = feature_engineer.validate_features(sample_data)
        
        assert is_valid == False
        assert len(errors) > 0
        assert any('invalid_feature' in error for error in errors)
    
    def test_engineer_features_batch(self, feature_engineer):
        """测试批量特征工程"""
        data = {
            'AAPL': pd.DataFrame({
                'close': [100, 101, 102, 103, 104],
                'volume': [1000, 1100, 1200, 1300, 1400]
            }),
            'MSFT': pd.DataFrame({
                'close': [200, 201, 202, 203, 204],
                'volume': [2000, 2100, 2200, 2300, 2400]
            })
        }
        
        result = feature_engineer.engineer_features_batch(data)
        
        assert len(result) == 2
        assert 'AAPL' in result
        assert 'MSFT' in result
        
        # 检查每个符号的特征
        for symbol, df in result.items():
            assert 'returns' in df.columns
            assert 'volume_change' in df.columns
            assert len(df) == 5
