"""
AI Trading System - 数据处理模块单元测试

测试DataProcessor、DataValidator、DataCleaner类的各种功能。
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.data.data_processor import DataProcessor, DataValidator, DataCleaner


class TestDataValidator:
    """测试数据验证器"""
    
    def test_validate_ohlcv_valid_data(self):
        """测试验证有效的OHLCV数据"""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [104, 105, 106],
            'volume': [1000, 1100, 1200]
        })
        
        is_valid, errors = DataValidator.validate_ohlcv(df, 'AAPL')
        assert is_valid == True
        assert len(errors) == 0
    
    def test_validate_ohlcv_missing_columns(self):
        """测试验证缺少列的数据"""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'close': [104, 105, 106]
        })
        
        is_valid, errors = DataValidator.validate_ohlcv(df, 'AAPL')
        assert is_valid == False
        assert any('Missing columns' in error for error in errors)
    
    def test_validate_ohlcv_empty_dataframe(self):
        """测试验证空DataFrame"""
        df = pd.DataFrame()
        
        is_valid, errors = DataValidator.validate_ohlcv(df, 'AAPL')
        assert is_valid == False
        assert any('Empty DataFrame' in error for error in errors)
    
    def test_validate_ohlcv_invalid_prices(self):
        """测试验证无效价格"""
        df = pd.DataFrame({
            'open': [100, -101, 102],  # 负价格
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [104, 105, 106],
            'volume': [1000, 1100, 1200]
        })
        
        is_valid, errors = DataValidator.validate_ohlcv(df, 'AAPL')
        assert is_valid == False
        assert any('Invalid open prices' in error for error in errors)
    
    def test_validate_ohlcv_ohlc_logic_violation(self):
        """测试验证OHLC逻辑违规"""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [99, 100, 101],  # high < low
            'low': [99, 100, 101],
            'close': [104, 105, 106],
            'volume': [1000, 1100, 1200]
        })
        
        is_valid, errors = DataValidator.validate_ohlcv(df, 'AAPL')
        assert is_valid == False
        assert any('High < Low prices' in error for error in errors)
    
    def test_validate_ohlcv_negative_volume(self):
        """测试验证负成交量"""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [104, 105, 106],
            'volume': [1000, -1100, 1200]  # 负成交量
        })
        
        is_valid, errors = DataValidator.validate_ohlcv(df, 'AAPL')
        assert is_valid == False
        assert any('Negative volume' in error for error in errors)
    
    def test_validate_time_series_valid(self):
        """测试验证有效时间序列"""
        df = pd.DataFrame({
            'close': [100, 101, 102]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        is_valid, errors = DataValidator.validate_time_series(df)
        assert is_valid == True
        assert len(errors) == 0
    
    def test_validate_time_series_not_datetime_index(self):
        """测试验证非时间索引"""
        df = pd.DataFrame({
            'close': [100, 101, 102]
        }, index=[0, 1, 2])
        
        is_valid, errors = DataValidator.validate_time_series(df)
        assert is_valid == False
        assert any('Index is not DatetimeIndex' in error for error in errors)
    
    def test_validate_time_series_duplicate_timestamps(self):
        """测试验证重复时间戳"""
        df = pd.DataFrame({
            'close': [100, 101, 102]
        }, index=pd.DatetimeIndex(['2023-01-01', '2023-01-01', '2023-01-02']))
        
        is_valid, errors = DataValidator.validate_time_series(df)
        assert is_valid == False
        assert any('Duplicate timestamps' in error for error in errors)


class TestDataCleaner:
    """测试数据清洗器"""
    
    def test_remove_outliers_iqr(self):
        """测试IQR方法移除异常值"""
        df = pd.DataFrame({
            'open': [100, 101, 102, 200, 103],  # 200是异常值
            'close': [104, 105, 106, 107, 108]
        })
        
        result = DataCleaner.remove_outliers(df, method='iqr', threshold=1.5)
        
        assert len(result) < len(df)
        assert 200 not in result['open'].values
    
    def test_remove_outliers_zscore(self):
        """测试Z-score方法移除异常值"""
        df = pd.DataFrame({
            'open': [100, 101, 102, 200, 103],  # 200是异常值
            'close': [104, 105, 106, 107, 108]
        })
        
        result = DataCleaner.remove_outliers(df, method='zscore', threshold=2.0)
        
        assert len(result) < len(df)
        assert 200 not in result['open'].values
    
    def test_fill_missing_values_forward(self):
        """测试前向填充缺失值"""
        df = pd.DataFrame({
            'open': [100, np.nan, 102, np.nan, 104],
            'close': [104, 105, np.nan, 107, 108]
        })
        
        result = DataCleaner.fill_missing_values(df, method='forward')
        
        assert not result.isnull().any().any()
        assert result['open'].iloc[1] == 100  # 前向填充
        assert result['close'].iloc[2] == 105  # 前向填充
    
    def test_fill_missing_values_backward(self):
        """测试后向填充缺失值"""
        df = pd.DataFrame({
            'open': [100, np.nan, 102, np.nan, 104],
            'close': [104, 105, np.nan, 107, 108]
        })
        
        result = DataCleaner.fill_missing_values(df, method='backward')
        
        assert not result.isnull().any().any()
    
    def test_fill_missing_values_interpolate(self):
        """测试插值填充缺失值"""
        df = pd.DataFrame({
            'open': [100, np.nan, 102, np.nan, 104],
            'close': [104, 105, np.nan, 107, 108]
        })
        
        result = DataCleaner.fill_missing_values(df, method='interpolate')
        
        assert not result.isnull().any().any()
    
    def test_remove_duplicates(self):
        """测试移除重复数据"""
        df = pd.DataFrame({
            'open': [100, 101, 101, 102],
            'close': [104, 105, 105, 106]
        }, index=pd.DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03']))
        
        result = DataCleaner.remove_duplicates(df)
        
        assert len(result) == 3
        assert not result.index.duplicated().any()
    
    def test_filter_volume(self):
        """测试过滤成交量"""
        df = pd.DataFrame({
            'volume': [500, 1000, 1500, 2000],
            'close': [100, 101, 102, 103]
        })
        
        result = DataCleaner.filter_volume(df, min_volume=1000)
        
        assert len(result) == 3
        assert (result['volume'] >= 1000).all()
    
    def test_filter_volume_no_volume_column(self):
        """测试无成交量列时的过滤"""
        df = pd.DataFrame({
            'close': [100, 101, 102, 103]
        })
        
        result = DataCleaner.filter_volume(df, min_volume=1000)
        
        assert len(result) == len(df)  # 应该返回原DataFrame


class TestDataProcessor:
    """测试数据处理器"""
    
    @pytest.fixture
    def sample_data(self):
        """样本数据"""
        return pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [99, 100, 101, 102, 103],
            'close': [104, 105, 106, 107, 108],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }, index=pd.date_range('2023-01-01', periods=5))
    
    @pytest.fixture
    def processor(self):
        """数据处理器实例"""
        return DataProcessor()
    
    def test_init(self):
        """测试初始化"""
        processor = DataProcessor()
        assert processor.validator is not None
        assert processor.cleaner is not None
        assert processor.config == {}
    
    def test_init_with_config(self):
        """测试带配置初始化"""
        config = {'outlier_method': 'iqr'}
        processor = DataProcessor(config)
        assert processor.config == config
    
    def test_clean_data_valid(self, processor, sample_data):
        """测试清洗有效数据"""
        result = processor.clean_data(sample_data, 'AAPL')
        
        assert not result.empty
        assert len(result) <= len(sample_data)
        assert 'open' in result.columns
        assert 'close' in result.columns
    
    def test_clean_data_with_missing_values(self, processor):
        """测试清洗有缺失值的数据"""
        df = pd.DataFrame({
            'open': [100, np.nan, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [99, 100, 101, 102, 103],
            'close': [104, 105, 106, 107, 108],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        result = processor.clean_data(df, 'AAPL')
        
        assert not result.empty
        assert not result.isnull().any().any()
    
    def test_clean_data_empty(self, processor):
        """测试清洗空数据"""
        df = pd.DataFrame()
        
        result = processor.clean_data(df, 'AAPL')
        
        assert result.empty
    
    @patch('src.data.data_processor.add_all_technical_indicators')
    def test_add_technical_indicators_success(self, mock_add_indicators, processor, sample_data):
        """测试成功添加技术指标"""
        mock_add_indicators.return_value = sample_data.assign(rsi=50, macd=0.1)
        
        result = processor.add_technical_indicators(sample_data)
        
        assert 'rsi' in result.columns
        assert 'macd' in result.columns
        mock_add_indicators.assert_called_once_with(sample_data)
    
    @patch('src.data.data_processor.add_all_technical_indicators')
    def test_add_technical_indicators_error(self, mock_add_indicators, processor, sample_data):
        """测试添加技术指标时出错"""
        mock_add_indicators.side_effect = Exception("Indicator error")
        
        result = processor.add_technical_indicators(sample_data)
        
        assert result.equals(sample_data)  # 应该返回原数据
    
    def test_process_data_single_symbol(self, processor, sample_data):
        """测试处理单个符号数据"""
        data = {'AAPL': sample_data}
        
        with patch.object(processor, 'add_technical_indicators') as mock_add_indicators:
            mock_add_indicators.return_value = sample_data.assign(rsi=50)
            
            result = processor.process_data(data)
            
            assert 'AAPL' in result
            assert not result['AAPL'].empty
            assert 'rsi' in result['AAPL'].columns
    
    def test_process_data_multiple_symbols(self, processor):
        """测试处理多个符号数据"""
        data = {
            'AAPL': pd.DataFrame({
                'open': [100, 101], 'high': [105, 106], 'low': [99, 100],
                'close': [104, 105], 'volume': [1000, 1100]
            }),
            'MSFT': pd.DataFrame({
                'open': [200, 201], 'high': [205, 206], 'low': [199, 200],
                'close': [204, 205], 'volume': [2000, 2100]
            })
        }
        
        with patch.object(processor, 'add_technical_indicators') as mock_add_indicators:
            mock_add_indicators.return_value = pd.DataFrame({'close': [100, 101]})
            
            result = processor.process_data(data)
            
            assert len(result) == 2
            assert 'AAPL' in result
            assert 'MSFT' in result
    
    def test_process_data_empty_symbol(self, processor):
        """测试处理空符号数据"""
        data = {'EMPTY': pd.DataFrame()}
        
        result = processor.process_data(data)
        
        assert len(result) == 0  # 空数据应该被跳过
    
    def test_process_data_error_handling(self, processor):
        """测试处理数据时的错误处理"""
        data = {'ERROR': pd.DataFrame({'invalid': [1, 2, 3]})}
        
        result = processor.process_data(data)
        
        assert len(result) == 0  # 错误数据应该被跳过
    
    def test_get_data_summary(self, processor, sample_data):
        """测试获取数据摘要"""
        data = {'AAPL': sample_data}
        
        summary = processor.get_data_summary(data)
        
        assert summary['total_symbols'] == 1
        assert 'AAPL' in summary['symbols']
        assert 'AAPL' in summary['date_ranges']
        assert 'AAPL' in summary['data_quality']
        assert summary['date_ranges']['AAPL']['days'] == 5
    
    def test_get_data_summary_empty_data(self, processor):
        """测试获取空数据摘要"""
        data = {}
        
        summary = processor.get_data_summary(data)
        
        assert summary['total_symbols'] == 0
        assert summary['symbols'] == []
    
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_export_processed_data(self, mock_to_csv, mock_makedirs, processor, sample_data):
        """测试导出处理后的数据"""
        data = {'AAPL': sample_data}
        
        result = processor.export_processed_data(data, 'test_output')
        
        assert 'AAPL' in result
        assert result['AAPL'].endswith('.csv')
        mock_makedirs.assert_called_once_with('test_output', exist_ok=True)
        mock_to_csv.assert_called_once()
    
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_export_processed_data_error(self, mock_to_csv, mock_makedirs, processor, sample_data):
        """测试导出数据时出错"""
        mock_to_csv.side_effect = Exception("Export error")
        data = {'AAPL': sample_data}
        
        result = processor.export_processed_data(data, 'test_output')
        
        assert len(result) == 0  # 错误时应该返回空字典
