"""
AI Trading System - 数据获取模块单元测试

测试DataFetcher类的各种功能，包括数据获取、缓存、多数据源等。
"""

import pytest
import pandas as pd
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.data.data_fetcher import DataFetcher, YahooFinanceSource, PolygonSource


class TestYahooFinanceSource:
    """测试Yahoo Finance数据源"""
    
    def test_init(self):
        """测试初始化"""
        source = YahooFinanceSource()
        assert source.name == "Yahoo Finance"
    
    @patch('yfinance.Ticker')
    def test_fetch_data_success(self, mock_ticker):
        """测试成功获取数据"""
        # 模拟数据
        mock_df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [104, 105, 106],
            'Volume': [1000, 1100, 1200]
        })
        mock_df.index = pd.date_range('2023-01-01', periods=3)
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker.return_value = mock_ticker_instance
        
        source = YahooFinanceSource()
        result = source.fetch_data('AAPL', '2023-01-01', '2023-01-03')
        
        assert result is not None
        assert not result.empty
        assert 'open' in result.columns
        assert 'close' in result.columns
        assert result.attrs['symbol'] == 'AAPL'
        assert result.attrs['source'] == 'Yahoo Finance'
    
    @patch('yfinance.Ticker')
    def test_fetch_data_empty(self, mock_ticker):
        """测试获取空数据"""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance
        
        source = YahooFinanceSource()
        result = source.fetch_data('INVALID', '2023-01-01', '2023-01-03')
        
        assert result is None
    
    @patch('yfinance.Ticker')
    def test_fetch_data_error(self, mock_ticker):
        """测试获取数据时出错"""
        mock_ticker.side_effect = Exception("API Error")
        
        source = YahooFinanceSource()
        result = source.fetch_data('AAPL', '2023-01-01', '2023-01-03')
        
        assert result is None


class TestPolygonSource:
    """测试Polygon API数据源"""
    
    def test_init_with_api_key(self):
        """测试带API密钥初始化"""
        source = PolygonSource(api_key="test_key")
        assert source.api_key == "test_key"
        assert source.name == "Polygon API"
    
    def test_init_without_api_key(self):
        """测试无API密钥初始化"""
        with patch.dict(os.environ, {}, clear=True):
            source = PolygonSource()
            assert source.api_key is None
    
    @patch('requests.get')
    def test_fetch_data_success(self, mock_get):
        """测试成功获取数据"""
        # 模拟API响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [
                {'t': 1672531200000, 'o': 100, 'h': 105, 'l': 99, 'c': 104, 'v': 1000},
                {'t': 1672617600000, 'o': 101, 'h': 106, 'l': 100, 'c': 105, 'v': 1100}
            ]
        }
        mock_get.return_value = mock_response
        
        source = PolygonSource(api_key="test_key")
        result = source.fetch_data('AAPL', '2023-01-01', '2023-01-03')
        
        assert result is not None
        assert not result.empty
        assert 'open' in result.columns
        assert 'close' in result.columns
        assert result.attrs['symbol'] == 'AAPL'
    
    @patch('requests.get')
    def test_fetch_data_no_api_key(self, mock_get):
        """测试无API密钥时返回None"""
        source = PolygonSource(api_key=None)
        result = source.fetch_data('AAPL', '2023-01-01', '2023-01-03')
        
        assert result is None
        mock_get.assert_not_called()
    
    @patch('requests.get')
    def test_fetch_data_api_error(self, mock_get):
        """测试API错误"""
        mock_get.side_effect = Exception("Network Error")
        
        source = PolygonSource(api_key="test_key")
        result = source.fetch_data('AAPL', '2023-01-01', '2023-01-03')
        
        assert result is None


class TestDataFetcher:
    """测试DataFetcher主类"""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """临时缓存目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def data_fetcher(self, temp_cache_dir):
        """数据获取器实例"""
        return DataFetcher(cache_dir=temp_cache_dir)
    
    def test_init(self, temp_cache_dir):
        """测试初始化"""
        fetcher = DataFetcher(cache_dir=temp_cache_dir)
        assert fetcher.cache_dir == temp_cache_dir
        assert fetcher.primary_source == "yahoo"
        assert fetcher.fallback_source == "polygon"
        assert os.path.exists(temp_cache_dir)
    
    def test_get_cache_file(self, data_fetcher):
        """测试获取缓存文件路径"""
        cache_file = data_fetcher._get_cache_file('AAPL', '2023-01-01', '2023-01-31', '1d')
        expected = os.path.join(data_fetcher.cache_dir, 'AAPL_2023-01-01_2023-01-31_1d.pkl')
        assert cache_file == expected
    
    def test_validate_data(self, data_fetcher):
        """测试数据验证"""
        # 有效数据
        valid_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [104, 105, 106],
            'volume': [1000, 1100, 1200]
        })
        assert data_fetcher.validate_data(valid_df, 'AAPL') == True
        
        # 无效数据 - 空DataFrame
        empty_df = pd.DataFrame()
        assert data_fetcher.validate_data(empty_df, 'AAPL') == False
        
        # 无效数据 - 缺少列
        invalid_df = pd.DataFrame({'open': [100, 101]})
        assert data_fetcher.validate_data(invalid_df, 'AAPL') == False
    
    @patch('pickle.load')
    @patch('os.path.exists')
    def test_load_from_cache_success(self, mock_exists, mock_pickle_load, data_fetcher):
        """测试成功从缓存加载"""
        mock_exists.return_value = True
        mock_df = pd.DataFrame({'close': [100, 101, 102]})
        mock_pickle_load.return_value = mock_df
        
        data = {}
        result = data_fetcher._load_from_cache('test.pkl', 'AAPL', data)
        
        assert result == True
        assert 'AAPL' in data
        assert data['AAPL'].equals(mock_df)
    
    @patch('os.path.exists')
    def test_load_from_cache_file_not_exists(self, mock_exists, data_fetcher):
        """测试缓存文件不存在"""
        mock_exists.return_value = False
        
        data = {}
        result = data_fetcher._load_from_cache('test.pkl', 'AAPL', data)
        
        assert result == False
        assert 'AAPL' not in data
    
    @patch('pickle.dump')
    def test_save_to_cache(self, mock_pickle_dump, data_fetcher):
        """测试保存到缓存"""
        df = pd.DataFrame({'close': [100, 101, 102]})
        
        data_fetcher._save_to_cache(df, 'test.pkl', 'AAPL')
        
        mock_pickle_dump.assert_called_once()
    
    def test_get_cache_info(self, data_fetcher):
        """测试获取缓存信息"""
        # 创建一些测试文件
        test_files = ['AAPL_2023-01-01_2023-01-31_1d.pkl', 'MSFT_2023-01-01_2023-01-31_1d.pkl']
        for file in test_files:
            file_path = os.path.join(data_fetcher.cache_dir, file)
            with open(file_path, 'w') as f:
                f.write('test data')
        
        info = data_fetcher.get_cache_info()
        
        assert info['file_count'] == 2
        assert info['total_size_mb'] > 0
        assert len(info['files']) == 2
    
    def test_clear_cache(self, data_fetcher):
        """测试清理缓存"""
        # 创建测试文件
        test_file = os.path.join(data_fetcher.cache_dir, 'AAPL_2023-01-01_2023-01-31_1d.pkl')
        with open(test_file, 'w') as f:
            f.write('test data')
        
        # 清理特定符号的缓存
        count = data_fetcher.clear_cache('AAPL')
        assert count == 1
        assert not os.path.exists(test_file)
    
    @patch.object(DataFetcher, '_fetch_from_sources')
    @patch.object(DataFetcher, '_load_from_cache')
    def test_fetch_data_with_cache(self, mock_load_cache, mock_fetch_sources, data_fetcher):
        """测试带缓存的获取数据"""
        # 模拟缓存命中
        mock_load_cache.return_value = True
        
        data = data_fetcher.fetch_data(['AAPL'], '2023-01-01', '2023-01-31')
        
        assert 'AAPL' in data
        mock_fetch_sources.assert_not_called()
    
    @patch.object(DataFetcher, '_fetch_from_sources')
    @patch.object(DataFetcher, '_load_from_cache')
    def test_fetch_data_no_cache(self, mock_load_cache, mock_fetch_sources, data_fetcher):
        """测试无缓存的获取数据"""
        # 模拟缓存未命中
        mock_load_cache.return_value = False
        mock_df = pd.DataFrame({'close': [100, 101, 102]})
        mock_fetch_sources.return_value = mock_df
        
        data = data_fetcher.fetch_data(['AAPL'], '2023-01-01', '2023-01-31')
        
        assert 'AAPL' in data
        mock_fetch_sources.assert_called_once()
    
    @patch.object(DataFetcher, '_fetch_from_sources')
    @patch.object(DataFetcher, '_load_from_cache')
    def test_fetch_data_failed_symbols(self, mock_load_cache, mock_fetch_sources, data_fetcher):
        """测试获取数据失败的符号"""
        mock_load_cache.return_value = False
        mock_fetch_sources.return_value = None
        
        data = data_fetcher.fetch_data(['INVALID'], '2023-01-01', '2023-01-31')
        
        assert data == {}
    
    @patch('yfinance.Ticker')
    def test_get_info(self, mock_ticker, data_fetcher):
        """测试获取公司信息"""
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {
            'longName': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'marketCap': 3000000000000,
            'trailingPE': 25.5
        }
        mock_ticker.return_value = mock_ticker_instance
        
        info = data_fetcher.get_info('AAPL')
        
        assert info['name'] == 'Apple Inc.'
        assert info['sector'] == 'Technology'
        assert info['market_cap'] == 3000000000000
