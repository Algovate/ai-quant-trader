"""
AI Trading System - 数据获取模块

这个模块负责从各种数据源获取股票历史数据，包括Yahoo Finance、
Polygon API等。支持数据缓存、错误处理、重试机制等功能。
"""

import yfinance as yf
import pandas as pd
import os
import pickle
import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Tuple, Any
import logging
from pathlib import Path

from ..core.constants import DataConfig, APIConfig
from ..core.utils import FileUtils, DisplayUtils

logger = logging.getLogger(__name__)


class DataSource:
    """数据源基类"""

    def __init__(self, name: str):
        self.name = name

    def fetch_data(self, symbol: str, start_date: str, end_date: str, interval: str = "1d") -> Optional[pd.DataFrame]:
        """获取数据"""
        raise NotImplementedError


class YahooFinanceSource(DataSource):
    """Yahoo Finance数据源"""

    def __init__(self):
        super().__init__("Yahoo Finance")

    def fetch_data(self, symbol: str, start_date: str, end_date: str, interval: str = "1d") -> Optional[pd.DataFrame]:
        """从Yahoo Finance获取数据"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                logger.warning(f"No data found for {symbol} from Yahoo Finance")
                return None

            # 标准化列名
            df.columns = df.columns.str.lower()
            df.index.name = 'date'

            # 添加元数据
            df.attrs['symbol'] = symbol
            df.attrs['source'] = self.name
            df.attrs['fetched_at'] = datetime.now().isoformat()

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol} from Yahoo Finance: {e}")
            return None


class PolygonSource(DataSource):
    """Polygon API数据源"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__("Polygon API")
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        self.base_url = DataConfig.POLYGON_API_BASE

    def fetch_data(self, symbol: str, start_date: str, end_date: str, interval: str = "1d") -> Optional[pd.DataFrame]:
        """从Polygon API获取数据"""
        if not self.api_key:
            logger.warning("Polygon API key not found")
            return None

        try:
            # 转换间隔格式
            interval_map = {
                "1d": "day",
                "1h": "hour",
                "1m": "minute"
            }
            polygon_interval = interval_map.get(interval, "day")

            # 构建API请求
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/{polygon_interval}/{start_date}/{end_date}"
            headers = {"Authorization": f"Bearer {self.api_key}"}

            response = requests.get(url, headers=headers, timeout=APIConfig.DEFAULT_TIMEOUT)
            response.raise_for_status()

            data = response.json()

            if not data.get('results'):
                logger.warning(f"No data found for {symbol} from Polygon API")
                return None

            # 转换为DataFrame
            df = pd.DataFrame(data['results'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # 重命名列
            column_map = {
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            }
            df = df.rename(columns=column_map)

            # 只保留需要的列
            df = df[['open', 'high', 'low', 'close', 'volume']]

            # 添加元数据
            df.attrs['symbol'] = symbol
            df.attrs['source'] = self.name
            df.attrs['fetched_at'] = datetime.now().isoformat()

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol} from Polygon API: {e}")
            return None


class DataFetcher:
    """数据获取器 - 支持多数据源和缓存"""

    def __init__(self, cache_dir: str = DataConfig.CACHE_DIR,
                 primary_source: str = "yahoo",
                 fallback_source: str = "polygon"):
        self.cache_dir = cache_dir
        self.primary_source = primary_source
        self.fallback_source = fallback_source

        # 初始化数据源
        self.sources = {
            "yahoo": YahooFinanceSource(),
            "polygon": PolygonSource()
        }

        # 确保缓存目录存在
        FileUtils.ensure_directory(cache_dir)

    def fetch_data(self,
                   symbols: List[str],
                   start_date: str,
                   end_date: str,
                   interval: str = "1d",
                   use_cache: bool = True,
                   max_retries: int = 3) -> Dict[str, pd.DataFrame]:
        """获取历史数据，支持多数据源和缓存"""
        data = {}
        failed_symbols = []

        for symbol in symbols:
            cache_file = self._get_cache_file(symbol, start_date, end_date, interval)

            # 尝试从缓存加载
            if use_cache and self._load_from_cache(cache_file, symbol, data):
                continue

            # 从数据源获取数据
            df = self._fetch_from_sources(symbol, start_date, end_date, interval, max_retries)

            if df is not None and not df.empty:
                data[symbol] = df
                if use_cache:
                    self._save_to_cache(df, cache_file, symbol)
                DisplayUtils.print_success(f"Fetched data for {symbol}")
            else:
                failed_symbols.append(symbol)
                DisplayUtils.print_error(f"Failed to fetch data for {symbol}")

        if failed_symbols:
            DisplayUtils.print_warning(f"Failed to fetch data for: {', '.join(failed_symbols)}")

        return data

    def _get_cache_file(self, symbol: str, start_date: str, end_date: str, interval: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{symbol}_{start_date}_{end_date}_{interval}.pkl")

    def _load_from_cache(self, cache_file: str, symbol: str, data: Dict[str, pd.DataFrame]) -> bool:
        """从缓存加载数据"""
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    df = pickle.load(f)
                data[symbol] = df
                logger.info(f"Loaded cached data for {symbol}")
                return True
        except Exception as e:
            logger.warning(f"Error loading cache for {symbol}: {e}")
        return False

    def _fetch_from_sources(self, symbol: str, start_date: str, end_date: str,
                           interval: str, max_retries: int) -> Optional[pd.DataFrame]:
        """从数据源获取数据，支持重试和备用源"""
        # 尝试主数据源
        for attempt in range(max_retries):
            try:
                source = self.sources[self.primary_source]
                df = source.fetch_data(symbol, start_date, end_date, interval)
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol} from {self.primary_source}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(APIConfig.DEFAULT_RATE_LIMIT_DELAY)

        # 尝试备用数据源
        if self.fallback_source != self.primary_source:
            try:
                source = self.sources[self.fallback_source]
                df = source.fetch_data(symbol, start_date, end_date, interval)
                if df is not None and not df.empty:
                    logger.info(f"Used fallback source {self.fallback_source} for {symbol}")
                    return df
            except Exception as e:
                logger.error(f"Fallback source {self.fallback_source} failed for {symbol}: {e}")

        return None

    def _save_to_cache(self, df: pd.DataFrame, cache_file: str, symbol: str) -> None:
        """保存数据到缓存"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
            logger.info(f"Cached data for {symbol}")
        except Exception as e:
            logger.error(f"Error caching data for {symbol}: {e}")

    def get_info(self, symbol: str) -> Dict:
        """获取公司信息"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'description': info.get('longBusinessSummary', ''),
                'website': info.get('website', ''),
                'employees': info.get('fullTimeEmployees', 0)
            }
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {}

    def clear_cache(self, symbol: Optional[str] = None) -> int:
        """清理缓存数据"""
        try:
            if symbol:
                files = [f for f in os.listdir(self.cache_dir) if f.startswith(f"{symbol}_")]
            else:
                files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]

            for file in files:
                file_path = os.path.join(self.cache_dir, file)
                os.remove(file_path)

            logger.info(f"Cleared cache for {len(files)} files")
            return len(files)
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0

    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        try:
            files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
            total_size = 0

            for file in files:
                file_path = os.path.join(self.cache_dir, file)
                total_size += os.path.getsize(file_path)

            return {
                'file_count': len(files),
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'files': files
            }
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return {'file_count': 0, 'total_size_mb': 0, 'files': []}

    def validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """验证数据质量"""
        if df.empty:
            logger.warning(f"Empty data for {symbol}")
            return False

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.warning(f"Missing columns for {symbol}: {missing_columns}")
            return False

        # 检查数据完整性
        if df.isnull().any().any():
            logger.warning(f"Missing values in data for {symbol}")
            return False

        # 检查价格合理性
        if (df['close'] <= 0).any():
            logger.warning(f"Invalid close prices for {symbol}")
            return False

        return True