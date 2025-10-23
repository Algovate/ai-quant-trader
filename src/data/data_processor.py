"""
AI Trading System - 数据处理模块

这个模块负责数据清洗、验证、技术指标计算等数据处理任务。
提供了完整的数据处理流水线，包括数据质量检查、异常值处理、
技术指标计算等功能。
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

from .technical_indicators import add_all_technical_indicators
from ..core.constants import TechnicalIndicators
from ..core.utils import DisplayUtils, FileUtils

logger = logging.getLogger(__name__)


class DataValidator:
    """数据验证器"""

    @staticmethod
    def validate_ohlcv(df: pd.DataFrame, symbol: str) -> Tuple[bool, List[str]]:
        """验证OHLCV数据"""
        errors = []

        # 检查必需列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")

        # 检查数据完整性
        if df.empty:
            errors.append("Empty DataFrame")
            return False, errors

        # 检查缺失值
        null_counts = df.isnull().sum()
        if null_counts.any():
            errors.append(f"Missing values: {null_counts[null_counts > 0].to_dict()}")

        # 检查价格合理性
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                if (df[col] <= 0).any():
                    errors.append(f"Invalid {col} prices (<= 0)")
                if (df[col] > 1e6).any():
                    errors.append(f"Suspicious {col} prices (> 1M)")

        # 检查OHLC逻辑
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            if (df['high'] < df['low']).any():
                errors.append("High < Low prices")
            if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
                errors.append("High < Open/Close prices")
            if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
                errors.append("Low > Open/Close prices")

        # 检查成交量
        if 'volume' in df.columns:
            if (df['volume'] < 0).any():
                errors.append("Negative volume")

        return len(errors) == 0, errors

    @staticmethod
    def validate_time_series(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """验证时间序列数据"""
        errors = []

        # 检查索引是否为时间类型
        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append("Index is not DatetimeIndex")

        # 检查时间顺序
        if not df.index.is_monotonic_increasing:
            errors.append("Index is not monotonic increasing")

        # 检查重复时间戳
        if df.index.duplicated().any():
            errors.append("Duplicate timestamps found")

        return len(errors) == 0, errors


class DataCleaner:
    """数据清洗器"""

    @staticmethod
    def remove_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """移除异常值"""
        df_clean = df.copy()

        if method == 'iqr':
            for col in ['open', 'high', 'low', 'close']:
                if col in df_clean.columns:
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

        elif method == 'zscore':
            for col in ['open', 'high', 'low', 'close']:
                if col in df_clean.columns:
                    z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                    df_clean = df_clean[z_scores < threshold]

        return df_clean

    @staticmethod
    def fill_missing_values(df: pd.DataFrame, method: str = 'forward') -> pd.DataFrame:
        """填充缺失值"""
        df_filled = df.copy()

        if method == 'forward':
            df_filled = df_filled.fillna(method='ffill')
        elif method == 'backward':
            df_filled = df_filled.fillna(method='bfill')
        elif method == 'interpolate':
            df_filled = df_filled.interpolate()
        elif method == 'mean':
            for col in df_filled.columns:
                if df_filled[col].dtype in ['float64', 'int64']:
                    df_filled[col] = df_filled[col].fillna(df_filled[col].mean())

        return df_filled

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """移除重复数据"""
        return df[~df.index.duplicated(keep='last')]

    @staticmethod
    def filter_volume(df: pd.DataFrame, min_volume: int = 0) -> pd.DataFrame:
        """过滤成交量"""
        if 'volume' in df.columns:
            return df[df['volume'] >= min_volume]
        return df


class DataProcessor:
    """数据处理器 - 完整的数据处理流水线"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.validator = DataValidator()
        self.cleaner = DataCleaner()

    def clean_data(self, df: pd.DataFrame, symbol: str = "Unknown") -> pd.DataFrame:
        """清洗和验证股票数据"""
        original_length = len(df)

        # 数据验证
        is_valid, errors = self.validator.validate_ohlcv(df, symbol)
        if not is_valid:
            logger.warning(f"Data validation errors for {symbol}: {errors}")

        # 移除重复数据
        df = self.cleaner.remove_duplicates(df)

        # 按时间排序
        df = df.sort_index()

        # 过滤成交量
        df = self.cleaner.filter_volume(df, min_volume=0)

        # 填充缺失值
        df = self.cleaner.fill_missing_values(df, method='forward')

        # 移除异常值（可选）
        outlier_method = self.config.get('outlier_method', 'none')
        if outlier_method != 'none':
            df = self.cleaner.remove_outliers(df, method=outlier_method)

        # 移除极端价格变化
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                pct_change = df[col].pct_change()
                extreme_changes = abs(pct_change) > 0.5  # 50%以上变化
                if extreme_changes.any():
                    logger.warning(f"Found {extreme_changes.sum()} extreme price changes in {col} for {symbol}")
                    # 可以选择移除或标记这些数据点

        final_length = len(df)
        logger.info(f"Cleaned data for {symbol}: {original_length} -> {final_length} rows")

        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        try:
            df_with_indicators = add_all_technical_indicators(df)
            logger.info(f"Added technical indicators: {len(df_with_indicators.columns)} columns")
            return df_with_indicators
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df

    def process_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """处理所有数据"""
        processed_data = {}
        success_count = 0
        error_count = 0

        for symbol, df in data.items():
            logger.info(f"Processing data for {symbol}")
            try:
                # 数据清洗
                df_clean = self.clean_data(df, symbol)

                if df_clean.empty:
                    logger.warning(f"No data remaining after cleaning for {symbol}")
                    error_count += 1
                    continue

                # 添加技术指标
                df_processed = self.add_technical_indicators(df_clean)

                # 最终验证
                is_valid, errors = self.validator.validate_ohlcv(df_processed, symbol)
                if not is_valid:
                    logger.warning(f"Final validation errors for {symbol}: {errors}")

                processed_data[symbol] = df_processed
                success_count += 1
                DisplayUtils.print_success(f"Processed {symbol}: {len(df_processed)} rows, {len(df_processed.columns)} columns")

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                error_count += 1
                continue

        logger.info(f"Data processing completed: {success_count} successful, {error_count} failed")
        return processed_data

    def get_data_summary(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """获取数据摘要"""
        summary = {
            'total_symbols': len(data),
            'symbols': list(data.keys()),
            'date_ranges': {},
            'data_quality': {}
        }

        for symbol, df in data.items():
            if not df.empty:
                summary['date_ranges'][symbol] = {
                    'start': df.index.min().strftime('%Y-%m-%d'),
                    'end': df.index.max().strftime('%Y-%m-%d'),
                    'days': len(df)
                }

                # 数据质量指标
                summary['data_quality'][symbol] = {
                    'missing_values': df.isnull().sum().sum(),
                    'duplicate_rows': df.index.duplicated().sum(),
                    'columns': list(df.columns),
                    'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
                }

        return summary

    def export_processed_data(self, data: Dict[str, pd.DataFrame], output_dir: str = "data/results") -> Dict[str, str]:
        """导出处理后的数据"""
        FileUtils.ensure_directory(output_dir)
        exported_files = {}

        for symbol, df in data.items():
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"processed_{symbol}_{timestamp}.csv"
                filepath = os.path.join(output_dir, filename)

                df.to_csv(filepath)
                exported_files[symbol] = filepath
                logger.info(f"Exported processed data for {symbol} to {filepath}")

            except Exception as e:
                logger.error(f"Error exporting data for {symbol}: {e}")

        return exported_files