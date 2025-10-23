"""
技术指标计算模块
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

from src.core.constants import (
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BOLLINGER_PERIOD, BOLLINGER_STD, STOCH_K_PERIOD, STOCH_D_PERIOD,
    WILLIAMS_R_PERIOD, ATR_PERIOD, MA_PERIODS
)

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """技术指标计算器"""

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = MACD_FAST,
                      slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram

    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = BOLLINGER_PERIOD,
                                 std_dev: float = BOLLINGER_STD) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """计算布林带指标"""
        bb_middle = prices.rolling(window=period).mean()
        bb_std_val = prices.rolling(window=period).std()
        bb_upper = bb_middle + (bb_std_val * std_dev)
        bb_lower = bb_middle - (bb_std_val * std_dev)
        bb_width = (bb_upper - bb_lower) / bb_middle
        bb_position = (prices - bb_lower) / (bb_upper - bb_lower)
        return bb_middle, bb_upper, bb_lower, bb_width, bb_position

    @staticmethod
    def calculate_stochastic(df: pd.DataFrame, k_period: int = STOCH_K_PERIOD,
                           d_period: int = STOCH_D_PERIOD) -> Tuple[pd.Series, pd.Series]:
        """计算随机振荡器"""
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent

    @staticmethod
    def calculate_williams_r(df: pd.DataFrame, period: int = WILLIAMS_R_PERIOD) -> pd.Series:
        """计算威廉指标"""
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
        return williams_r

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
        """计算平均真实波幅"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr

    @staticmethod
    def calculate_moving_averages(prices: pd.Series, periods: list = MA_PERIODS) -> dict:
        """计算移动平均线"""
        mas = {}
        for period in periods:
            mas[f'ma_{period}'] = prices.rolling(window=period).mean()
        return mas

    @staticmethod
    def calculate_volume_indicators(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """计算成交量指标"""
        volume_ma = df['volume'].rolling(window=20).mean()
        volume_ratio = df['volume'] / volume_ma
        return volume_ma, volume_ratio

    @staticmethod
    def calculate_price_indicators(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算价格指标"""
        returns = df['close'].pct_change()
        log_returns = np.log(df['close'] / df['close'].shift(1))
        volatility = returns.rolling(window=20).std()
        return returns, log_returns, volatility

    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """计算支撑阻力位"""
        resistance = df['high'].rolling(window=20).max()
        support = df['low'].rolling(window=20).min()
        return resistance, support


def add_all_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加所有技术指标到DataFrame"""
    df = df.copy()

    # Moving Averages
    mas = TechnicalIndicators.calculate_moving_averages(df['close'])
    for name, ma in mas.items():
        df[name] = ma

    # MACD
    macd, macd_signal, macd_histogram = TechnicalIndicators.calculate_macd(df['close'])
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_histogram'] = macd_histogram

    # RSI
    df['rsi'] = TechnicalIndicators.calculate_rsi(df['close'])

    # Bollinger Bands
    bb_middle, bb_upper, bb_lower, bb_width, bb_position = TechnicalIndicators.calculate_bollinger_bands(df['close'])
    df['bb_middle'] = bb_middle
    df['bb_upper'] = bb_upper
    df['bb_lower'] = bb_lower
    df['bb_width'] = bb_width
    df['bb_position'] = bb_position

    # Stochastic
    df['stoch_k'], df['stoch_d'] = TechnicalIndicators.calculate_stochastic(df)

    # Williams %R
    df['williams_r'] = TechnicalIndicators.calculate_williams_r(df)

    # ATR
    df['atr'] = TechnicalIndicators.calculate_atr(df)

    # Volume indicators
    df['volume_ma'], df['volume_ratio'] = TechnicalIndicators.calculate_volume_indicators(df)

    # Price indicators
    df['returns'], df['log_returns'], df['volatility'] = TechnicalIndicators.calculate_price_indicators(df)

    # Support and Resistance
    df['resistance'], df['support'] = TechnicalIndicators.calculate_support_resistance(df)

    return df
