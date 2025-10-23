"""
Feature engineering module for creating ML features from stock data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates ML features from processed stock data."""

    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length

    def create_features(self, df: pd.DataFrame, target_col: str = 'close') -> pd.DataFrame:
        """
        Create ML features from stock data.

        Args:
            df: Processed stock data with technical indicators
            target_col: Column to use as target variable

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Price-based features
        df = self._add_price_features(df)

        # Technical indicator features
        df = self._add_technical_features(df)

        # Volume features
        df = self._add_volume_features(df)

        # Time-based features
        df = self._add_time_features(df)

        # Lag features
        df = self._add_lag_features(df)

        # Rolling statistics
        df = self._add_rolling_features(df)

        # Target variable (future returns)
        df = self._add_target_variables(df, target_col)

        # Remove rows with NaN values
        df = df.dropna()

        logger.info(f"Created features: {len(df.columns)} columns, {len(df)} rows")
        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Price ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['high_close_ratio'] = df['high'] / df['close']
        df['low_close_ratio'] = df['low'] / df['close']

        # Price position within daily range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

        # Gap features
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_ratio'] = df['gap'] / df['close'].shift(1)

        # Intraday volatility
        df['intraday_volatility'] = (df['high'] - df['low']) / df['close']

        return df

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features."""
        # Moving average crossovers
        df['ma_5_20_cross'] = np.where(df['ma_5'] > df['ma_20'], 1, 0)
        df['ma_10_50_cross'] = np.where(df['ma_10'] > df['ma_50'], 1, 0)
        df['ma_20_50_cross'] = np.where(df['ma_20'] > df['ma_50'], 1, 0)

        # Price relative to moving averages
        df['price_ma5_ratio'] = df['close'] / df['ma_5']
        df['price_ma20_ratio'] = df['close'] / df['ma_20']
        df['price_ma50_ratio'] = df['close'] / df['ma_50']
        df['price_ma200_ratio'] = df['close'] / df['ma_200']

        # MACD features
        df['macd_signal_cross'] = np.where(df['macd'] > df['macd_signal'], 1, 0)
        df['macd_momentum'] = df['macd'] - df['macd'].shift(1)

        # RSI features
        df['rsi_overbought'] = np.where(df['rsi'] > 70, 1, 0)
        df['rsi_oversold'] = np.where(df['rsi'] < 30, 1, 0)
        df['rsi_momentum'] = df['rsi'] - df['rsi'].shift(1)

        # Bollinger Band features
        df['bb_squeeze'] = np.where(df['bb_width'] < df['bb_width'].rolling(20).mean(), 1, 0)
        df['bb_breakout'] = np.where(df['close'] > df['bb_upper'], 1, 0)
        df['bb_breakdown'] = np.where(df['close'] < df['bb_lower'], 1, 0)

        # Stochastic features
        df['stoch_overbought'] = np.where(df['stoch_k'] > 80, 1, 0)
        df['stoch_oversold'] = np.where(df['stoch_k'] < 20, 1, 0)
        df['stoch_cross'] = np.where(df['stoch_k'] > df['stoch_d'], 1, 0)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume ratios
        df['volume_ma_ratio'] = df['volume'] / df['volume_ma']
        df['volume_price_trend'] = df['volume'] * df['returns']

        # Volume momentum
        df['volume_momentum'] = df['volume'] - df['volume'].shift(1)
        df['volume_acceleration'] = df['volume_momentum'] - df['volume_momentum'].shift(1)

        # On-Balance Volume (simplified)
        df['obv'] = (df['volume'] * np.sign(df['close'] - df['close'].shift(1))).cumsum()
        df['obv_ma'] = df['obv'].rolling(20).mean()
        df['obv_signal'] = np.where(df['obv'] > df['obv_ma'], 1, 0)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        # Day of week
        df['day_of_week'] = df.index.dayofweek
        df['is_monday'] = np.where(df['day_of_week'] == 0, 1, 0)
        df['is_friday'] = np.where(df['day_of_week'] == 4, 1, 0)

        # Month
        df['month'] = df.index.month
        df['is_january'] = np.where(df['month'] == 1, 1, 0)
        df['is_december'] = np.where(df['month'] == 12, 1, 0)

        # Quarter
        df['quarter'] = df.index.quarter

        # Days since start of year
        df['day_of_year'] = df.index.dayofyear

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features."""
        lag_periods = [1, 2, 3, 5, 10, 20]

        for lag in lag_periods:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)

        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistical features."""
        windows = [5, 10, 20, 50]

        for window in windows:
            # Rolling statistics for price
            df[f'close_mean_{window}'] = df['close'].rolling(window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window).std()
            df[f'close_min_{window}'] = df['close'].rolling(window).min()
            df[f'close_max_{window}'] = df['close'].rolling(window).max()

            # Rolling statistics for volume
            df[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()
            df[f'volume_std_{window}'] = df['volume'].rolling(window).std()

            # Rolling statistics for returns
            df[f'returns_mean_{window}'] = df['returns'].rolling(window).mean()
            df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
            df[f'returns_skew_{window}'] = df['returns'].rolling(window).skew()
            df[f'returns_kurt_{window}'] = df['returns'].rolling(window).kurt()

        return df

    def _add_target_variables(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add target variables for prediction."""
        # Future returns (1, 3, 5, 10 days ahead)
        for horizon in [1, 3, 5, 10]:
            df[f'future_return_{horizon}'] = df[target_col].shift(-horizon) / df[target_col] - 1
            df[f'future_direction_{horizon}'] = np.where(df[f'future_return_{horizon}'] > 0, 1, 0)

        # Volatility prediction
        df['future_volatility_5'] = df['returns'].shift(-5).rolling(5).std()
        df['future_volatility_10'] = df['returns'].shift(-10).rolling(10).std()

        return df

    def create_sequences(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series models.

        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
            target_col: Target column name

        Returns:
            Tuple of (X, y) arrays
        """
        # Select features and target
        data = df[feature_cols + [target_col]].values

        X, y = [], []

        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i, :-1])  # Features
            y.append(data[i, -1])  # Target

        return np.array(X), np.array(y)

    def engineer_features(self, data: Dict[str, pd.DataFrame],
                         feature_cols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Engineer features for all symbols.

        Args:
            data: Dictionary mapping symbols to DataFrames
            feature_cols: List of specific columns to use as features

        Returns:
            Dictionary of DataFrames with engineered features
        """
        engineered_data = {}

        for symbol, df in data.items():
            logger.info(f"Engineering features for {symbol}")
            try:
                df_features = self.create_features(df)
                engineered_data[symbol] = df_features
                logger.info(f"Successfully engineered features for {symbol}")
            except Exception as e:
                logger.error(f"Error engineering features for {symbol}: {e}")
                continue

        return engineered_data
