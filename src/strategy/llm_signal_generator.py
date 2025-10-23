"""
LLM-based signal generator that uses AI reasoning for trading decisions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from enum import Enum

from src.models.llm_predictor import LLMPredictor

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class LLMSignalGenerator:
    """Signal generator that uses LLM for intelligent trading decisions."""

    def __init__(self, llm_predictor: LLMPredictor,
                 min_confidence: float = 0.6,
                 signal_threshold: float = 0.02):
        """
        Initialize LLM signal generator.

        Args:
            llm_predictor: LLM predictor instance
            min_confidence: Minimum confidence level for signals
            signal_threshold: Minimum prediction magnitude for buy/sell signals
        """
        self.llm_predictor = llm_predictor
        self.min_confidence = min_confidence
        self.signal_threshold = signal_threshold
        self.signal_history = []

        logger.info(f"LLM Signal Generator initialized with confidence threshold: {min_confidence}")

    def _prediction_to_signal(self, prediction: float, confidence: float) -> Tuple[SignalType, float]:
        """
        Convert LLM prediction to trading signal.

        Args:
            prediction: LLM prediction (percentage)
            confidence: Prediction confidence (0-1)

        Returns:
            Tuple of (signal_type, signal_strength)
        """
        # Check confidence threshold
        if confidence < self.min_confidence:
            return SignalType.HOLD, 0.0

        # Determine signal based on prediction magnitude and direction
        abs_prediction = abs(prediction)

        if abs_prediction < self.signal_threshold:
            return SignalType.HOLD, abs_prediction

        if prediction > 0:
            # Positive prediction -> BUY signal
            signal_strength = min(abs_prediction * confidence, 1.0)
            return SignalType.BUY, signal_strength
        else:
            # Negative prediction -> SELL signal
            signal_strength = min(abs_prediction * confidence, 1.0)
            return SignalType.SELL, signal_strength

    def generate_llm_signals(self, data: Dict[str, pd.DataFrame],
                           symbols: List[str] = None) -> Dict[str, Dict]:
        """
        Generate trading signals using LLM predictions.

        Args:
            data: Dictionary mapping symbols to market data
            symbols: List of symbols to generate signals for

        Returns:
            Dictionary mapping symbols to signal information
        """
        if symbols is None:
            symbols = list(data.keys())

        logger.info(f"Generating LLM signals for {len(symbols)} symbols")

        # Get LLM predictions
        predictions = self.llm_predictor.predict(data, symbols)

        signals = {}

        for symbol in symbols:
            if symbol not in predictions:
                logger.warning(f"No prediction available for {symbol}")
                continue

            prediction_data = predictions[symbol]
            prediction = prediction_data['prediction']
            confidence = prediction_data['confidence']

            # Convert prediction to signal
            signal_type, signal_strength = self._prediction_to_signal(prediction, confidence)

            signal_info = {
                'symbol': symbol,
                'signal': signal_type.value,
                'strength': signal_strength,
                'confidence': confidence,
                'prediction': prediction,
                'timestamp': datetime.now(),
                'reasoning': prediction_data.get('raw_response', 'No reasoning provided')
            }

            signals[symbol] = signal_info

            # Log signal generation
            logger.info(f"Generated signal for {symbol}: {signal_type.value} "
                       f"(strength: {signal_strength:.2f}, confidence: {confidence:.2%})")

            # Store in history
            self.signal_history.append(signal_info)

        return signals

    def generate_signals_from_predictions(self, predictions: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Generate trading signals from existing LLM predictions.

        Args:
            predictions: Dictionary mapping symbols to prediction data

        Returns:
            Dictionary mapping symbols to signal information
        """
        signals = {}

        for symbol, prediction_data in predictions.items():
            prediction = prediction_data.get('prediction', 0.0)
            confidence = prediction_data.get('confidence', 0.5)

            signal_type, strength = self._prediction_to_signal(prediction, confidence)

            signals[symbol] = {
                'signal': signal_type.value,
                'strength': strength,
                'confidence': confidence,
                'prediction': prediction,
                'timestamp': prediction_data.get('timestamp'),
                'raw_response': prediction_data.get('raw_response', '')
            }

            logger.info(f"Generated signal for {symbol}: {signal_type.value} (strength: {strength:.2f}, confidence: {confidence:.2%})")

        return signals

    def generate_single_signal(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Generate signal for a single symbol.

        Args:
            symbol: Stock symbol
            data: Market data for the symbol

        Returns:
            Signal information dictionary
        """
        predictions = self.llm_predictor.predict({symbol: data}, [symbol])

        if symbol not in predictions:
            return {
                'symbol': symbol,
                'signal': SignalType.HOLD.value,
                'strength': 0.0,
                'confidence': 0.0,
                'prediction': 0.0,
                'timestamp': datetime.now(),
                'error': 'No prediction available'
            }

        prediction_data = predictions[symbol]
        prediction = prediction_data['prediction']
        confidence = prediction_data['confidence']

        signal_type, signal_strength = self._prediction_to_signal(prediction, confidence)

        signal_info = {
            'symbol': symbol,
            'signal': signal_type.value,
            'strength': signal_strength,
            'confidence': confidence,
            'prediction': prediction,
            'timestamp': datetime.now(),
            'reasoning': prediction_data.get('raw_response', 'No reasoning provided')
        }

        logger.info(f"Generated signal for {symbol}: {signal_type.value} "
                   f"(strength: {signal_strength:.2f}, confidence: {confidence:.2%})")

        return signal_info

    def get_signal_summary(self, signals: Dict[str, Dict]) -> Dict:
        """
        Get summary statistics for generated signals.

        Args:
            signals: Dictionary of signals

        Returns:
            Summary statistics
        """
        if not signals:
            return {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'hold_signals': 0,
                'average_confidence': 0.0,
                'average_strength': 0.0
            }

        buy_count = sum(1 for s in signals.values() if s['signal'] == SignalType.BUY.value)
        sell_count = sum(1 for s in signals.values() if s['signal'] == SignalType.SELL.value)
        hold_count = sum(1 for s in signals.values() if s['signal'] == SignalType.HOLD.value)

        avg_confidence = np.mean([s['confidence'] for s in signals.values()])
        avg_strength = np.mean([s['strength'] for s in signals.values()])

        return {
            'total_signals': len(signals),
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'hold_signals': hold_count,
            'average_confidence': avg_confidence,
            'average_strength': avg_strength
        }

    def get_signal_history(self) -> List[Dict]:
        """Get history of all generated signals."""
        return self.signal_history.copy()

    def clear_history(self):
        """Clear signal history."""
        self.signal_history.clear()
        logger.info("Signal history cleared")

    def update_parameters(self, min_confidence: float = None,
                         signal_threshold: float = None):
        """
        Update signal generation parameters.

        Args:
            min_confidence: New minimum confidence threshold
            signal_threshold: New signal magnitude threshold
        """
        if min_confidence is not None:
            self.min_confidence = min_confidence
            logger.info(f"Updated minimum confidence threshold to {min_confidence}")

        if signal_threshold is not None:
            self.signal_threshold = signal_threshold
            logger.info(f"Updated signal threshold to {signal_threshold}")

    def get_parameters(self) -> Dict:
        """Get current signal generation parameters."""
        return {
            'min_confidence': self.min_confidence,
            'signal_threshold': self.signal_threshold,
            'llm_model': self.llm_predictor.model
        }


def create_llm_signal_generator(llm_predictor: LLMPredictor,
                               min_confidence: float = 0.6,
                               signal_threshold: float = 0.02) -> LLMSignalGenerator:
    """
    Factory function to create LLM signal generator.

    Args:
        llm_predictor: LLM predictor instance
        min_confidence: Minimum confidence threshold
        signal_threshold: Signal magnitude threshold

    Returns:
        LLM signal generator instance
    """
    return LLMSignalGenerator(llm_predictor, min_confidence, signal_threshold)