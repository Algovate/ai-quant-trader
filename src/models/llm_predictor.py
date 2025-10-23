"""
LLM-based predictor using OpenRouter API for stock prediction.
"""

import requests
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

from src.core.constants import (
    OPENROUTER_BASE_URL, DEFAULT_TIMEOUT, DEFAULT_MAX_RETRIES,
    DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, DEFAULT_MIN_CONFIDENCE,
    DEFAULT_SYSTEM_PROMPT, FINANCIAL_ANALYST_PROMPT
)

load_dotenv()
logger = logging.getLogger(__name__)


class LLMPredictor:
    """LLM-based predictor using OpenRouter API."""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("API key not provided and OPENROUTER_API_KEY not found in environment variables")

        self.model = model or os.getenv('DEFAULT_LLM_MODEL', DEFAULT_MODEL)
        self.base_url = OPENROUTER_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "AI Quant Trading System"
        }
        self.is_trained = True
        self.model_type = "llm"
        self.last_prompts = {}
        self.last_responses = {}

    def _call_llm(self, prompt: str, symbol: str = None, max_retries: int = DEFAULT_MAX_RETRIES) -> str:
        """Call OpenRouter API with retry logic."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": FINANCIAL_ANALYST_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": DEFAULT_TEMPERATURE
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=DEFAULT_TIMEOUT
                )

                if response.status_code == 200:
                    result = response.json()
                    response_content = result["choices"][0]["message"]["content"]

                    # Save prompt and response for debugging
                    if symbol:
                        self.last_prompts[symbol] = {
                            'system_prompt': FINANCIAL_ANALYST_PROMPT,
                            'user_prompt': prompt,
                            'timestamp': datetime.now().isoformat()
                        }
                        self.last_responses[symbol] = {
                            'response': response_content,
                            'full_response': result,
                            'timestamp': datetime.now().isoformat()
                        }

                    return response_content
                else:
                    logger.warning(f"API call failed with status {response.status_code}: {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)

            except Exception as e:
                logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        raise Exception(f"Failed to get LLM response after {max_retries} attempts")

    def _format_market_data(self, symbol: str, data: pd.DataFrame,
                           lookback_days: int = 30) -> str:
        """Format market data for LLM input."""
        if not isinstance(data, pd.DataFrame):
            logger.error(f"Expected DataFrame for {symbol}, got {type(data)}")
            raise ValueError(f"Data must be a DataFrame, got {type(data)}")

        recent_data = data.tail(lookback_days)
        current_price = recent_data['close'].iloc[-1]

        # Safe price change calculation
        if len(recent_data) >= 2:
            price_change = (current_price - recent_data['close'].iloc[-2]) / recent_data['close'].iloc[-2] * 100
        else:
            price_change = 0.0

        volume = recent_data['volume'].iloc[-1]
        avg_volume = recent_data['volume'].mean()

        # Technical indicators (safe access)
        rsi = recent_data['rsi'].iloc[-1] if 'rsi' in recent_data.columns and len(recent_data) > 0 else None
        macd = recent_data['macd'].iloc[-1] if 'macd' in recent_data.columns and len(recent_data) > 0 else None
        ma_20 = recent_data['ma_20'].iloc[-1] if 'ma_20' in recent_data.columns and len(recent_data) > 0 else None
        ma_50 = recent_data['ma_50'].iloc[-1] if 'ma_50' in recent_data.columns and len(recent_data) > 0 else None

        # Price trend (safe calculation)
        if len(recent_data) >= 6:
            price_trend_5d = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[-6]) / recent_data['close'].iloc[-6] * 100
        else:
            price_trend_5d = 0.0

        if len(recent_data) >= 21:
            price_trend_20d = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[-21]) / recent_data['close'].iloc[-21] * 100
        else:
            price_trend_20d = 0.0

        # Volatility
        volatility = recent_data['returns'].std() * np.sqrt(252) if 'returns' in recent_data.columns else None

        # Format technical indicators safely
        rsi_str = f"{rsi:.2f}" if rsi is not None else 'N/A'
        macd_str = f"{macd:.4f}" if macd is not None else 'N/A'
        ma_20_str = f"${ma_20:.2f}" if ma_20 is not None else 'N/A'
        ma_50_str = f"${ma_50:.2f}" if ma_50 is not None else 'N/A'
        volatility_str = f"{volatility:.2%}" if volatility is not None else 'N/A'

        data_str = f"""
Stock Symbol: {symbol}
Current Price: ${current_price:.2f}
Price Change (1d): {price_change:.2f}%
Volume: {volume:,.0f} (Avg: {avg_volume:,.0f})
Price Trend (5d): {price_trend_5d:.2f}%
Price Trend (20d): {price_trend_20d:.2f}%

Technical Indicators:
- RSI: {rsi_str}
- MACD: {macd_str}
- MA 20: {ma_20_str}
- MA 50: {ma_50_str}
- Volatility: {volatility_str}

Recent Price History (last 10 days):
"""

        # Add recent price history
        for i, (date, row) in enumerate(recent_data.tail(10).iterrows()):
            data_str += f"  {date.strftime('%Y-%m-%d')}: ${row['close']:.2f} (Vol: {row['volume']:,.0f})\n"

        return data_str

    def _parse_llm_response(self, response: str) -> Dict[str, float]:
        """Parse LLM response to extract prediction and confidence."""
        try:
            lines = response.strip().split('\n')

            prediction = 0.0
            confidence = 0.5

            for line in lines:
                line = line.lower().strip()

                # Look for prediction indicators
                if 'prediction' in line or 'forecast' in line or 'expected' in line:
                    if 'up' in line or 'positive' in line or 'bullish' in line:
                        prediction = 0.02  # 2% positive prediction
                    elif 'down' in line or 'negative' in line or 'bearish' in line:
                        prediction = -0.02  # 2% negative prediction
                    elif '%' in line:
                        import re
                        percentages = re.findall(r'([+-]?\d+\.?\d*)%', line)
                        if percentages:
                            prediction = float(percentages[0]) / 100

                # Look for confidence indicators
                if 'confidence' in line or 'certainty' in line:
                    import re
                    confidences = re.findall(r'(\d+\.?\d*)%', line)
                    if confidences:
                        confidence = float(confidences[0]) / 100
                    elif 'high' in line:
                        confidence = 0.8
                    elif 'medium' in line or 'moderate' in line:
                        confidence = 0.6
                    elif 'low' in line:
                        confidence = 0.4

            # If no structured data found, use sentiment analysis
            if prediction == 0.0:
                positive_words = ['up', 'positive', 'bullish', 'strong', 'good', 'increase', 'rise', 'gain']
                negative_words = ['down', 'negative', 'bearish', 'weak', 'bad', 'decrease', 'fall', 'drop']

                response_lower = response.lower()
                positive_count = sum(1 for word in positive_words if word in response_lower)
                negative_count = sum(1 for word in negative_words if word in response_lower)

                if positive_count > negative_count:
                    prediction = 0.015  # 1.5% positive
                elif negative_count > positive_count:
                    prediction = -0.015  # 1.5% negative
                else:
                    prediction = 0.0  # Neutral

            # Ensure confidence is within reasonable bounds
            confidence = max(0.1, min(0.9, confidence))

            return {
                'prediction': prediction,
                'confidence': confidence,
                'raw_response': response
            }

        except Exception as e:
            logger.warning(f"Error parsing LLM response: {e}")
            return {
                'prediction': 0.0,
                'confidence': 0.5,
                'raw_response': response
            }

    def predict(self, data: Dict[str, pd.DataFrame],
                symbols: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Make predictions for given symbols using LLM."""
        if symbols is None:
            symbols = list(data.keys())

        predictions = {}

        for symbol in symbols:
            if symbol not in data:
                logger.warning(f"No data available for symbol {symbol}")
                continue

            try:
                symbol_data = data[symbol]
                if not isinstance(symbol_data, pd.DataFrame):
                    logger.error(f"Expected DataFrame for {symbol}, got {type(symbol_data)}")
                    continue
                market_data = self._format_market_data(symbol, symbol_data)

                prompt = f"""
Based on the following market data for {symbol}, provide a prediction for the next trading day's price movement.

{market_data}

Please provide:
1. Your prediction for the next day's price movement (as a percentage)
2. Your confidence level in this prediction (as a percentage)
3. Brief reasoning for your prediction

Format your response clearly and concisely.
"""

                response = self._call_llm(prompt, symbol)
                result = self._parse_llm_response(response)
                result['symbol'] = symbol
                result['timestamp'] = datetime.now()

                predictions[symbol] = result

                logger.info(f"LLM prediction for {symbol}: {result['prediction']:.2%} (confidence: {result['confidence']:.2%})")

                time.sleep(1)  # Rate limiting

            except Exception as e:
                logger.error(f"Error predicting for {symbol}: {e}")
                # Save prompt information even if API call fails
                if symbol not in self.last_prompts:
                    self.last_prompts[symbol] = {
                        'system_prompt': FINANCIAL_ANALYST_PROMPT,
                        'user_prompt': prompt,
                        'timestamp': datetime.now().isoformat()
                    }
                predictions[symbol] = {
                    'prediction': 0.0,
                    'confidence': 0.5,
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'error': str(e)
                }

        return predictions

    def predict_single(self, symbol: str, data: pd.DataFrame) -> Dict[str, float]:
        """Make prediction for a single symbol."""
        predictions = self.predict({symbol: data}, [symbol])
        return predictions.get(symbol, {
            'prediction': 0.0,
            'confidence': 0.5,
            'symbol': symbol,
            'timestamp': datetime.now()
        })

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the LLM model."""
        return {
            'model_type': 'llm',
            'model_name': self.model,
            'api_provider': 'OpenRouter',
            'is_trained': True,
            'supports_confidence': True,
            'supports_reasoning': True
        }

    def save_model(self, filepath: str):
        """Save model configuration."""
        config = {
            'model': self.model,
            'api_key': self.api_key[:10] + '...',
            'base_url': self.base_url,
            'model_type': 'llm'
        }

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"LLM model configuration saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model configuration."""
        with open(filepath, 'r') as f:
            config = json.load(f)

        self.model = config['model']
        self.base_url = config['base_url']

        logger.info(f"LLM model configuration loaded from {filepath}")


def create_llm_predictor(api_key: str = None, model: str = None) -> LLMPredictor:
    """Factory function to create LLM predictor."""
    return LLMPredictor(api_key, model)