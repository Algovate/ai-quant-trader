"""
Data fetcher module for retrieving stock data using yfinance.
"""

import yfinance as yf
import pandas as pd
import os
import pickle
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches and caches stock data using yfinance."""

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def fetch_data(self,
                   symbols: List[str],
                   start_date: str,
                   end_date: str,
                   interval: str = "1d",
                   use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for given symbols."""
        data = {}

        for symbol in symbols:
            cache_file = os.path.join(self.cache_dir, f"{symbol}_{start_date}_{end_date}_{interval}.pkl")

            if use_cache and os.path.exists(cache_file):
                logger.info(f"Loading cached data for {symbol}")
                with open(cache_file, 'rb') as f:
                    data[symbol] = pickle.load(f)
                continue

            logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval=interval)

                if df.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue

                df.columns = df.columns.str.lower()
                df.index.name = 'date'

                data[symbol] = df

                if use_cache:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(df, f)
                    logger.info(f"Cached data for {symbol}")

            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                continue

        return data

    def get_info(self, symbol: str) -> Dict:
        """Get company information for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0)
            }
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {}

    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cached data."""
        if symbol:
            pattern = f"{symbol}_*.pkl"
            files = [f for f in os.listdir(self.cache_dir) if f.startswith(f"{symbol}_")]
        else:
            files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]

        for file in files:
            os.remove(os.path.join(self.cache_dir, file))
        logger.info(f"Cleared cache for {len(files)} files")