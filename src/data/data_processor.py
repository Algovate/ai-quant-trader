"""
Data processor module for cleaning and generating technical indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from src.data.technical_indicators import add_all_technical_indicators

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes and cleans stock data, generates technical indicators."""
    
    def __init__(self):
        pass
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate stock data."""
        df = df.dropna()
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        df = df[df['volume'] > 0]
        
        for col in ['open', 'high', 'low', 'close']:
            df[f'{col}_pct_change'] = df[col].pct_change()
            df = df[abs(df[f'{col}_pct_change']) < 1.0]
            df = df.drop(columns=[f'{col}_pct_change'])
        
        logger.info(f"Cleaned data: {len(df)} rows remaining")
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the DataFrame."""
        return add_all_technical_indicators(df)
    
    def process_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Process all data in the dictionary."""
        processed_data = {}
        
        for symbol, df in data.items():
            logger.info(f"Processing data for {symbol}")
            try:
                df_clean = self.clean_data(df)
                df_processed = self.add_technical_indicators(df_clean)
                processed_data[symbol] = df_processed
                logger.info(f"Successfully processed {symbol}: {len(df_processed)} rows")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        return processed_data