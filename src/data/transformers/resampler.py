"""
Component for resampling time series data.
"""
import pandas as pd
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class Resampler:
    """Component for resampling time series data."""
    
    def __init__(self, rule: str = 'D', agg_dict: Optional[Dict[str, str]] = None):
        """
        Initialize the resampler.
        
        Args:
            rule: Resampling frequency rule (e.g., 'D', 'H', '5min')
            agg_dict: Dictionary of column -> aggregation function
        """
        self.rule = rule
        self.agg_dict = agg_dict or {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
    
    def resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample a DataFrame.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            Resampled DataFrame
            
        Raises:
            ValueError: If index is not a DatetimeIndex
        """
        # Check if index is a datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("DataFrame index must be a DatetimeIndex")
            raise ValueError("DataFrame index must be a DatetimeIndex")
            
        # Prepare aggregation dictionary
        agg_dict = {}
        for col, func in self.agg_dict.items():
            if col in df.columns:
                agg_dict[col] = func
        
        if not agg_dict:
            logger.warning("No columns to aggregate")
            return df
            
        try:
            # Resample data
            resampled = df.resample(self.rule).agg(agg_dict)
            
            # Drop rows with NaN values
            resampled = resampled.dropna()
            
            return resampled
        except Exception as e:
            logger.error(f"Error resampling data: {e}")
            return df

    def resample_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample data to a specific timeframe.

        Args:
            df: DataFrame with datetime index
            timeframe: Target timeframe ('1d', '1h', '5m', etc.)

        Returns:
            Resampled DataFrame
        """
        # Map common timeframe strings to pandas frequency strings
        timeframe_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1h', 
            '4h': '4h', 
            '1d': 'D',
            '1w': 'W',
            '1M': 'ME'  
        }

        # Convert timeframe to pandas frequency
        rule = timeframe_map.get(timeframe, timeframe)

        # Update rule and resample
        self.rule = rule
        return self.resample(df)
    
