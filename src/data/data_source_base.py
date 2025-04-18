"""
Abstract base class for all data sources in the trading system.
"""
from abc import ABC, abstractmethod
import datetime
from typing import Dict, List, Optional, Union, Any
import pandas as pd

class DataSourceBase(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    def get_data(self, symbol: str, start_date=None, end_date=None, timeframe='1d') -> pd.DataFrame:
        """
        Get data for a symbol within a date range.
        
        Args:
            symbol: Instrument symbol
            start_date: Start date (datetime or string)
            end_date: End date (datetime or string)
            timeframe: Data timeframe (e.g., '1d', '1h', '5m')
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    @abstractmethod
    def is_available(self, symbol: str, start_date=None, end_date=None, timeframe='1d') -> bool:
        """
        Check if data is available for the specified parameters.
        
        Args:
            symbol: Symbol to check
            start_date: Start date (datetime or string)
            end_date: End date (datetime or string)
            timeframe: Data timeframe (e.g., '1d', '1h', '5m')
            
        Returns:
            True if data is available, False otherwise
        """
        pass
