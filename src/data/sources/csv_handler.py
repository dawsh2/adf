"""
CSV data source implementation.
"""
import os
import pandas as pd
import datetime
import logging
from typing import Dict, List, Optional, Union, Any

from ..data_source_base import DataSourceBase

logger = logging.getLogger(__name__)

class CSVDataSource(DataSourceBase):
    """Data source for CSV files."""
    
    def __init__(self, data_dir: str, filename_pattern='{symbol}_{timeframe}.csv', 
                 date_column='timestamp', date_format='%Y-%m-%d', 
                 column_map=None):
        """
        Initialize the CSV data source.
        
        Args:
            data_dir: Directory containing CSV files
            filename_pattern: Pattern for filenames
            date_column: Column containing dates
            date_format: Format of dates in CSV
            column_map: Map of CSV columns to standard column names
        """
        self.data_dir = data_dir
        self.filename_pattern = filename_pattern
        self.date_column = date_column
        self.date_format = date_format
        self.column_map = column_map or {
            'open': ['open', 'Open'],
            'high': ['high', 'High'],
            'low': ['low', 'Low'],
            'close': ['close', 'Close'],
            'volume': ['volume', 'Volume', 'vol', 'Vol']
        }

    def get_data(self, symbol: str, start_date=None, end_date=None, timeframe='1m') -> pd.DataFrame:
        """
        Get data for a symbol within a date range.
        
        Args:
            symbol: Symbol to get data for
            start_date: Start date (datetime or string)
            end_date: End date (datetime or string)
            timeframe: Data timeframe (e.g., '1d', '1h', '5m')
            
        Returns:
            DataFrame with OHLCV data
        """
        filename = self._get_filename(symbol, timeframe)
        logger.info(f"CSVDataSource: Loading file {filename}")

        if not os.path.exists(filename):
            logger.warning(f"File not found: {filename}")
            return pd.DataFrame()

        try:
            # Convert string dates to datetime if needed
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
                
            # Read CSV
            df = pd.read_csv(filename)
            logger.info(f"CSVDataSource: Loaded {len(df)} rows")

            # Convert date column to datetime
            if self.date_column in df.columns:
                # Log a sample of original timestamps
                if not df.empty:
                    logger.info(f"CSVDataSource: Original timestamp sample: {df[self.date_column].iloc[0]}")
                
                # Parse timestamps with pd.to_datetime which handles various formats
                df[self.date_column] = pd.to_datetime(df[self.date_column])
                
                # Log a sample of parsed timestamps
                if not df.empty:
                    logger.info(f"CSVDataSource: Parsed timestamp sample: {df[self.date_column].iloc[0]}")
                
                # Remove timezone info to ensure consistency
                if df[self.date_column].dt.tz is not None:
                    df[self.date_column] = df[self.date_column].dt.tz_localize(None)
                    logger.info(f"CSVDataSource: Removed timezone info from timestamps")
                
                # Make sure filter dates are also timezone-naive
                if start_date is not None and hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None:
                    start_date = start_date.replace(tzinfo=None)
                if end_date is not None and hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None:
                    end_date = end_date.replace(tzinfo=None)

                # Filter by date range
                if start_date is not None:
                    df = df[df[self.date_column] >= start_date]
                    logger.info(f"CSVDataSource: Filtered for dates >= {start_date}")
                    
                if end_date is not None:
                    df = df[df[self.date_column] <= end_date]
                    logger.info(f"CSVDataSource: Filtered for dates <= {end_date}")

                # Set date as index
                df.set_index(self.date_column, inplace=True)
                logger.info(f"CSVDataSource: Set {self.date_column} as index")

            # Map columns to standard names
            column_mapping = self._map_columns(df.columns)
            df = df.rename(columns=column_mapping)
            
            # Log info about the processed DataFrame
            if not df.empty:
                logger.info(f"CSVDataSource: Date range in data: {df.index.min()} to {df.index.max()}")
                logger.info(f"CSVDataSource: Columns after mapping: {list(df.columns)}")

            return df

        except Exception as e:
            logger.error(f"Error reading CSV file {filename}: {e}", exc_info=True)
            return pd.DataFrame()        

    def is_available(self, symbol: str, start_date=None, end_date=None, 
                   timeframe='1m') -> bool:
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
        filename = self._get_filename(symbol, timeframe)
        return os.path.exists(filename)
    
    def _get_filename(self, symbol: str, timeframe: str) -> str:
        """
        Get the filename for a symbol and timeframe.
        
        Args:
            symbol: Symbol
            timeframe: Timeframe
            
        Returns:
            Full path to the file
        """
        filename = self.filename_pattern.format(symbol=symbol, timeframe=timeframe)
        return os.path.join(self.data_dir, filename)

    def _map_columns(self, columns: List[str]) -> Dict[str, str]:
        """
        Map CSV columns to standard names.

        Args:
            columns: List of column names from CSV

        Returns:
            Dictionary mapping from CSV column names to standard names
        """
        result = {}

        for std_name, possible_names in self.column_map.items():
            for col in columns:
                # Compare case-insensitively
                if col.lower() in [name.lower() for name in possible_names]:
                    result[col] = std_name
                    break

        logger.debug(f"Column mapping result: {result}")
        return result
