"""
CSV data source implementation.
"""
import os
import pandas as pd
import datetime
import pytz
import logging
from typing import Dict, List, Optional, Union, Any

from ..data_source_base import DataSourceBase

logger = logging.getLogger(__name__)

class CSVDataSource(DataSourceBase):
    """Data source for CSV files."""
    
    def __init__(self, data_dir: str, filename_pattern='{symbol}_{timeframe}.csv', 
                 date_column='date', date_format='%Y-%m-%d', 
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


    def get_data(self, symbol: str, start_date=None, end_date=None, timeframe='1d') -> pd.DataFrame:
        """Get data for a symbol within a date range."""
        filename = self._get_filename(symbol, timeframe)
        logger.info(f"CSVDataSource: Loading file {filename}")

        if not os.path.exists(filename):
            logger.warning(f"File not found: {filename}")
            return pd.DataFrame()

        try:
            # Read CSV
            df = pd.read_csv(filename)
            logger.info(f"CSVDataSource: Loaded {len(df)} rows")

            # Convert date column to datetime
            if self.date_column in df.columns:
                # 1. Parse timestamps
                logger.info(f"CSVDataSource: Original timestamp sample: {df[self.date_column].iloc[0]}")
                df[self.date_column] = pd.to_datetime(df[self.date_column])
                logger.info(f"CSVDataSource: Parsed timestamp sample: {df[self.date_column].iloc[0]}")

                # 2. CRITICAL FIX: Remove timezone info
                df[self.date_column] = df[self.date_column].dt.tz_localize(None)
                logger.info(f"CSVDataSource: Fixed timestamp sample: {df[self.date_column].iloc[0]}")

                # Rest of your code...



                # 3. Ensure filter dates are also timezone-naive
                if start_date is not None and hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None:
                    start_date = start_date.replace(tzinfo=None)
                if end_date is not None and hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None:
                    end_date = end_date.replace(tzinfo=None)

                # 4. Now all timestamps are naive, comparison will work properly
                if start_date:
                    df = df[df[self.date_column] >= start_date]
                if end_date:
                    df = df[df[self.date_column] <= end_date]

                # Set date as index
                df.set_index(self.date_column, inplace=True)

            # Map columns
            column_mapping = self._map_columns(df.columns)
            df = df.rename(columns=column_mapping)

            return df

        except Exception as e:
            logger.error(f"Error reading CSV file {filename}: {e}")
            return pd.DataFrame()        

 
 



    
    def is_available(self, symbol: str, start_date=None, end_date=None, 
                   timeframe='1m') -> bool:
        """Check if data is available for the specified parameters."""
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

        # Print for debugging
        print(f"Input columns: {columns}")
        print(f"Column map: {self.column_map}")

        for std_name, possible_names in self.column_map.items():
            for col in columns:
                # Compare case-insensitively
                if col.lower() in [name.lower() for name in possible_names]:
                    result[col] = std_name
                    break

        # Print mapping result
        print(f"Column mapping result: {result}")

        return result
    
