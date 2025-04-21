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



    def get_data(self, symbol: str, start_date=None, end_date=None, 
               timeframe='1d') -> pd.DataFrame:
        """Get data for a symbol within a date range."""
        filename = self._get_filename(symbol, timeframe)
        if not os.path.exists(filename):
            logger.warning(f"File not found: {filename}")
            return pd.DataFrame()

        try:
            # Read CSV
            df = pd.read_csv(filename)

            # Find date column (case-insensitive)
            date_col = None
            for col in df.columns:
                if col.lower() == self.date_column.lower():
                    date_col = col
                    break

            if date_col is None:
                logger.warning(f"Date column '{self.date_column}' not found in {df.columns}")
                return pd.DataFrame()

            # Check if required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            actual_cols = self._map_columns(df.columns)

            if not all(col in actual_cols.values() for col in required_cols):
                missing = [col for col in required_cols if col not in actual_cols.values()]
                logger.warning(f"Missing required columns: {missing}")
                return pd.DataFrame()

            # Rename columns to standard names
            df = df.rename(columns=actual_cols)

            # Convert date column to datetime
            if date_col in df.columns:
                # MODIFICATION: Convert to datetime without timezone information
                df[date_col] = pd.to_datetime(df[date_col], 
                                            format=self.date_format,
                                            errors='coerce',
                                            utc=False)  # Don't force timezone

                # Drop rows with invalid dates
                df = df.dropna(subset=[date_col])

                # MODIFICATION: Make start_date and end_date timezone-naive if they have timezone
                if start_date is not None:
                    if hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None:
                        start_date = start_date.replace(tzinfo=None)

                if end_date is not None:
                    if hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None:
                        end_date = end_date.replace(tzinfo=None)

                # Filter by date range
                if start_date:
                    if isinstance(start_date, str):
                        start_date = pd.to_datetime(start_date, utc=False)
                    df = df[df[date_col] >= start_date]

                if end_date:
                    if isinstance(end_date, str):
                        end_date = pd.to_datetime(end_date, utc=False)
                    df = df[df[date_col] <= end_date]

                # Set date as index
                df.set_index(date_col, inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error reading CSV file {filename}: {e}")
            return pd.DataFrame()

    
    def is_available(self, symbol: str, start_date=None, end_date=None, 
                   timeframe='1d') -> bool:
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
    
