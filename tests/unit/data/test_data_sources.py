import unittest
import os
import tempfile
import pandas as pd
import datetime
import numpy as np
from unittest.mock import MagicMock, patch

from src.data.sources.csv_handler import CSVDataSource


class TestCSVDataSource(unittest.TestCase):
    """Test cases for the CSVDataSource class."""
    
    def setUp(self):
        """Set up test environment."""
                # Get data - should skip the invalid date
        df = self.data_source.get_data('DATES', timeframe='1d')
        
        # Check that the invalid date was dropped
        self.assertEqual(len(df), 4)
        
    def test_custom_patterns(self):
        """Test custom filename patterns."""
        # Create a data source with custom pattern
        custom_source = CSVDataSource(
            self.temp_dir.name, 
            filename_pattern='{symbol}.csv'
        )
        
        # Create and save test data with custom filename
        custom_data = self.test_data.copy()
        custom_data_csv = custom_data.copy()
        custom_data_csv['date'] = custom_data_csv['date'].dt.strftime('%Y-%m-%d')
        
        custom_file = os.path.join(self.temp_dir.name, 'GOOG.csv')
        custom_data_csv.to_csv(custom_file, index=False)
        
        # Get data
        df = custom_source.get_data('GOOG')
        
        # Check data shape
        self.assertEqual(len(df), 10)
        self.assertEqual(set(df.columns), {'open', 'high', 'low', 'close', 'volume'})
        
        # Check specific values
        self.assertEqual(df.iloc[0]['close'], 101.0)
        self.assertEqual(df.iloc[-1]['close'], 110.0)
        
        # Check standard pattern no longer works
        df = custom_source.get_data('AAPL', timeframe='1d')
        self.assertTrue(df.empty)
    
    def test_custom_column_mapping(self):
        """Test custom column mapping."""
        # Create data with completely different column names
        different_cols = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='D'),
            'price_open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'price_max': [102.0, 103.0, 104.0, 105.0, 106.0],
            'price_min': [99.0, 100.0, 101.0, 102.0, 103.0],
            'price_last': [101.0, 102.0, 103.0, 104.0, 105.0],
            'quantity': [1000, 1100, 1200, 1300, 1400]
        })
        
        # Save as string dates for CSV
        different_cols_csv = different_cols.copy()
        different_cols_csv['timestamp'] = different_cols_csv['timestamp'].dt.strftime('%Y-%m-%d')
        
        # Save the test data
        custom_file = os.path.join(self.temp_dir.name, 'CUSTOM_1d.csv')
        different_cols_csv.to_csv(custom_file, index=False)
        
        # Create a data source with custom column mapping
        custom_source = CSVDataSource(
            self.temp_dir.name,
            date_column='timestamp',
            column_map={
                'open': ['price_open'],
                'high': ['price_max'],
                'low': ['price_min'],
                'close': ['price_last'],
                'volume': ['quantity']
            }
        )
        
        # Get data
        df = custom_source.get_data('CUSTOM', timeframe='1d')
        
        # Check data was loaded correctly
        self.assertEqual(len(df), 5)
        self.assertEqual(set(df.columns), {'open', 'high', 'low', 'close', 'volume'})
        
        # Check specific values (now mapped to standard column names)
        self.assertEqual(df.iloc[0]['open'], 100.0)
        self.assertEqual(df.iloc[0]['high'], 102.0)
        self.assertEqual(df.iloc[0]['low'], 99.0)
        self.assertEqual(df.iloc[0]['close'], 101.0)
        self.assertEqual(df.iloc[0]['volume'], 1000.0)
    
    def test_file_not_found(self):
        """Test handling of file not found."""
        # Get data for non-existent file
        df = self.data_source.get_data('NONEXISTENT', timeframe='1d')
        
        # Check that we got an empty DataFrame
        self.assertTrue(df.empty)
    
    def test_numeric_conversion(self):
        """Test numeric conversion of data."""
        # Create a test CSV file with string numeric values
        string_nums = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
            'open': ['100.0', '101.0', '102.0', '103.0', '104.0'],
            'high': ['102.0', '103.0', '104.0', '105.0', '106.0'],
            'low': ['99.0', '100.0', '101.0', '102.0', '103.0'],
            'close': ['101.0', '102.0', '103.0', '104.0', '105.0'],
            'volume': ['1000', '1100', '1200', '1300', '1400']
        })
        
        # Save the test data
        string_file = os.path.join(self.temp_dir.name, 'STRING_1d.csv')
        string_nums.to_csv(string_file, index=False)
        
        # Get data
        df = self.data_source.get_data('STRING', timeframe='1d')
        
        # Check data types
        self.assertTrue(pd.api.types.is_numeric_dtype(df['open']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['high']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['low']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['close']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['volume']))
        
        # Check specific values
        self.assertEqual(df.iloc[0]['open'], 100.0)
        self.assertEqual(df.iloc[0]['volume'], 1000.0)


if __name__ == '__main__':
    unittest.main() Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a test CSV file
        self.test_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
            'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'high': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            'close': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
        
        # Save as string dates for CSV
        test_data_csv = self.test_data.copy()
        test_data_csv['date'] = test_data_csv['date'].dt.strftime('%Y-%m-%d')
        
        # Save the test data
        self.test_file = os.path.join(self.temp_dir.name, 'AAPL_1d.csv')
        test_data_csv.to_csv(self.test_file, index=False)
        
        # Create data source
        self.data_source = CSVDataSource(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test initialization of CSVDataSource."""
        # Test default parameters
        self.assertEqual(self.data_source.data_dir, self.temp_dir.name)
        self.assertEqual(self.data_source.filename_pattern, '{symbol}_{timeframe}.csv')
        self.assertEqual(self.data_source.date_column, 'date')
        self.assertEqual(self.data_source.date_format, '%Y-%m-%d')
        
        # Test custom parameters
        custom_source = CSVDataSource(
            data_dir=self.temp_dir.name,
            filename_pattern='{symbol}.csv',
            date_column='timestamp',
            date_format='%d/%m/%Y'
        )
        self.assertEqual(custom_source.data_dir, self.temp_dir.name)
        self.assertEqual(custom_source.filename_pattern, '{symbol}.csv')
        self.assertEqual(custom_source.date_column, 'timestamp')
        self.assertEqual(custom_source.date_format, '%d/%m/%Y')
    
    def test_get_data(self):
        """Test getting data from the CSV source."""
        # Get data
        df = self.data_source.get_data('AAPL', timeframe='1d')
        
        # Check data shape
        self.assertEqual(len(df), 10)
        self.assertEqual(set(df.columns), {'open', 'high', 'low', 'close', 'volume'})
        
        # Check specific values
        self.assertEqual(df.iloc[0]['close'], 101.0)
        self.assertEqual(df.iloc[9]['open'], 109.0)
        
        # Check that date is set as index
        self.assertTrue(isinstance(df.index, pd.DatetimeIndex))
        self.assertEqual(df.index[0].strftime('%Y-%m-%d'), '2023-01-01')
    
    def test_get_data_different_column_names(self):
        """Test handling of different column names."""
        # Create a test CSV file with different column names
        alt_data = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'High': [102.0, 103.0, 104.0, 105.0, 106.0],
            'Low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'Close': [101.0, 102.0, 103.0, 104.0, 105.0],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        # Save as string dates for CSV
        alt_data_csv = alt_data.copy()
        alt_data_csv['Date'] = alt_data_csv['Date'].dt.strftime('%Y-%m-%d')
        
        # Save the test data
        alt_file = os.path.join(self.temp_dir.name, 'MSFT_1d.csv')
        alt_data_csv.to_csv(alt_file, index=False)
        
        # Create a custom source with uppercase date column
        custom_source = CSVDataSource(
            data_dir=self.temp_dir.name,
            date_column='Date'
        )
        
        # Get data
        df = custom_source.get_data('MSFT', timeframe='1d')
        
        # Check that data was loaded correctly
        self.assertEqual(len(df), 5)
        self.assertEqual(set(df.columns), {'open', 'high', 'low', 'close', 'volume'})
        
        # Check specific values
        self.assertEqual(df.iloc[0]['close'], 101.0)
        self.assertEqual(df.iloc[4]['open'], 104.0)
    
    def test_date_filtering(self):
        """Test filtering data by date."""
        # Get data with date filter
        start_date = '2023-01-03'
        end_date = '2023-01-07'
        df = self.data_source.get_data('AAPL', start_date=start_date, end_date=end_date, timeframe='1d')
        
        # Check filtered data
        self.assertEqual(len(df), 5)
        self.assertEqual(df.index[0].strftime('%Y-%m-%d'), '2023-01-03')
        self.assertEqual(df.index[-1].strftime('%Y-%m-%d'), '2023-01-07')
        
        # Check using datetime objects
        start_date_dt = datetime.datetime(2023, 1, 3)
        end_date_dt = datetime.datetime(2023, 1, 7)
        df_dt = self.data_source.get_data('AAPL', start_date=start_date_dt, end_date=end_date_dt, timeframe='1d')
        
        # Check filtered data is the same
        pd.testing.assert_frame_equal(df, df_dt)
    
    def test_is_available(self):
        """Test checking if data is available."""
        # Test existing symbol
        self.assertTrue(self.data_source.is_available('AAPL', timeframe='1d'))
        
        # Test non-existent symbol
        self.assertFalse(self.data_source.is_available('NONEXISTENT', timeframe='1d'))
        
        # Test different timeframe
        self.assertFalse(self.data_source.is_available('AAPL', timeframe='1h'))
    
    def test_missing_columns(self):
        """Test handling of missing columns."""
        # Create a test CSV file with missing columns
        bad_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [102.0, 103.0, 104.0, 105.0, 106.0],
            # Missing 'low' column
            'close': [101.0, 102.0, 103.0, 104.0, 105.0],
            # Missing 'volume' column
        })
        
        # Save as string dates for CSV
        bad_data_csv = bad_data.copy()
        bad_data_csv['date'] = bad_data_csv['date'].dt.strftime('%Y-%m-%d')
        
        # Save the test data
        bad_file = os.path.join(self.temp_dir.name, 'BAD_1d.csv')
        bad_data_csv.to_csv(bad_file, index=False)
        
        # Get data
        df = self.data_source.get_data('BAD', timeframe='1d')
        
        # Check that we got an empty DataFrame due to missing required columns
        self.assertTrue(df.empty)
    
    def test_invalid_dates(self):
        """Test handling of invalid dates."""
        # Create a test CSV file with invalid dates
        bad_dates = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', 'invalid', '2023-01-04', '2023-01-05'],
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [102.0, 103.0, 104.0, 105.0, 106.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [101.0, 102.0, 103.0, 104.0, 105.0],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        # Save the test data
        bad_file = os.path.join(self.temp_dir.name, 'DATES_1d.csv')
        bad_dates.to_csv(bad_file, index=False)
        
        #
