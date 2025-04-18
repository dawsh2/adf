"""
Unit tests for CSV data source.
"""
import unittest
import os
import tempfile
import pandas as pd
import datetime
import numpy as np

from src.data.sources.csv_handler import CSVDataSource


class TestCSVDataSource(unittest.TestCase):
    """Test cases for CSVDataSource class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
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
        
        # Convert dates to strings for CSV
        date_col = self.test_data['date'].copy()
        self.test_data['date'] = self.test_data['date'].dt.strftime('%Y-%m-%d')
        
        # Save the test data
        self.test_file = os.path.join(self.temp_dir.name, 'AAPL_1d.csv')
        self.test_data.to_csv(self.test_file, index=False)
        
        # Restore the datetime column for comparison
        self.test_data['date'] = date_col
        
        # Create data source
        self.data_source = CSVDataSource(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_get_data(self):
        """Test getting data from the CSV source."""
        # Get data
        data = self.data_source.get_data('AAPL', timeframe='1d')
        
        # Check data shape
        self.assertEqual(len(data), 10)
        self.assertEqual(set(data.columns), {'open', 'high', 'low', 'close', 'volume'})
        
        # Check specific values
        self.assertEqual(data.iloc[0]['close'], 101.0)
        self.assertEqual(data.iloc[9]['open'], 109.0)
    
    def test_date_filtering(self):
        """Test filtering data by date."""
        # Get data with date filter
        data = self.data_source.get_data('AAPL', start_date='2023-01-03', 
                                       end_date='2023-01-07', timeframe='1d')
        
        # Check filtered data
        self.assertEqual(len(data), 5)
        self.assertEqual(data.iloc[0]['close'], 103.0)
        self.assertEqual(data.iloc[-1]['close'], 107.0)
    
    def test_is_available(self):
        """Test checking if data is available."""
        # Test existing symbol
        self.assertTrue(self.data_source.is_available('AAPL', timeframe='1d'))
        
        # Test non-existent symbol
        self.assertFalse(self.data_source.is_available('NONEXISTENT', timeframe='1d'))


    def test_different_column_names(self):
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

        # Convert dates to strings for CSV
        date_col = alt_data['Date'].copy()
        alt_data['Date'] = alt_data['Date'].dt.strftime('%Y-%m-%d')

        # Save the test data
        alt_file = os.path.join(self.temp_dir.name, 'MSFT_1d.csv')
        alt_data.to_csv(alt_file, index=False)

        # Let's see what's in the created CSV
        print(f"\nCreated file {alt_file}")
        with open(alt_file, 'r') as f:
            print(f.read())

        # Create a new data source with custom date column name
        custom_source = CSVDataSource(
            self.temp_dir.name,
            date_column='Date'  # Specify the uppercase date column name
        )

        # Get data
        data = custom_source.get_data('MSFT', timeframe='1d')

        # Print the data for debugging
        print(f"\nReturned data shape: {data.shape}")
        if not data.empty:
            print(f"Returned data columns: {data.columns.tolist()}")
            print(f"Returned data head:\n{data.head()}")
        else:
            print("Returned data is empty")

        # Check data shape
        self.assertEqual(len(data), 5)
        self.assertEqual(set(data.columns), {'open', 'high', 'low', 'close', 'volume'})

        # Check specific values
        self.assertEqual(data.iloc[0]['close'], 101.0)
        self.assertEqual(data.iloc[-1]['close'], 105.0)        

 

    
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
        
        # Save the test data
        bad_file = os.path.join(self.temp_dir.name, 'BAD_1d.csv')
        bad_data.to_csv(bad_file, index=False)
        
        # Get data - should return empty DataFrame due to missing required columns
        data = self.data_source.get_data('BAD', timeframe='1d')
        
        # Check that we got an empty DataFrame
        self.assertTrue(data.empty)
    
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
        
        # Get data - should skip the invalid date
        data = self.data_source.get_data('DATES', timeframe='1d')
        
        # Check that the invalid date was dropped
        self.assertEqual(len(data), 4)
    
    def test_custom_patterns(self):
        """Test custom filename patterns."""
        # Create a data source with custom pattern
        custom_source = CSVDataSource(
            self.temp_dir.name, 
            filename_pattern='{symbol}.csv'
        )
        
        # Create and save test data with custom filename
        custom_data = self.test_data.copy()
        custom_file = os.path.join(self.temp_dir.name, 'GOOG.csv')
        custom_data.to_csv(custom_file, index=False)
        
        # Get data
        data = custom_source.get_data('GOOG')
        
        # Check data shape
        self.assertEqual(len(data), 10)
        self.assertEqual(set(data.columns), {'open', 'high', 'low', 'close', 'volume'})
        
        # Check specific values
        self.assertEqual(data.iloc[0]['close'], 101.0)
        self.assertEqual(data.iloc[-1]['close'], 110.0)
        
        # Check standard pattern no longer works
        data = custom_source.get_data('AAPL', timeframe='1d')
        self.assertTrue(data.empty)


if __name__ == '__main__':
    unittest.main()
