"""
Unit tests for Resampler component.
"""
import unittest
import pandas as pd
import numpy as np
import datetime

from src.data.transformers.resampler import Resampler


class TestResampler(unittest.TestCase):
    """Test cases for Resampler class."""

    def setUp(self):
        """Set up test environment."""
        # Create test data - use 'h' instead of 'H'
        self.dates = pd.date_range(start='2023-01-01', periods=100, freq='h')
        self.test_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, size=100),
            'high': np.random.uniform(110, 120, size=100),
            'low': np.random.uniform(90, 100, size=100),
            'close': np.random.uniform(100, 110, size=100),
            'volume': np.random.randint(1000, 10000, size=100)
        }, index=self.dates)

        # Ensure high/low are actually high/low - fix copy warning
        for i in range(len(self.test_data)):
            row = self.test_data.iloc[i]
            min_price = min(row['open'], row['close'])
            max_price = max(row['open'], row['close'])

            # Use loc instead of iloc[] to avoid SettingWithCopyWarning
            self.test_data.loc[self.test_data.index[i], 'low'] = min(row['low'], min_price)
            self.test_data.loc[self.test_data.index[i], 'high'] = max(row['high'], max_price)
    

    def test_resample_daily(self):
        """Test resampling hourly data to daily."""
        # Create resampler
        resampler = Resampler(rule='D')
        
        # Resample data
        daily_data = resampler.resample(self.test_data)
        
        # Check shape
        self.assertLess(len(daily_data), len(self.test_data))
        expected_days = len(self.test_data.index.normalize().unique())
        self.assertEqual(len(daily_data), expected_days)
        
        # Check that daily data has the correct structure
        self.assertEqual(set(daily_data.columns), set(self.test_data.columns))
        
        # Check aggregation for a specific day
        day = daily_data.index[0].date()
        day_data = self.test_data[self.test_data.index.date == day]
        
        # Open should be first value of the day
        self.assertEqual(daily_data.loc[daily_data.index[0], 'open'], day_data.iloc[0]['open'])
        
        # High should be max of the day
        self.assertEqual(daily_data.loc[daily_data.index[0], 'high'], day_data['high'].max())
        
        # Low should be min of the day
        self.assertEqual(daily_data.loc[daily_data.index[0], 'low'], day_data['low'].min())
        
        # Close should be last value of the day
        self.assertEqual(daily_data.loc[daily_data.index[0], 'close'], day_data.iloc[-1]['close'])
        
        # Volume should be sum of the day
        self.assertEqual(daily_data.loc[daily_data.index[0], 'volume'], day_data['volume'].sum())

    def test_resample_weekly(self):
        """Test resampling hourly data to weekly."""
        # Create resampler
        resampler = Resampler(rule='W')

        # Resample data
        weekly_data = resampler.resample(self.test_data)

        # Check shape
        self.assertLess(len(weekly_data), len(self.test_data))

        # Check that weekly data has the correct structure
        self.assertEqual(set(weekly_data.columns), set(self.test_data.columns))

        # Get the week range for testing
        week = weekly_data.index[0]
        week_start = week - pd.Timedelta(days=week.dayofweek)
        week_end = week

        # Get data for just this week  
        week_data = self.test_data[(self.test_data.index >= week_start) & 
                                  (self.test_data.index <= week_end)]

        # High should be max of the week
        self.assertEqual(weekly_data.loc[week, 'high'], week_data['high'].max())

        # Low should be min of the week
        self.assertEqual(weekly_data.loc[week, 'low'], week_data['low'].min())

        # Volume should be sum of the week
        self.assertEqual(weekly_data.loc[week, 'volume'], week_data['volume'].sum())

    def test_resample_weekly(self):
        """Test resampling hourly data to weekly."""
        # Create resampler
        resampler = Resampler(rule='W')

        # Resample data
        weekly_data = resampler.resample(self.test_data)

        # Check shape
        self.assertLess(len(weekly_data), len(self.test_data))

        # Check that weekly data has the correct structure
        self.assertEqual(set(weekly_data.columns), set(self.test_data.columns))

        # Instead of checking specific values which can vary, 
        # let's just verify the aggregation logic:
        for week_idx in weekly_data.index:
            # Get data for this week
            mask = (self.test_data.index.floor('D') >= week_idx.floor('D') - pd.Timedelta(days=6)) & (self.test_data.index.floor('D') <= week_idx.floor('D'))
            week_data = self.test_data[mask]

            if len(week_data) > 0:
                # Check first test case: high should be maximum
                self.assertEqual(weekly_data.loc[week_idx, 'high'], week_data['high'].max())

                # Check second test case: low should be minimum
                self.assertEqual(weekly_data.loc[week_idx, 'low'], week_data['low'].min())

                # Check third test case: volume should be sum
                self.assertEqual(weekly_data.loc[week_idx, 'volume'], week_data['volume'].sum())

            # If we processed at least one week successfully, we can stop
            break
        

    def test_resample_to_timeframe(self):
        """Test resampling using timeframe notation."""
        # Create resampler
        resampler = Resampler()
        
        # Resample to 4-hour bars
        data_4h = resampler.resample_to_timeframe(self.test_data, '4h')
        
        # Check number of bars
        expected_bars = (len(self.test_data) + 3) // 4  # Ceiling division
        self.assertEqual(len(data_4h), expected_bars)
        
        # Check that 4h data has the correct structure
        self.assertEqual(set(data_4h.columns), set(self.test_data.columns))
    
    def test_custom_aggregation(self):
        """Test custom aggregation functions."""
        # Create resampler with custom aggregation
        custom_agg = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'mean'  # Use mean instead of sum for volume
        }
        resampler = Resampler(rule='D', agg_dict=custom_agg)
        
        # Resample data
        daily_data = resampler.resample(self.test_data)
        
        # Check aggregation for a specific day
        day = daily_data.index[0].date()
        day_data = self.test_data[self.test_data.index.date == day]
        
        # Volume should be mean of the day (not sum)
        self.assertEqual(daily_data.loc[daily_data.index[0], 'volume'], day_data['volume'].mean())
    
    def test_non_datetime_index(self):
        """Test handling of non-datetime index."""
        # Create data with non-datetime index
        bad_data = self.test_data.reset_index()
        
        # Create resampler
        resampler = Resampler(rule='D')
        
        # Resample data - should raise ValueError
        with self.assertRaises(ValueError):
            resampler.resample(bad_data)

    def test_timeframe_mapping(self):
        """Test timeframe mapping to pandas frequency strings."""
        # Create resampler
        resampler = Resampler()

        # Test various timeframe mappings
        timeframes = {
            '1m': '1min',
            '5m': '5min',
            '1h': '1h',
            '1d': 'D',
            '1w': 'W',
            '1M': 'ME'
        }

        for tf, freq in timeframes.items():
            # Directly set rule
            resampler.rule = freq
            result1 = resampler.resample(self.test_data)

            # Use resample_to_timeframe
            result2 = resampler.resample_to_timeframe(self.test_data, tf)

            # Both should produce same results
            pd.testing.assert_frame_equal(result1, result2)
            
 
if __name__ == '__main__':
    unittest.main()
