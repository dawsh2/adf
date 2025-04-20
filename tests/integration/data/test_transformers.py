import unittest
import pandas as pd
import numpy as np
import datetime
from unittest.mock import MagicMock

from src.data.transformers.resampler import Resampler
from src.data.transformers.normalizer import Normalizer


class TestResampler(unittest.TestCase):
    """Test cases for the Resampler class."""
    
    def setUp(self):
        """Set up test data."""
        # Create hourly data
        self.dates = pd.date_range(start='2023-01-01', periods=48, freq='H')
        self.hourly_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, size=48),
            'high': np.random.uniform(110, 120, size=48),
            'low': np.random.uniform(90, 100, size=48),
            'close': np.random.uniform(100, 110, size=48),
            'volume': np.random.randint(1000, 10000, size=48)
        }, index=self.dates)
        
        # Fix high/low consistency
        for i in range(len(self.hourly_data)):
            row = self.hourly_data.iloc[i]
            open_price = row['open']
            close_price = row['close']
            
            # Make high the max of open, close, high
            self.hourly_data.loc[self.hourly_data.index[i], 'high'] = max(row['high'], open_price, close_price)
            
            # Make low the min of open, close, low
            self.hourly_data.loc[self.hourly_data.index[i], 'low'] = min(row['low'], open_price, close_price)
    
    def test_initialization(self):
        """Test initialization of resampler."""
        # Default initialization
        resampler = Resampler()
        self.assertEqual(resampler.rule, 'D')
        self.assertEqual(resampler.agg_dict, {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Custom initialization
        custom_resampler = Resampler(
            rule='H',
            agg_dict={
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'mean'
            }
        )
        self.assertEqual(custom_resampler.rule, 'H')
        self.assertEqual(custom_resampler.agg_dict['volume'], 'mean')
    
    def test_resample_hourly_to_daily(self):
        """Test resampling hourly data to daily."""
        resampler = Resampler(rule='D')
        
        # Resample hourly to daily
        daily_data = resampler.resample(self.hourly_data)
        
        # Check result shape
        self.assertEqual(len(daily_data), 2)  # 48 hours = 2 days
        
        # Check columns
        self.assertEqual(set(daily_data.columns), set(self.hourly_data.columns))
        
        # Check aggregation logic for first day
        day1_data = self.hourly_data.iloc[:24]  # First 24 hours (1 day)
        
        # Open should be the first value of the day
        self.assertEqual(daily_data.iloc[0]['open'], day1_data.iloc[0]['open'])
        
        # High should be the max of the day
        self.assertEqual(daily_data.iloc[0]['high'], day1_data['high'].max())
        
        # Low should be the min of the day
        self.assertEqual(daily_data.iloc[0]['low'], day1_data['low'].min())
        
        # Close should be the last value of the day
        self.assertEqual(daily_data.iloc[0]['close'], day1_data.iloc[-1]['close'])
        
        # Volume should be the sum of the day
        self.assertEqual(daily_data.iloc[0]['volume'], day1_data['volume'].sum())
    
    def test_resample_hourly_to_4hour(self):
        """Test resampling hourly data to 4-hour bars."""
        resampler = Resampler(rule='4H')
        
        # Resample hourly to 4-hour
        four_hour_data = resampler.resample(self.hourly_data)
        
        # Check result shape
        self.assertEqual(len(four_hour_data), 12)  # 48 hours / 4 = 12 bars
        
        # Check aggregation logic for first 4-hour bar
        bar1_data = self.hourly_data.iloc[:4]  # First 4 hours
        
        # Open should be the first value of the bar
        self.assertEqual(four_hour_data.iloc[0]['open'], bar1_data.iloc[0]['open'])
        
        # High should be the max of the bar
        self.assertEqual(four_hour_data.iloc[0]['high'], bar1_data['high'].max())
        
        # Low should be the min of the bar
        self.assertEqual(four_hour_data.iloc[0]['low'], bar1_data['low'].min())
        
        # Close should be the last value of the bar
        self.assertEqual(four_hour_data.iloc[0]['close'], bar1_data.iloc[-1]['close'])
        
        # Volume should be the sum of the bar
        self.assertEqual(four_hour_data.iloc[0]['volume'], bar1_data['volume'].sum())
    
    def test_resample_with_missing_columns(self):
        """Test resampling with missing columns."""
        # Create data with missing columns
        partial_data = self.hourly_data[['open', 'close', 'volume']].copy()
        
        # Create resampler with full agg_dict
        resampler = Resampler()
        
        # Resample data
        resampled = resampler.resample(partial_data)
        
        # Check only available columns are in result
        self.assertEqual(set(resampled.columns), set(['open', 'close', 'volume']))
        
        # Check available columns are correctly aggregated
        day1_data = partial_data.iloc[:24]  # First 24 hours (1 day)
        
        self.assertEqual(resampled.iloc[0]['open'], day1_data.iloc[0]['open'])
        self.assertEqual(resampled.iloc[0]['close'], day1_data.iloc[-1]['close'])
        self.assertEqual(resampled.iloc[0]['volume'], day1_data['volume'].sum())
    
    def test_resample_with_custom_aggregation(self):
        """Test resampling with custom aggregation functions."""
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
        daily_data = resampler.resample(self.hourly_data)
        
        # Check volume aggregation
        day1_data = self.hourly_data.iloc[:24]  # First 24 hours (1 day)
        self.assertEqual(daily_data.iloc[0]['volume'], day1_data['volume'].mean())
    
    def test_non_datetime_index(self):
        """Test handling of non-datetime index."""
        # Create data with non-datetime index
        bad_data = self.hourly_data.reset_index()
        
        # Create resampler
        resampler = Resampler()
        
        # Resample data - should raise ValueError
        with self.assertRaises(ValueError):
            resampler.resample(bad_data)
    
    def test_resample_to_timeframe(self):
        """Test resample_to_timeframe method."""
        # Create resampler
        resampler = Resampler()
        
        # Test various timeframes
        timeframes = {
            '1m': '1min',
            '5m': '5min',
            '1h': '1h',
            '4h': '4h',
            '1d': 'D',
            '1w': 'W'
        }
        
        for tf, freq in timeframes.items():
            # Test with timeframe string
            resampled = resampler.resample_to_timeframe(self.hourly_data, tf)
            
            # Compare with direct rule setting
            resampler.rule = freq
            expected = resampler.resample(self.hourly_data)
            
            # Check results match
            pd.testing.assert_frame_equal(resampled, expected)


class TestNormalizer(unittest.TestCase):
    """Test cases for the Normalizer class."""
    
    def setUp(self):
        """Set up test data."""
        # Create test data
        self.data = pd.DataFrame({
            'close': [100, 105, 110, 115, 120, 125, 130, 135, 140, 145],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
    
    def test_initialization(self):
        """Test initialization of normalizer."""
        # Default initialization
        normalizer = Normalizer()
        self.assertEqual(normalizer.method, 'zscore')
        self.assertIsNone(normalizer.columns)
        
        # Custom initialization
        custom_normalizer = Normalizer(method='minmax', columns=['close'])
        self.assertEqual(custom_normalizer.method, 'minmax')
        self.assertEqual(custom_normalizer.columns, ['close'])
        
        # Invalid method
        invalid_normalizer = Normalizer(method='invalid')
        self.assertEqual(invalid_normalizer.method, 'zscore')  # Should default to zscore
    
    def test_zscore_normalization(self):
        """Test z-score normalization."""
        normalizer = Normalizer(method='zscore')
        
        # Fit and transform
        normalized = normalizer.fit_transform(self.data)
        
        # Check shape
        self.assertEqual(normalized.shape, self.data.shape)
        
        # Check that mean is approximately 0 and std is approximately 1
        for col in self.data.columns:
            self.assertAlmostEqual(normalized[col].mean(), 0, places=10)
            self.assertAlmostEqual(normalized[col].std(), 1, places=10)
        
        # Check actual values for first row of close
        close_mean = self.data['close'].mean()
        close_std = self.data['close'].std()
        expected_first_close = (self.data['close'][0] - close_mean) / close_std
        self.assertAlmostEqual(normalized['close'][0], expected_first_close)
    
    def test_minmax_normalization(self):
        """Test min-max normalization."""
        normalizer = Normalizer(method='minmax')
        
        # Fit and transform
        normalized = normalizer.fit_transform(self.data)
        
        # Check shape
        self.assertEqual(normalized.shape, self.data.shape)
        
        # Check that values are in [0, 1]
        for col in self.data.columns:
            self.assertGreaterEqual(normalized[col].min(), 0)
            self.assertLessEqual(normalized[col].max(), 1)
        
        # Check actual values for close
        close_min = self.data['close'].min()
        close_max = self.data['close'].max()
        
        # First value should be 0 (min)
        self.assertAlmostEqual(normalized['close'][0], 0)
        
        # Last value should be 1 (max)
        self.assertAlmostEqual(normalized['close'][9], 1)
        
        # Middle value
        expected_middle = (self.data['close'][5] - close_min) / (close_max - close_min)
        self.assertAlmostEqual(normalized['close'][5], expected_middle)
    
    def test_robust_normalization(self):
        """Test robust normalization."""
        normalizer = Normalizer(method='robust')
        
        # Fit and transform
        normalized = normalizer.fit_transform(self.data)
        
        # Check shape
        self.assertEqual(normalized.shape, self.data.shape)
        
        # Check actual values for close
        close_median = self.data['close'].median()
        close_q1 = self.data['close'].quantile(0.25)
        close_q3 = self.data['close'].quantile(0.75)
        close_iqr = close_q3 - close_q1
        
        expected_first_close = (self.data['close'][0] - close_median) / close_iqr
        self.assertAlmostEqual(normalized['close'][0], expected_first_close)
    
    def test_specific_columns(self):
        """Test normalizing specific columns."""
        normalizer = Normalizer(method='zscore', columns=['close'])
        
        # Fit and transform
        normalized = normalizer.fit_transform(self.data)
        
        # Check that close is normalized
        self.assertAlmostEqual(normalized['close'].mean(), 0, places=10)
        self.assertAlmostEqual(normalized['close'].std(), 1, places=10)
        
        # Check that volume is unchanged
        pd.testing.assert_series_equal(normalized['volume'], self.data['volume'])
    
    def test_inverse_transform(self):
        """Test inverse transformation."""
        for method in ['zscore', 'minmax', 'robust']:
            normalizer = Normalizer(method=method)
            
            # Fit and transform
            normalized = normalizer.fit_transform(self.data)
            
            # Inverse transform
            restored = normalizer.inverse_transform(normalized)
            
            # Check shape
            self.assertEqual(restored.shape, self.data.shape)
            
            # Check values are restored (approximately)
            pd.testing.assert_frame_equal(restored, self.data, check_exact=False, atol=1e-10)
    
    def test_save_load_stats(self):
        """Test saving and loading normalization statistics."""
        normalizer = Normalizer(method='zscore')
        
        # Fit and get stats
        normalizer.fit(self.data)
        stats = normalizer.save_stats()
        
        # Check stats format
        self.assertIn('close', stats)
        self.assertIn('volume', stats)
        self.assertIn('mean', stats['close'])
        self.assertIn('std', stats['close'])
        
        # Create new normalizer and load stats
        new_normalizer = Normalizer(method='zscore')
        new_normalizer.load_stats(stats)
        
        # Transform with both normalizers
        transformed1 = normalizer.transform(self.data)
        transformed2 = new_normalizer.transform(self.data)
        
        # Check results are the same
        pd.testing.assert_frame_equal(transformed1, transformed2)
    
    def test_fit_transform_empty_dataframe(self):
        """Test fit_transform with empty DataFrame."""
        normalizer = Normalizer()
        empty_df = pd.DataFrame(columns=['close', 'volume'])
        
        # Fit and transform
        result = normalizer.fit_transform(empty_df)
        
        # Check result is empty but with correct columns
        self.assertTrue(result.empty)
        self.assertEqual(set(result.columns), set(['close', 'volume']))
    
    def test_transform_with_division_by_zero(self):
        """Test transform handling division by zero."""
        # Create data with constant values (std = 0)
        constant_data = pd.DataFrame({
            'close': [100, 100, 100, 100, 100],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        normalizer = Normalizer(method='zscore')
        
        # Fit and transform
        result = normalizer.fit_transform(constant_data)
        
        # Check close column (std = 0 case)
        # Should be all zeros (normalized to mean=0, std=1 when std=0)
        self.assertTrue(np.all(result['close'] == 0))
        
        # Check volume column (normal case)
        self.assertAlmostEqual(result['volume'].mean(), 0, places=10)
        self.assertAlmostEqual(result['volume'].std(), 1, places=10)


if __name__ == '__main__':
    unittest.main()
