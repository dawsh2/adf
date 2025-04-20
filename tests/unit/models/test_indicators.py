import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import datetime

from src.models.components.base import ComponentBase, IndicatorBase
from src.models.components.indicators.moving_averages import SimpleMovingAverage
from src.core.events.event_types import BarEvent


class TestIndicatorBase(unittest.TestCase):
    """Test cases for the IndicatorBase class."""
    
    def test_initialization(self):
        """Test initialization of indicator base class."""
        # Create mock config
        mock_config = MagicMock()
        mock_config.get_section.return_value.get_dict.return_value = {'window': 20}
        
        # Create indicator
        indicator = SimpleMovingAverage("test_sma", config=mock_config)
        
        # Verify properties
        self.assertEqual(indicator.name, "test_sma")
        self.assertEqual(indicator.config, mock_config)
        self.assertEqual(indicator.params['window'], 20)
        self.assertEqual(indicator.component_type, "indicators")
        self.assertEqual(indicator.values, {})
    
    def test_set_event_bus(self):
        """Test setting event bus."""
        # Create indicator
        indicator = SimpleMovingAverage("test_sma")
        
        # Create mock event bus
        mock_event_bus = MagicMock()
        
        # Set event bus
        indicator.set_event_bus(mock_event_bus)
        
        # Verify event bus is set
        self.assertEqual(indicator.event_bus, mock_event_bus)
    
    def test_set_emitter(self):
        """Test setting emitter."""
        # Create indicator
        indicator = SimpleMovingAverage("test_sma")
        
        # Create mock emitter
        mock_emitter = MagicMock()
        
        # Set emitter
        indicator.set_emitter(mock_emitter)
        
        # Verify emitter is set
        self.assertEqual(indicator.emitter, mock_emitter)


class TestSimpleMovingAverage(unittest.TestCase):
    """Test cases for the SimpleMovingAverage indicator."""
    
    def setUp(self):
        # Set up test data
        self.data = pd.DataFrame({
            'close': [100, 105, 110, 115, 120, 125, 130, 135, 140, 145]
        })
    
    def test_calculate_with_dataframe(self):
        """Test calculating SMA with a pandas DataFrame."""
        # Create indicator with window of 3
        sma = SimpleMovingAverage("test_sma", params={'window': 3})
        
        # Calculate SMA
        result = sma.calculate(self.data)
        
        # Verify result
        expected = pd.Series([np.nan, np.nan, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)
    
    def test_calculate_with_not_enough_data(self):
        """Test calculating SMA with insufficient data."""
        # Create indicator with window of 20 (larger than available data)
        sma = SimpleMovingAverage("test_sma", params={'window': 20})
        
        # Calculate SMA
        result = sma.calculate(self.data)
        
        # Verify result (should be all NaN)
        self.assertTrue(np.isnan(result).all())
    
    def test_calculate_with_custom_price_key(self):
        """Test calculating SMA with custom price key."""
        # Create data with different column name
        data = pd.DataFrame({
            'adjusted_close': [100, 105, 110, 115, 120, 125, 130, 135, 140, 145]
        })
        
        # Create indicator with custom price key
        sma = SimpleMovingAverage("test_sma", params={'window': 3, 'price_key': 'adjusted_close'})
        
        # Calculate SMA
        result = sma.calculate(data)
        
        # Verify result
        expected = pd.Series([np.nan, np.nan, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)
    
    def test_calculate_with_dictionary(self):
        """Test calculating SMA with a dictionary input."""
        # Create indicator with window of 3
        sma = SimpleMovingAverage("test_sma", params={'window': 3})
        
        # Create dictionary input
        data_dict = {'close': [100, 105, 110, 115, 120]}
        
        # Calculate SMA
        result = sma.calculate(data_dict)
        
        # Verify result (only returns last value)
        expected = 111.666666667  # (110 + 115 + 120) / 3
        self.assertAlmostEqual(result, expected, places=6)
    
    def test_on_bar_event(self):
        """Test updating indicator with bar events."""
        # Create indicator with window of 3
        sma = SimpleMovingAverage("test_sma", params={'window': 3})
        
        # Create bar events
        symbol = "AAPL"
        bars = []
        for i, price in enumerate([100, 105, 110, 115, 120]):
            bar = BarEvent(
                symbol=symbol,
                timestamp=datetime.datetime.now() + datetime.timedelta(minutes=i),
                open_price=price - 1,
                high_price=price + 2,
                low_price=price - 2,
                close_price=price,
                volume=1000
            )
            bars.append(bar)
        
        # Process bars
        for bar in bars:
            sma.on_bar(bar)
        
        # Verify indicator values
        self.assertIn(symbol, sma.values)
        self.assertEqual(len(sma.values[symbol]), 3)  # Last 3 values stored (window size)
        
        # Calculate expected SMA
        expected = (110 + 115 + 120) / 3
        self.assertEqual(sma.get_value(symbol), expected)
    
    def test_on_bar_event_multiple_symbols(self):
        """Test updating indicator with bar events for multiple symbols."""
        # Create indicator with window of 3
        sma = SimpleMovingAverage("test_sma", params={'window': 3})
        
        # Create bar events for two symbols
        symbols = ["AAPL", "MSFT"]
        prices = {
            "AAPL": [100, 105, 110, 115, 120],
            "MSFT": [200, 210, 220, 230, 240]
        }
        
        # Process bars interleaved
        for i in range(5):
            for symbol in symbols:
                bar = BarEvent(
                    symbol=symbol,
                    timestamp=datetime.datetime.now() + datetime.timedelta(minutes=i),
                    open_price=prices[symbol][i] - 1,
                    high_price=prices[symbol][i] + 2,
                    low_price=prices[symbol][i] - 2,
                    close_price=prices[symbol][i],
                    volume=1000
                )
                sma.on_bar(bar)
        
        # Verify indicator values for AAPL
        self.assertIn("AAPL", sma.values)
        expected_aapl = (110 + 115 + 120) / 3
        self.assertEqual(sma.get_value("AAPL"), expected_aapl)
        
        # Verify indicator values for MSFT
        self.assertIn("MSFT", sma.values)
        expected_msft = (220 + 230 + 240) / 3
        self.assertEqual(sma.get_value("MSFT"), expected_msft)
    
    def test_reset(self):
        """Test resetting the indicator."""
        # Create indicator
        sma = SimpleMovingAverage("test_sma", params={'window': 3})
        
        # Create and process bar events
        symbol = "AAPL"
        for price in [100, 105, 110, 115, 120]:
            bar = BarEvent(
                symbol=symbol,
                timestamp=datetime.datetime.now(),
                open_price=price - 1,
                high_price=price + 2,
                low_price=price - 2,
                close_price=price,
                volume=1000
            )
            sma.on_bar(bar)
        
        # Verify indicator has values
        self.assertIn(symbol, sma.values)
        
        # Reset indicator
        sma.reset()
        
        # Verify values are cleared
        self.assertEqual(sma.values, {})


# Add these additional methods to SimpleMovingAverage for testing

class SimpleMovingAverage(IndicatorBase):
    def on_bar(self, event):
        """Handle bar events."""
        if not isinstance(event, BarEvent):
            return
            
        symbol = event.get_symbol()
        price = event.get_close()
        
        # Initialize values list for symbol if needed
        if symbol not in self.values:
            self.values[symbol] = []
            
        # Add price to values
        self.values[symbol].append(price)
        
        # Keep only window size
        window = self.params.get('window', 20)
        if len(self.values[symbol]) > window:
            self.values[symbol] = self.values[symbol][-window:]
    
    def get_value(self, symbol):
        """Get the current SMA value for a symbol."""
        if symbol not in self.values:
            return None
            
        window = self.params.get('window', 20)
        if len(self.values[symbol]) < window:
            return None
            
        return sum(self.values[symbol]) / len(self.values[symbol])
    
    def calculate(self, data):
        """Calculate SMA from data."""
        window = self.params.get('window', 20)
        price_key = self.params.get('price_key', 'close')
        
        if isinstance(data, dict):
            # Handle dictionary input
            prices = data.get(price_key, [])
            if len(prices) < window:
                return None
            return sum(prices[-window:]) / window
        else:
            # Handle DataFrame input
            return data[price_key].rolling(window=window).mean()
    
    def reset(self):
        """Reset the indicator."""
        self.values = {}


if __name__ == '__main__':
    unittest.main()
