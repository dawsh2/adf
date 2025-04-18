"""
Integration tests for data module with event system.
"""
import unittest
import os
import tempfile
import logging
import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import core event system components
from src.core.events.event_types import EventType, Event, BarEvent, SignalEvent
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_utils import create_bar_event, create_signal_event
from src.core.events.event_handlers import LoggingHandler, FilterHandler, ChainHandler

# Import data module components
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler
from src.data.transformers.resampler import Resampler
from src.data.transformers.normalizer import Normalizer


def create_test_data(temp_dir, symbols=None, days=30):
    """
    Create test data files for testing.
    
    Args:
        temp_dir: Temporary directory
        symbols: List of symbols to create (default: ['AAPL', 'MSFT'])
        days: Number of days of data to create
        
    Returns:
        Dictionary mapping symbols to dataframes
    """
    if symbols is None:
        symbols = ['AAPL', 'MSFT']
        
    start_date = datetime.datetime(2023, 1, 1)
    df_dict = {}
    
    for symbol in symbols:
        # Set seed for reproducibility with different patterns per symbol
        np.random.seed(hash(symbol) % 2**32)
        
        # Create base price series - random walk with trend
        base_price = 100.0 * (symbols.index(symbol) + 1)  # Different starting prices
        prices = np.zeros(days)
        prices[0] = base_price
        
        # Add trend and volatility
        daily_return = 0.001 * (symbols.index(symbol) + 1)  # Different trends
        volatility = 0.01 * (symbols.index(symbol) + 1)     # Different volatilities
        
        for i in range(1, days):
            # Random walk with drift
            change = np.random.normal(daily_return, volatility) * prices[i-1]
            prices[i] = prices[i-1] + change
        
        # Create dataframe
        dates = pd.date_range(start=start_date, periods=days, freq='D')
        df = pd.DataFrame(index=dates, columns=['open', 'high', 'low', 'close', 'volume'])
        
        # Fill data
        for i in range(days):
            price = prices[i]
            daily_range = price * 0.02  # 2% daily range
            
            df.iloc[i]['open'] = price - daily_range/4 + np.random.uniform(-daily_range/8, daily_range/8)
            df.iloc[i]['high'] = price + daily_range/2 + np.random.uniform(0, daily_range/4)
            df.iloc[i]['low'] = price - daily_range/2 - np.random.uniform(0, daily_range/4)
            df.iloc[i]['close'] = price + np.random.uniform(-daily_range/4, daily_range/4)
            df.iloc[i]['volume'] = int(np.random.uniform(1000, 10000))
        
        # Save to CSV
        filename = os.path.join(temp_dir, f"{symbol}_1d.csv")
        df.to_csv(filename, index=True)
        
        # Store dataframe
        df_dict[symbol] = df
    
    return df_dict


class EventCounter:
    """Component to count events by type."""
    
    def __init__(self):
        """Initialize the counter."""
        self.counts = defaultdict(int)
        self.events = defaultdict(list)
    
    def handle_event(self, event):
        """
        Handle an event by counting it.
        
        Args:
            event: Event to handle
        """
        event_type = event.get_type()
        self.counts[event_type] += 1
        self.events[event_type].append(event)
    
    def reset(self):
        """Reset the counter."""
        self.counts = defaultdict(int)
        self.events = defaultdict(list)


class TestDataEventIntegration(unittest.TestCase):
    """
    Test integration between data and event systems.
    
    This test suite verifies that the data module works correctly with the
    event system, focusing on the flow of events between components.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test data
        self.test_data = create_test_data(self.temp_dir.name)
        
        # Create event system
        self.event_bus = EventBus()
        self.event_manager = EventManager(self.event_bus)
        
        # Create data source and handler
        self.data_source = CSVDataSource(self.temp_dir.name)
        self.data_handler = HistoricalDataHandler(self.data_source, self.event_bus)
        
        # Create event counter
        self.event_counter = EventCounter()
        self.event_bus.register(EventType.BAR, self.event_counter.handle_event)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_event_flow_single_symbol(self):
        """Test event flow with a single symbol."""
        # Load data
        self.data_handler.load_data('AAPL', timeframe='1d')
        
        # Process all bars
        bars_processed = 0
        while True:
            bar = self.data_handler.get_next_bar('AAPL')
            if bar is None:
                break
            bars_processed += 1
        
        # Check that events were emitted
        self.assertEqual(self.event_counter.counts[EventType.BAR], bars_processed)
        self.assertEqual(len(self.event_counter.events[EventType.BAR]), bars_processed)
        
        # Check bar content
        for event in self.event_counter.events[EventType.BAR]:
            self.assertEqual(event.get_symbol(), 'AAPL')
            self.assertIsNotNone(event.get_timestamp())
            self.assertIsInstance(event.get_open(), float)
            self.assertIsInstance(event.get_high(), float)
            self.assertIsInstance(event.get_low(), float)
            self.assertIsInstance(event.get_close(), float)
            self.assertIsInstance(event.get_volume(), int)
    
    def test_event_flow_multiple_symbols(self):
        """Test event flow with multiple symbols."""
        symbols = ['AAPL', 'MSFT']
        
        # Load data for multiple symbols
        self.data_handler.load_data(symbols, timeframe='1d')
        
        # Process all bars for each symbol
        total_bars = 0
        for symbol in symbols:
            bars_processed = 0
            while True:
                bar = self.data_handler.get_next_bar(symbol)
                if bar is None:
                    break
                bars_processed += 1
                total_bars += 1
        
        # Check that events were emitted
        self.assertEqual(self.event_counter.counts[EventType.BAR], total_bars)
        
        # Check symbol distribution
        symbols_in_events = [event.get_symbol() for event in self.event_counter.events[EventType.BAR]]
        for symbol in symbols:
            symbol_count = symbols_in_events.count(symbol)
            self.assertGreater(symbol_count, 0)
            self.assertEqual(symbol_count, 30)  # We created 30 days of data
    
    def test_filter_handlers(self):
        """Test using filter handlers with data events."""
        symbols = ['AAPL', 'MSFT']
        
        # Create symbol-specific counters
        apple_counter = EventCounter()
        msft_counter = EventCounter()
        
        # Create filter functions
        def is_apple(event):
            return hasattr(event, 'get_symbol') and event.get_symbol() == 'AAPL'
            
        def is_msft(event):
            return hasattr(event, 'get_symbol') and event.get_symbol() == 'MSFT'
        
        # Create filter handlers
        apple_filter = FilterHandler("apple_filter", is_apple, apple_counter.handle_event)
        msft_filter = FilterHandler("msft_filter", is_msft, msft_counter.handle_event)
        
        # Register filters with event bus
        self.event_bus.register(EventType.BAR, apple_filter.handle)
        self.event_bus.register(EventType.BAR, msft_filter.handle)
        
        # Load data and process
        self.data_handler.load_data(symbols, timeframe='1d')
        
        # Process all bars for each symbol
        for symbol in symbols:
            while True:
                bar = self.data_handler.get_next_bar(symbol)
                if bar is None:
                    break
        
        # Check filter counts
        self.assertEqual(apple_counter.counts[EventType.BAR], 30)  # 30 days of AAPL data
        self.assertEqual(msft_counter.counts[EventType.BAR], 30)   # 30 days of MSFT data
        
        # Verify the filter worked correctly
        for event in apple_counter.events[EventType.BAR]:
            self.assertEqual(event.get_symbol(), 'AAPL')
            
        for event in msft_counter.events[EventType.BAR]:
            self.assertEqual(event.get_symbol(), 'MSFT')


class SimpleStrategy:
    """Simple strategy for integration testing."""
    
    def __init__(self, symbols=None, fast_ma=5, slow_ma=15):
        """
        Initialize the strategy.
        
        Args:
            symbols: List of symbols to trade
            fast_ma: Fast moving average window
            slow_ma: Slow moving average window
        """
        self.symbols = symbols or []
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.prices = {symbol: [] for symbol in self.symbols}
        self.signals = []
        self.last_signal_type = {}  # symbol -> last signal type
        self.event_bus = None
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
    
    def add_symbol(self, symbol):
        """Add a symbol to the strategy."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            self.prices[symbol] = []
    
    def on_bar(self, event):
        """Handle bar events."""
        if not isinstance(event, BarEvent):
            return
            
        symbol = event.get_symbol()
        if symbol not in self.symbols:
            return
            
        # Add price to history
        price = event.get_close()
        self.prices[symbol].append(price)
        
        # Keep only necessary history
        max_window = max(self.fast_ma, self.slow_ma)
        if len(self.prices[symbol]) > max_window + 10:
            self.prices[symbol] = self.prices[symbol][-(max_window + 10):]
            
        # Check if we have enough data
        if len(self.prices[symbol]) < self.slow_ma:
            return
            
        # Calculate MAs
        fast_ma_value = sum(self.prices[symbol][-self.fast_ma:]) / self.fast_ma
        slow_ma_value = sum(self.prices[symbol][-self.slow_ma:]) / self.slow_ma
        
        # Generate signal on crossover
        signal_value = None
        
        # Only check crossover if we have at least slow_ma + 1 prices
        if len(self.prices[symbol]) > self.slow_ma:
            # Previous MAs
            prev_fast_ma = sum(self.prices[symbol][-(self.fast_ma+1):-1]) / self.fast_ma
            prev_slow_ma = sum(self.prices[symbol][-(self.slow_ma+1):-1]) / self.slow_ma
            
            # Buy signal: fast MA crosses above slow MA
            if fast_ma_value > slow_ma_value and prev_fast_ma <= prev_slow_ma:
                signal_value = SignalEvent.BUY
                
            # Sell signal: fast MA crosses below slow MA
            elif fast_ma_value < slow_ma_value and prev_fast_ma >= prev_slow_ma:
                signal_value = SignalEvent.SELL
        
        # Only emit if we have a signal and it's different from the last one
        if signal_value is not None:
            # Check if this is different from the last signal
            if symbol not in self.last_signal_type or self.last_signal_type[symbol] != signal_value:
                # Create signal
                signal = create_signal_event(
                    signal_value=signal_value,
                    price=price,
                    symbol=symbol,
                    rule_id='ma_crossover',
                    metadata={
                        'fast_ma': fast_ma_value,
                        'slow_ma': slow_ma_value
                    },
                    timestamp=event.get_timestamp()
                )
                
                # Store signal
                self.signals.append(signal)
                self.last_signal_type[symbol] = signal_value
                
                # Emit signal
                if self.event_bus:
                    self.event_bus.emit(signal)
                    
                if signal_value == SignalEvent.BUY:
                    logger.debug(f"BUY signal for {symbol} at {price:.2f}")
                else:
                    logger.debug(f"SELL signal for {symbol} at {price:.2f}")
    
    def reset(self):
        """Reset the strategy."""
        self.prices = {symbol: [] for symbol in self.symbols}
        self.signals = []
        self.last_signal_type = {}


class TestStrategyEventIntegration(unittest.TestCase):
    """
    Test integration between data, event system, and strategy.
    
    This test suite verifies that the data flows correctly through the
    event system to strategies and that strategy signals are emitted correctly.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test data with predictable patterns
        symbols = ['AAPL', 'MSFT']
        days = 60
        
        # Create dataframes with specific patterns that will cause crossovers
        for i, symbol in enumerate(symbols):
            # Create a dataframe with a specific pattern
            dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
            df = pd.DataFrame(index=dates, columns=['open', 'high', 'low', 'close', 'volume'])
            
            # Fill data with a specific pattern:
            # - First 20 days: uptrend
            # - Next 20 days: downtrend
            # - Last 20 days: uptrend again
            
            base_price = 100.0 * (i + 1)  # Different starting price for each symbol
            daily_change = 2.0  # Price changes for trend
            
            for day in range(days):
                if day < 20:
                    # Uptrend
                    price = base_price + day * daily_change
                elif day < 40:
                    # Downtrend
                    price = base_price + 20 * daily_change - (day - 20) * daily_change
                else:
                    # Uptrend again
                    price = base_price - (day - 40) * daily_change
                
                # Add some noise
                noise = np.random.uniform(-1.0, 1.0)
                price += noise
                
                # Fill OHLC values
                df.iloc[day]['open'] = price - 0.5
                df.iloc[day]['high'] = price + 1.0
                df.iloc[day]['low'] = price - 1.0
                df.iloc[day]['close'] = price + 0.5
                df.iloc[day]['volume'] = int(np.random.uniform(1000, 10000))
            
            # Save to CSV
            filename = os.path.join(self.temp_dir.name, f"{symbol}_1d.csv")
            df.to_csv(filename, index=True)
        
        # Create event system
        self.event_bus = EventBus()
        self.event_manager = EventManager(self.event_bus)
        
        # Create data source and handler
        self.data_source = CSVDataSource(self.temp_dir.name)
        self.data_handler = HistoricalDataHandler(self.data_source, self.event_bus)
        
        # Create strategy
        self.strategy = SimpleStrategy(symbols, fast_ma=5, slow_ma=15)
        
        # Create event counters
        self.bar_counter = EventCounter()
        self.signal_counter = EventCounter()
        
        # Register components with event system
        self.event_manager.register_component('strategy', self.strategy, [EventType.BAR])
        self.event_bus.register(EventType.BAR, self.bar_counter.handle_event)
        self.event_bus.register(EventType.SIGNAL, self.signal_counter.handle_event)
        
        # Load data
        self.data_handler.load_data(symbols, timeframe='1d')
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_strategy_signal_generation(self):
        """Test that the strategy generates signals based on data events."""
        symbols = ['AAPL', 'MSFT']
        
        # Process all bars
        for symbol in symbols:
            while True:
                bar = self.data_handler.get_next_bar(symbol)
                if bar is None:
                    break
        
        # Check that we received bar events
        self.assertEqual(self.bar_counter.counts[EventType.BAR], 60 * len(symbols))  # 60 days * 2 symbols
        
        # Check that signals were generated
        self.assertGreater(self.signal_counter.counts[EventType.SIGNAL], 0)
        self.assertGreater(len(self.strategy.signals), 0)
        
        # Verify signal details
        for signal in self.signal_counter.events[EventType.SIGNAL]:
            self.assertIn(signal.get_symbol(), symbols)
            self.assertIn(signal.get_signal_value(), [SignalEvent.BUY, SignalEvent.SELL])
            
            # Check metadata
            metadata = signal.data.get('metadata', {})
            self.assertIn('fast_ma', metadata)
            self.assertIn('slow_ma', metadata)
            
            # Verify crossover condition
            if signal.get_signal_value() == SignalEvent.BUY:
                self.assertGreater(metadata['fast_ma'], metadata['slow_ma'])
            else:
                self.assertLess(metadata['fast_ma'], metadata['slow_ma'])
    
    def test_reset_and_reprocess(self):
        """Test resetting and reprocessing data."""
        symbols = ['AAPL', 'MSFT']
        
        # Process all bars once
        for symbol in symbols:
            while True:
                bar = self.data_handler.get_next_bar(symbol)
                if bar is None:
                    break
        
        # Store signal count
        signal_count = self.signal_counter.counts[EventType.SIGNAL]
        
        # Reset components
        self.data_handler.reset()
        self.strategy.reset()
        self.bar_counter.reset()
        self.signal_counter.reset()
        
        # Process all bars again
        for symbol in symbols:
            while True:
                bar = self.data_handler.get_next_bar(symbol)
                if bar is None:
                    break
        
        # Check that we got the same number of signals
        self.assertEqual(self.signal_counter.counts[EventType.SIGNAL], signal_count)


if __name__ == '__main__':
    unittest.main()
