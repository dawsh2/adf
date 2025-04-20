import unittest
import datetime
import tempfile
import os
import pandas as pd
import numpy as np

from src.core.events.event_types import EventType, Event, BarEvent, SignalEvent
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_emitters import BarEmitter, SignalEmitter
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler
from src.models.components.base import StrategyBase
from src.models.components.indicators.moving_averages import SimpleMovingAverage


class TestMovingAverageStrategy(unittest.TestCase):
    """Integration tests for a Moving Average Strategy."""
    
    def setUp(self):
        """Set up test environment."""
        # Create event system
        self.event_bus = EventBus(use_weak_refs=False)
        self.event_manager = EventManager(self.event_bus)
        
        # Create event counters
        self.event_counts = {event_type: 0 for event_type in EventType}
        self.event_data = {event_type: [] for event_type in EventType}
        
        # Create event handlers
        for event_type in EventType:
            self.event_bus.register(event_type, self._create_handler(event_type))
        
        # Create temporary directory for test data
        self.temp_dir = tempfile.TemporaryDirectory()
        self._create_test_data(self.temp_dir.name)
        
        # Create data source and handler
        self.data_source = CSVDataSource(self.temp_dir.name)
        
        # Create bar emitter
        self.bar_emitter = BarEmitter("test_bar_emitter", self.event_bus)
        self.bar_emitter.start()
        
        # Create data handler
        self.data_handler = HistoricalDataHandler(self.data_source, self.bar_emitter)
        
        # Create signal emitter
        self.signal_emitter = SignalEmitter("test_signal_emitter", self.event_bus)
        self.signal_emitter.start()
        
        # Register data handler with event manager
        self.event_manager.register_component('data_handler', self.data_handler)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
        self.bar_emitter.stop()
        self.signal_emitter.stop()
    
    def _create_handler(self, event_type):
        """Create an event handler that counts events."""
        def handler(event):
            self.event_counts[event_type] += 1
            self.event_data[event_type].append(event)
        return handler
    
    def _create_test_data(self, directory):
        """Create test data with known patterns for crossing."""
        # Create data for AAPL with price pattern causing multiple crossings
        # - First phase: price rises, causes fast MA to cross above slow MA
        # - Second phase: price falls, causes fast MA to cross below slow MA
        # - Third phase: price rises again
        dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
        
        # Create price series with specific pattern
        prices = np.zeros(60)
        
        # Phase 1: Uptrend (days 0-19)
        for i in range(20):
            prices[i] = 100 + i  # Linear uptrend from 100 to 119
        
        # Phase 2: Downtrend (days 20-39)
        for i in range(20, 40):
            prices[i] = 120 - (i - 20)  # Linear downtrend from 120 to 101
        
        # Phase 3: Uptrend again (days 40-59)
        for i in range(40, 60):
            prices[i] = 100 + (i - 40)  # Linear uptrend from 100 to 119
        
        # Add some noise for realism
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(0, 2, 60)  # Small noise
        prices += noise
        
        # Create DataFrame
        aapl_data = pd.DataFrame({
            'date': dates,
            'open': prices - 0.5,
            'high': prices + 1.0,
            'low': prices - 1.0,
            'close': prices + 0.5,
            'volume': np.random.randint(1000, 2000, 60)
        })
        
        # Save as CSV
        aapl_data_csv = aapl_data.copy()
        aapl_data_csv['date'] = aapl_data_csv['date'].dt.strftime('%Y-%m-%d')
        aapl_file = os.path.join(directory, 'AAPL_1d.csv')
        aapl_data_csv.to_csv(aapl_file, index=False)
    
    def test_moving_average_crossover(self):
        """Test a moving average crossover strategy."""
        # Create strategy
        strategy = MovingAverageCrossoverStrategy(
            name="ma_crossover",
            symbols=["AAPL"],
            fast_window=5,
            slow_window=20,
            signal_emitter=self.signal_emitter
        )
        
        # Register strategy with event manager
        self.event_manager.register_component('strategy', strategy, [EventType.BAR])
        
        # Load data
        self.data_handler.load_data(["AAPL"], timeframe="1d")
        
        # Process all bars
        while True:
            bar = self.data_handler.get_next_bar("AAPL")
            if bar is None:
                break
        
        # Verify bar events
        self.assertEqual(self.event_counts[EventType.BAR], 60)  # 60 days
        
        # Verify signals were generated
        self.assertGreater(self.event_counts[EventType.SIGNAL], 0)
        self.assertGreater(len(strategy.signals), 0)
        
        # Verify signal timing - first buy signal should be around bar 20
        # (when fast MA crosses above slow MA during initial uptrend)
        buy_signals = [s for s in strategy.signals if s.get_signal_value() == SignalEvent.BUY]
        sell_signals = [s for s in strategy.signals if s.get_signal_value() == SignalEvent.SELL]
        
        self.assertGreater(len(buy_signals), 0)
        self.assertGreater(len(sell_signals), 0)
        
        # Check sequence - should go buy, sell, buy
        signal_values = [s.get_signal_value() for s in strategy.signals]
        self.assertEqual(signal_values[0], SignalEvent.BUY)
        
        # There should be at least one buy-sell-buy sequence
        for i in range(len(signal_values) - 2):
            if signal_values[i:i+3] == [SignalEvent.BUY, SignalEvent.SELL, SignalEvent.BUY]:
                break
        else:
            self.fail("No buy-sell-buy sequence found in signals")
    
    def test_parameter_sensitivity(self):
        """Test strategy sensitivity to different parameters."""
        parameter_sets = [
            {"fast_window": 5, "slow_window": 10},  # Faster, more signals
            {"fast_window": 10, "slow_window": 30},  # Slower, fewer signals
            {"fast_window": 3, "slow_window": 5}     # Very fast, many signals
        ]
        
        results = []
        
        for params in parameter_sets:
            # Reset event counts
            for event_type in EventType:
                self.event_counts[event_type] = 0
                self.event_data[event_type] = []
            
            # Create strategy with these parameters
            strategy = MovingAverageCrossoverStrategy(
                name="ma_crossover",
                symbols=["AAPL"],
                fast_window=params["fast_window"],
                slow_window=params["slow_window"],
                signal_emitter=self.signal_emitter
            )
            
            # Register strategy with event manager
            self.event_manager.register_component('strategy', strategy, [EventType.BAR])
            
            # Load data
            self.data_handler.load_data(["AAPL"], timeframe="1d")
            self.data_handler.reset()
            
            # Process all bars
            while True:
                bar = self.data_handler.get_next_bar("AAPL")
                if bar is None:
                    break
            
            # Store results
            results.append({
                "params": params,
                "signals": len(strategy.signals),
                "buys": len([s for s in strategy.signals if s.get_signal_value() == SignalEvent.BUY]),
                "sells": len([s for s in strategy.signals if s.get_signal_value() == SignalEvent.SELL])
            })
        
        # Verify results
        self.assertEqual(len(results), 3)
        
        # Faster strategies should generate more signals
        self.assertGreater(results[2]["signals"], results[0]["signals"])
        self.assertGreater(results[0]["signals"], results[1]["signals"])
    
    def test_multiple_symbols(self):
        """Test strategy with multiple symbols."""
        # Create second test dataset with different pattern
        self._create_second_test_dataset()
        
        # Create strategy with multiple symbols
        strategy = MovingAverageCrossoverStrategy(
            name="ma_crossover",
            symbols=["AAPL", "MSFT"],
            fast_window=5,
            slow_window=20,
            signal_emitter=self.signal_emitter
        )
        
        # Register strategy with event manager
        self.event_manager.register_component('strategy', strategy, [EventType.BAR])
        
        # Load data for both symbols
        self.data_handler.load_data(["AAPL", "MSFT"], timeframe="1d")
        
        # Process all bars interleaved
        for symbol in ["AAPL", "MSFT"]:
            while True:
                bar = self.data_handler.get_next_bar(symbol)
                if bar is None:
                    break
        
        # Verify signals for each symbol
        signals_by_symbol = {"AAPL": [], "MSFT": []}
        for signal in strategy.signals:
            signals_by_symbol[signal.get_symbol()].append(signal)
        
        self.assertGreater(len(signals_by_symbol["AAPL"]), 0)
        self.assertGreater(len(signals_by_symbol["MSFT"]), 0)
        
        # Verify different signal patterns for each symbol
        aapl_signal_values = [s.get_signal_value() for s in signals_by_symbol["AAPL"]]
        msft_signal_values = [s.get_signal_value() for s in signals_by_symbol["MSFT"]]
        
        # Signals should not be identical
        self.assertNotEqual(aapl_signal_values, msft_signal_values)
    
    def _create_second_test_dataset(self):
        """Create second test dataset with different pattern."""
        # Create data for MSFT with reverse pattern to AAPL
        dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
        
        # Create price series with specific pattern (opposite of AAPL)
        prices = np.zeros(60)
        
        # Phase 1: Downtrend (days 0-19)
        for i in range(20):
            prices[i] = 120 - i  # Linear downtrend from 120 to 101
        
        # Phase 2: Uptrend (days 20-39)
        for i in range(20, 40):
            prices[i] = 100 + (i - 20)  # Linear uptrend from 100 to 119
        
        # Phase 3: Downtrend again (days 40-59)
        for i in range(40, 60):
            prices[i] = 120 - (i - 40)  # Linear downtrend from 120 to 101
        
        # Add some noise for realism
        np.random.seed(43)  # Different seed
        noise = np.random.normal(0, 2, 60)  # Small noise
        prices += noise
        
        # Create DataFrame
        msft_data = pd.DataFrame({
            'date': dates,
            'open': prices - 0.5,
            'high': prices + 1.0,
            'low': prices - 1.0,
            'close': prices + 0.5,
            'volume': np.random.randint(1000, 2000, 60)
        })
        
        # Save as CSV
        msft_data_csv = msft_data.copy()
        msft_data_csv['date'] = msft_data_csv['date'].dt.strftime('%Y-%m-%d')
        msft_file = os.path.join(self.temp_dir.name, 'MSFT_1d.csv')
        msft_data_csv.to_csv(msft_file, index=False)


# Moving Average Crossover Strategy implementation
class MovingAverageCrossoverStrategy(StrategyBase):
    """Moving Average Crossover strategy for testing."""
    
    def __init__(self, name, symbols, fast_window=5, slow_window=20, 
                config=None, container=None, signal_emitter=None, order_emitter=None):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            symbols: List of symbols to trade
            fast_window: Fast moving average window
            slow_window: Slow moving average window
            config: Configuration object
            container: DI container
            signal_emitter: Signal emitter
            order_emitter: Order emitter
        """
        super().__init__(name, config, container, signal_emitter, order_emitter)
        
        # Settings
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.fast_window = fast_window
        self.slow_window = slow_window
        
        # Indicators
        self.fast_ma = {}  # symbol -> indicator
        self.slow_ma = {}  # symbol -> indicator
        
        # State
        self.last_ma_values = {}  # symbol -> {'fast': value, 'slow': value}
        self.signals = []  # List of generated signals
        
        # Create indicators
        self._setup_indicators()
    
    def _setup_indicators(self):
        """Set up moving average indicators."""
        for symbol in self.symbols:
            # Create fast MA
            fast_name = f"{symbol}_fast_ma_{self.fast_window}"
            self.fast_ma[symbol] = SimpleMovingAverage(
                name=fast_name,
                params={'window': self.fast_window}
            )
            
            # Create slow MA
            slow_name = f"{symbol}_slow_ma_{self.slow_window}"
            self.slow_ma[symbol] = SimpleMovingAverage(
                name=slow_name,
                params={'window': self.slow_window}
            )
            
            # Initialize state
            self.last_ma_values[symbol] = {'fast': None, 'slow': None}
    
    def on_bar(self, event):
        """Process a bar event."""
        if not isinstance(event, BarEvent):
            return None
            
        symbol = event.get_symbol()
        if symbol not in self.symbols:
            return None
            
        # Update indicators
        self.fast_ma[symbol].on_bar(event)
        self.slow_ma[symbol].on_bar(event)
        
        # Get current MA values
        fast_value = self.fast_ma[symbol].get_value(symbol)
        slow_value = self.slow_ma[symbol].get_value(symbol)
        
        # Skip if not enough data
        if fast_value is None or slow_value is None:
            return None
            
        # Get previous MA values
        prev_fast = self.last_ma_values[symbol]['fast']
        prev_slow = self.last_ma_values[symbol]['slow']
        
        # Update last values
        self.last_ma_values[symbol]['fast'] = fast_value
        self.last_ma_values[symbol]['slow'] = slow_value
        
        # Skip if no previous values
        if prev_fast is None or prev_slow is None:
            return None
            
        # Check for crossover
        signal = None
        price = event.get_close()
        timestamp = event.get_timestamp()
        
        # Buy signal: fast MA crosses above slow MA
        if fast_value > slow_value and prev_fast <= prev_slow:
            signal = SignalEvent(
                signal_value=SignalEvent.BUY,
                price=price,
                symbol=symbol,
                rule_id=self.name,
                confidence=1.0,
                metadata={
                    'fast_ma': fast_value,
                    'slow_ma': slow_value
                },
                timestamp=timestamp
            )
            
        # Sell signal: fast MA crosses below slow MA
        elif fast_value < slow_value and prev_fast >= prev_slow:
            signal = SignalEvent(
                signal_value=SignalEvent.SELL,
                price=price,
                symbol=symbol,
                rule_id=self.name,
                confidence=1.0,
                metadata={
                    'fast_ma': fast_value,
                    'slow_ma': slow_value
                },
                timestamp=timestamp
            )
        
        # Emit signal if generated
        if signal:
            self.signals.append(signal)
            
            if self.signal_emitter:
                self.signal_emitter.emit(signal)
            elif self.event_bus:
                self.event_bus.emit(signal)
        
        return signal
    
    def on_signal(self, event):
        """Process a signal event."""
        # In this implementation, we don't need to handle signals
        pass
    
    def reset(self):
        """Reset the strategy state."""
        # Reset indicators
        for symbol in self.symbols:
            self.fast_ma[symbol].reset()
            self.slow_ma[symbol].reset()
            self.last_ma_values[symbol] = {'fast': None, 'slow': None}
        
        # Clear signals
        self.signals = []


if __name__ == '__main__':
    unittest.main()
