import unittest
import datetime
from unittest.mock import MagicMock, patch
from collections import defaultdict

from src.core.events.event_types import Event, EventType, BarEvent, SignalEvent
from src.core.events.event_bus import EventBus
from src.core.events.event_utils import create_bar_event, create_signal_event
from src.models.components.base import StrategyBase
from src.strategy.strategy_manager.manager_base import StrategyManager


class EventCollector:
    """Utility class to collect events for testing."""
    
    def __init__(self):
        """Initialize event collector."""
        self.events = defaultdict(list)
    
    def handle_event(self, event):
        """Handle an event by storing it."""
        event_type = event.get_type()
        self.events[event_type].append(event)
    
    def reset(self):
        """Reset collected events."""
        self.events = defaultdict(list)


class MockStrategy(StrategyBase):
    """Mock strategy for testing."""
    
    def __init__(self, name, signal_behavior=None, config=None, container=None,
                signal_emitter=None, order_emitter=None):
        """
        Initialize mock strategy.
        
        Args:
            name: Strategy name
            signal_behavior: Dictionary mapping symbols to signal values
            config: Configuration object
            container: DI container
            signal_emitter: Signal emitter
            order_emitter: Order emitter
        """
        super().__init__(name, config, container, signal_emitter, order_emitter)
        
        # Signal behavior determines what signals to generate
        # Format: {symbol: [{bars_required: int, signal: int}, ...]}
        self.signal_behavior = signal_behavior or {}
        
        # Track bar counts by symbol
        self.bar_counts = defaultdict(int)
        
        # Track generated signals
        self.signals = []
    
    def on_bar(self, event):
        """Process a bar event."""
        if not isinstance(event, BarEvent):
            return None
            
        symbol = event.get_symbol()
        price = event.get_close()
        
        # Update bar count for symbol
        self.bar_counts[symbol] += 1
        
        # Check if we should generate a signal based on behavior
        signal = None
        if symbol in self.signal_behavior:
            # Find matching rule
            for rule in self.signal_behavior[symbol]:
                if self.bar_counts[symbol] == rule.get('bars_required', 1):
                    signal_value = rule.get('signal')
                    confidence = rule.get('confidence', 1.0)
                    
                    # Create signal
                    signal = create_signal_event(
                        signal_value=signal_value,
                        price=price,
                        symbol=symbol,
                        rule_id=self.name,
                        confidence=confidence,
                        metadata={'bar_count': self.bar_counts[symbol]},
                        timestamp=event.get_timestamp()
                    )
                    
                    # Store signal
                    self.signals.append(signal)
                    
                    # Emit signal if we have an emitter
                    if self.signal_emitter:
                        self.signal_emitter.emit(signal)
                    elif self.event_bus:
                        self.event_bus.emit(signal)
                        
                    break
        
        return signal
    
    def on_signal(self, event):
        """Process a signal event."""
        # Simple pass-through implementation
        pass
    
    def reset(self):
        """Reset strategy state."""
        self.bar_counts = defaultdict(int)
        self.signals = []


class TestStrategyManager(unittest.TestCase):
    """Integration tests for StrategyManager."""
    
    def setUp(self):
        """Set up test environment."""
        # Create event bus
        self.event_bus = EventBus()
        
        # Create event collector
        self.collector = EventCollector()
        self.event_bus.register(EventType.SIGNAL, self.collector.handle_event)
        
        # Create strategy manager
        self.manager = StrategyManager("test_manager", self.event_bus)
    
    def test_single_strategy(self):
        """Test strategy manager with a single strategy."""
        # Create strategy with single buy signal after 3 bars
        strategy = MockStrategy(
            name="test_strategy",
            signal_behavior={
                "AAPL": [{"bars_required": 3, "signal": SignalEvent.BUY}]
            }
        )
        
        # Add strategy to manager
        self.manager.add_strategy(strategy)
        
        # Create and process bar events
        for i in range(5):
            bar = create_bar_event(
                symbol="AAPL",
                timestamp=datetime.datetime.now() + datetime.timedelta(minutes=i),
                open_price=100.0 + i,
                high_price=105.0 + i,
                low_price=95.0 + i,
                close_price=102.0 + i,
                volume=1000
            )
            self.manager.on_bar(bar)
        
        # Verify signals
        self.assertEqual(len(strategy.signals), 1)
        self.assertEqual(strategy.signals[0].get_signal_value(), SignalEvent.BUY)
        
        # Verify collected events
        self.assertEqual(len(self.collector.events[EventType.SIGNAL]), 1)
        self.assertEqual(self.collector.events[EventType.SIGNAL][0].get_signal_value(), SignalEvent.BUY)
    
    def test_multiple_strategies_agreement(self):
        """Test strategy manager with multiple strategies in agreement."""
        # Create strategies with similar signals
        strategy1 = MockStrategy(
            name="strategy1",
            signal_behavior={
                "AAPL": [{"bars_required": 3, "signal": SignalEvent.BUY, "confidence": 0.8}]
            }
        )
        
        strategy2 = MockStrategy(
            name="strategy2",
            signal_behavior={
                "AAPL": [{"bars_required": 3, "signal": SignalEvent.BUY, "confidence": 0.7}]
            }
        )
        
        # Add strategies to manager
        self.manager.add_strategy(strategy1, weight=0.6)
        self.manager.add_strategy(strategy2, weight=0.4)
        
        # Create and process bar events
        for i in range(5):
            bar = create_bar_event(
                symbol="AAPL",
                timestamp=datetime.datetime.now() + datetime.timedelta(minutes=i),
                open_price=100.0 + i,
                high_price=105.0 + i,
                low_price=95.0 + i,
                close_price=102.0 + i,
                volume=1000
            )
            self.manager.on_bar(bar)
        
        # Verify individual strategy signals
        self.assertEqual(len(strategy1.signals), 1)
        self.assertEqual(len(strategy2.signals), 1)
        
        # Verify collected events - should be one combined signal
        self.assertEqual(len(self.collector.events[EventType.SIGNAL]), 1)
        
        # Verify combined signal
        combined_signal = self.collector.events[EventType.SIGNAL][0]
        self.assertEqual(combined_signal.get_signal_value(), SignalEvent.BUY)
        self.assertEqual(combined_signal.get_symbol(), "AAPL")
        
        # Check metadata
        metadata = combined_signal.data.get('metadata', {})
        self.assertIn('votes', metadata)
        self.assertIn('strategies', metadata)
        self.assertEqual(len(metadata['strategies']), 2)
        self.assertIn('strategy1', metadata['strategies'])
        self.assertIn('strategy2', metadata['strategies'])
    
    def test_multiple_strategies_disagreement(self):
        """Test strategy manager with multiple strategies in disagreement."""
        # Create strategies with different signals
        strategy1 = MockStrategy(
            name="strategy1",
            signal_behavior={
                "AAPL": [{"bars_required": 3, "signal": SignalEvent.BUY, "confidence": 0.8}]
            }
        )
        
        strategy2 = MockStrategy(
            name="strategy2",
            signal_behavior={
                "AAPL": [{"bars_required": 3, "signal": SignalEvent.SELL, "confidence": 0.7}]
            }
        )
        
        # Add strategies to manager with equal weights
        self.manager.add_strategy(strategy1, weight=0.5)
        self.manager.add_strategy(strategy2, weight=0.5)
        
        # Create and process bar events
        for i in range(5):
            bar = create_bar_event(
                symbol="AAPL",
                timestamp=datetime.datetime.now() + datetime.timedelta(minutes=i),
                open_price=100.0 + i,
                high_price=105.0 + i,
                low_price=95.0 + i,
                close_price=102.0 + i,
                volume=1000
            )
            self.manager.on_bar(bar)
        
        # Verify individual strategy signals
        self.assertEqual(len(strategy1.signals), 1)
        self.assertEqual(len(strategy2.signals), 1)
        
        # Verify no combined signal due to tie
        self.assertEqual(len(self.collector.events[EventType.SIGNAL]), 0)
    
    def test_weighted_strategies(self):
        """Test strategy manager with weighted strategies."""
        # Create strategies with different signals and weights
        strategy1 = MockStrategy(
            name="strategy1",
            signal_behavior={
                "AAPL": [{"bars_required": 3, "signal": SignalEvent.BUY, "confidence": 0.9}]
            }
        )
        
        strategy2 = MockStrategy(
            name="strategy2",
            signal_behavior={
                "AAPL": [{"bars_required": 3, "signal": SignalEvent.SELL, "confidence": 0.8}]
            }
        )
        
        strategy3 = MockStrategy(
            name="strategy3",
            signal_behavior={
                "AAPL": [{"bars_required": 3, "signal": SignalEvent.SELL, "confidence": 0.7}]
            }
        )
        
        # Add strategies to manager with different weights
        self.manager.add_strategy(strategy1, weight=0.4)  # 40% weight
        self.manager.add_strategy(strategy2, weight=0.35) # 35% weight
        self.manager.add_strategy(strategy3, weight=0.25) # 25% weight
        
        # Create and process bar events
        for i in range(5):
            bar = create_bar_event(
                symbol="AAPL",
                timestamp=datetime.datetime.now() + datetime.timedelta(minutes=i),
                open_price=100.0 + i,
                high_price=105.0 + i,
                low_price=95.0 + i,
                close_price=102.0 + i,
                volume=1000
            )
            self.manager.on_bar(bar)
        
        # Verify individual strategy signals
        self.assertEqual(len(strategy1.signals), 1)
        self.assertEqual(len(strategy2.signals), 1)
        self.assertEqual(len(strategy3.signals), 1)
        
        # Verify combined signal - SELL should win due to combined weight
        # strategy2 (SELL, 35%) + strategy3 (SELL, 25%) = 60% > strategy1 (BUY, 40%)
        self.assertEqual(len(self.collector.events[EventType.SIGNAL]), 1)
        self.assertEqual(self.collector.events[EventType.SIGNAL][0].get_signal_value(), SignalEvent.SELL)
    
    def test_strategy_activation(self):
        """Test activating and deactivating strategies."""
        # Create strategies
        strategy1 = MockStrategy(
            name="strategy1",
            signal_behavior={
                "AAPL": [{"bars_required": 3, "signal": SignalEvent.BUY}]
            }
        )
        
        strategy2 = MockStrategy(
            name="strategy2",
            signal_behavior={
                "AAPL": [{"bars_required": 3, "signal": SignalEvent.SELL}]
            }
        )
        
        # Add strategies to manager - strategy2 initially inactive
        self.manager.add_strategy(strategy1)
        self.manager.add_strategy(strategy2, active=False)
        
        # Process first round of bars - only strategy1 active
        for i in range(5):
            bar = create_bar_event(
                symbol="AAPL",
                timestamp=datetime.datetime.now() + datetime.timedelta(minutes=i),
                open_price=100.0 + i,
                high_price=105.0 + i,
                low_price=95.0 + i,
                close_price=102.0 + i,
                volume=1000
            )
            self.manager.on_bar(bar)
        
        # Verify BUY signal from strategy1
        self.assertEqual(len(self.collector.events[EventType.SIGNAL]), 1)
        self.assertEqual(self.collector.events[EventType.SIGNAL][0].get_signal_value(), SignalEvent.BUY)
        
        # Reset
        self.collector.reset()
        strategy1.reset()
        strategy2.reset()
        
        # Deactivate strategy1, activate strategy2
        self.manager.set_strategy_active("strategy1", False)
        self.manager.set_strategy_active("strategy2", True)
        
        # Process second round of bars - only strategy2 active
        for i in range(5):
            bar = create_bar_event(
                symbol="MSFT",  # Different symbol to trigger new signals
                timestamp=datetime.datetime.now() + datetime.timedelta(minutes=i),
                open_price=200.0 + i,
                high_price=205.0 + i,
                low_price=195.0 + i,
                close_price=202.0 + i,
                volume=2000
            )
            self.manager.on_bar(bar)
        
        # No signals should be generated (different symbol)
        self.assertEqual(len(self.collector.events[EventType.SIGNAL]), 0)
        
        # Add behavior for MSFT
        strategy2.signal_behavior["MSFT"] = [{"bars_required": 3, "signal": SignalEvent.SELL}]
        
        # Reset and process again
        self.collector.reset()
        strategy1.reset()
        strategy2.reset()
        
        # Process third round of bars - only strategy2 active with MSFT behavior
        for i in range(5):
            bar = create_bar_event(
                symbol="MSFT",
                timestamp=datetime.datetime.now() + datetime.timedelta(minutes=i),
                open_price=200.0 + i,
                high_price=205.0 + i,
                low_price=195.0 + i,
                close_price=202.0 + i,
                volume=2000
            )
            self.manager.on_bar(bar)
        
        # Verify SELL signal from strategy2
        self.assertEqual(len(self.collector.events[EventType.SIGNAL]), 1)
        self.assertEqual(self.collector.events[EventType.SIGNAL][0].get_signal_value(), SignalEvent.SELL)
    
    def test_multiple_symbols(self):
        """Test strategy manager with multiple symbols."""
        # Create strategy with signals for multiple symbols
        strategy = MockStrategy(
            name="multi_symbol_strategy",
            signal_behavior={
                "AAPL": [{"bars_required": 3, "signal": SignalEvent.BUY}],
                "MSFT": [{"bars_required": 3, "signal": SignalEvent.SELL}],
                "GOOG": [{"bars_required": 3, "signal": SignalEvent.BUY}]
            }
        )
        
        # Add strategy to manager
        self.manager.add_strategy(strategy)
        
        # Process bars for multiple symbols
        symbols = ["AAPL", "MSFT", "GOOG"]
        start_prices = {"AAPL": 100.0, "MSFT": 200.0, "GOOG": 1000.0}
        
        # Send 5 bars for each symbol (interleaved)
        for i in range(5):
            for symbol in symbols:
                bar = create_bar_event(
                    symbol=symbol,
                    timestamp=datetime.datetime.now() + datetime.timedelta(minutes=i),
                    open_price=start_prices[symbol] + i,
                    high_price=start_prices[symbol] + 5 + i,
                    low_price=start_prices[symbol] - 5 + i,
                    close_price=start_prices[symbol] + 2 + i,
                    volume=1000
                )
                self.manager.on_bar(bar)
        
        # Verify signals for each symbol
        self.assertEqual(len(self.collector.events[EventType.SIGNAL]), 3)
        
        signal_by_symbol = {}
        for signal in self.collector.events[EventType.SIGNAL]:
            signal_by_symbol[signal.get_symbol()] = signal.get_signal_value()
        
        self.assertEqual(signal_by_symbol["AAPL"], SignalEvent.BUY)
        self.assertEqual(signal_by_symbol["MSFT"], SignalEvent.SELL)
        self.assertEqual(signal_by_symbol["GOOG"], SignalEvent.BUY)
    
    def test_reset(self):
        """Test resetting the strategy manager."""
        # Create strategy
        strategy = MockStrategy(
            name="test_strategy",
            signal_behavior={
                "AAPL": [{"bars_required": 3, "signal": SignalEvent.BUY}]
            }
        )
        
        # Add strategy to manager
        self.manager.add_strategy(strategy)
        
        # Process bars
        for i in range(5):
            bar = create_bar_event(
                symbol="AAPL",
                timestamp=datetime.datetime.now() + datetime.timedelta(minutes=i),
                open_price=100.0 + i,
                high_price=105.0 + i,
                low_price=95.0 + i,
                close_price=102.0 + i,
                volume=1000
            )
            self.manager.on_bar(bar)
        
        # Verify signals
        self.assertEqual(len(strategy.signals), 1)
        self.assertEqual(len(self.collector.events[EventType.SIGNAL]), 1)
        
        # Reset manager
        self.manager.reset()
        
        # Verify reset
        self.assertEqual(len(strategy.signals), 0)
        self.assertEqual(self.manager.signals, {})


if __name__ == '__main__':
    unittest.main()
