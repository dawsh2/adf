import unittest
import asyncio
import datetime
import os
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from src.core.events.event_types import EventType, Event, BarEvent, SignalEvent
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_emitters import BarEmitter, SignalEmitter
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler


class EventSystemIntegrationTest(unittest.TestCase):
    """Integration tests for event system components."""
    
    def setUp(self):
        """Set up test environment."""
        # Create event system
        self.event_bus = EventBus(use_weak_refs=False)  # Use strong refs for testing
        self.event_manager = EventManager(self.event_bus)
        
        # Create event counters
        self.event_counts = {event_type: 0 for event_type in EventType}
        self.event_data = {event_type: [] for event_type in EventType}
        
        # Create event handlers
        for event_type in EventType:
            self.event_bus.register(event_type, self._create_handler(event_type))
    
    def _create_handler(self, event_type):
        """Create an event handler that counts events."""
        def handler(event):
            self.event_counts[event_type] += 1
            self.event_data[event_type].append(event)
        return handler
    
    def test_event_flow(self):
        """Test event flow between components."""
        # Create bar emitter
        bar_emitter = BarEmitter("test_bar_emitter", self.event_bus)
        bar_emitter.start()
        
        # Create signal emitter
        signal_emitter = SignalEmitter("test_signal_emitter", self.event_bus)
        signal_emitter.start()
        
        # Create mock strategy that converts bars to signals
        class MockStrategy:
            def __init__(self, name, signal_emitter):
                self.name = name
                self.signal_emitter = signal_emitter
                self.event_bus = None
                self.processed_bars = []
                self.generated_signals = []
            
            def set_event_bus(self, event_bus):
                self.event_bus = event_bus
            
            def on_bar(self, event):
                """Handle bar events by generating signals."""
                if not isinstance(event, BarEvent):
                    return
                
                # Store bar
                self.processed_bars.append(event)
                
                # Generate a signal
                symbol = event.get_symbol()
                price = event.get_close()
                
                # Simple rule: Buy if price is above 100, sell if below
                signal_value = SignalEvent.BUY if price > 100 else SignalEvent.SELL
                
                # Create signal
                signal = SignalEvent(
                    signal_value=signal_value,
                    price=price,
                    symbol=symbol,
                    rule_id=self.name,
                    confidence=1.0,
                    metadata={'bar_id': event.get_id()},
                    timestamp=event.get_timestamp()
                )
                
                # Store signal
                self.generated_signals.append(signal)
                
                # Emit signal
                if self.signal_emitter:
                    self.signal_emitter.emit(signal)
                elif self.event_bus:
                    self.event_bus.emit(signal)
                
                return signal
        
        # Create and register mock strategy
        strategy = MockStrategy("test_strategy", signal_emitter)
        self.event_manager.register_component('strategy', strategy, [EventType.BAR])
        
        # Create mock executor
        class MockExecutor:
            def __init__(self):
                self.event_bus = None
                self.processed_signals = []
                self.trades = []
            
            def set_event_bus(self, event_bus):
                self.event_bus = event_bus
            
            def on_signal(self, event):
                """Handle signal events by executing trades."""
                if not isinstance(event, SignalEvent):
                    return
                
                # Store signal
                self.processed_signals.append(event)
                
                # Create trade
                symbol = event.get_symbol()
                price = event.get_price()
                signal_value = event.get_signal_value()
                
                trade = {
                    'symbol': symbol,
                    'price': price,
                    'action': 'BUY' if signal_value == SignalEvent.BUY else 'SELL',
                    'quantity': 100,
                    'timestamp': event.get_timestamp()
                }
                
                # Store trade
                self.trades.append(trade)
        
        # Create and register mock executor
        executor = MockExecutor()
        self.event_manager.register_component('executor', executor, [EventType.SIGNAL])
        
        # Generate bar events
        symbols = ['AAPL', 'MSFT']
        prices = {
            'AAPL': [95, 98, 102, 105, 110],
            'MSFT': [110, 105, 100, 95, 90]
        }
        
        # Emit bar events
        for i in range(5):
            for symbol in symbols:
                price = prices[symbol][i]
                bar = BarEvent(
                    symbol=symbol,
                    timestamp=datetime.datetime.now() + datetime.timedelta(minutes=i),
                    open_price=price - 1,
                    high_price=price + 2,
                    low_price=price - 2,
                    close_price=price,
                    volume=1000
                )
                bar_emitter.emit(bar)
        
        # Verify bar events
        self.assertEqual(self.event_counts[EventType.BAR], 10)  # 5 bars * 2 symbols
        
        # Verify strategy processed bars
        self.assertEqual(len(strategy.processed_bars), 10)
        
        # Verify signal events
        self.assertEqual(self.event_counts[EventType.SIGNAL], 10)
        self.assertEqual(len(strategy.generated_signals), 10)
        
        # Verify executor processed signals
        self.assertEqual(len(executor.processed_signals), 10)
        self.assertEqual(len(executor.trades), 10)
        
        # Verify signal values
        # AAPL prices go from 95 to 110, so expect SELL, SELL, BUY, BUY, BUY
        # MSFT prices go from 110 to 90, so expect BUY, BUY, SELL, SELL, SELL
        expected_actions = {
            'AAPL': ['SELL', 'SELL', 'BUY', 'BUY', 'BUY'],
            'MSFT': ['BUY', 'BUY', 'SELL', 'SELL', 'SELL']
        }
        
        for i, trade in enumerate(executor.trades):
            symbol = trade['symbol']
            index = i // 2  # Integer division to get the time index
            expected_action = expected_actions[symbol][index]
            self.assertEqual(trade['action'], expected_action)
    
    def test_event_manager_component_registration(self):
        """Test component registration and event routing."""
        # Create mock components
        class MockComponent:
            def __init__(self, name):
                self.name = name
                self.event_bus = None
                self.events = []
            
            def set_event_bus(self, event_bus):
                self.event_bus = event_bus
            
            def handle(self, event):
                self.events.append(event)
        
        # Create components with different handlers
        component1 = MockComponent("component1")
        component2 = MockComponent("component2")
        
        # Add on_bar method to component1
        def on_bar(self, event):
            self.events.append(('bar', event))
        component1.on_bar = types.MethodType(on_bar, component1)
        
        # Add on_signal method to component2
        def on_signal(self, event):
            self.events.append(('signal', event))
        component2.on_signal = types.MethodType(on_signal, component2)
        
        # Register components
        self.event_manager.register_component('component1', component1, [EventType.BAR])
        self.event_manager.register_component('component2', component2, [EventType.SIGNAL])
        
        # Emit events
        bar_event = BarEvent(
            symbol='AAPL',
            timestamp=datetime.datetime.now(),
            open_price=100.0,
            high_price=102.0,
            low_price=98.0,
            close_price=101.0,
            volume=1000
        )
        self.event_bus.emit(bar_event)
        
        signal_event = SignalEvent(
            signal_value=SignalEvent.BUY,
            price=101.0,
            symbol='AAPL',
            rule_id='test',
            confidence=1.0,
            timestamp=datetime.datetime.now()
        )
        self.event_bus.emit(signal_event)
        
        # Verify component1 received bar event
        self.assertEqual(len(component1.events), 1)
        self.assertEqual(component1.events[0][0], 'bar')
        self.assertEqual(component1.events[0][1], bar_event)
        
        # Verify component2 received signal event
        self.assertEqual(len(component2.events), 1)
        self.assertEqual(component2.events[0][0], 'signal')
        self.assertEqual(component2.events[0][1], signal_event)
    
    def test_data_handler_integration(self):
        """Test integration with data handlers."""
        # Create temporary directory for test data
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            self._create_test_data(temp_dir)
            
            # Create data source
            data_source = CSVDataSource(temp_dir)
            
            # Create bar emitter
            bar_emitter = BarEmitter("test_bar_emitter", self.event_bus)
            bar_emitter.start()
            
            # Create data handler
            data_handler = HistoricalDataHandler(data_source, bar_emitter)
            
            # Register with event manager
            self.event_manager.register_component('data_handler', data_handler)
            
            # Load data
            data_handler.load_data(['AAPL', 'MSFT'], timeframe='1d')
            
            # Process all bars
            for symbol in ['AAPL', 'MSFT']:
                while True:
                    bar = data_handler.get_next_bar(symbol)
                    if bar is None:
                        break
            
            # Verify bar events were emitted
            self.assertEqual(self.event_counts[EventType.BAR], 20)  # 10 days * 2 symbols
            
            # Verify bar data
            bars_by_symbol = {'AAPL': [], 'MSFT': []}
            for event in self.event_data[EventType.BAR]:
                symbol = event.get_symbol()
                bars_by_symbol[symbol].append(event)
            
            self.assertEqual(len(bars_by_symbol['AAPL']), 10)
            self.assertEqual(len(bars_by_symbol['MSFT']), 10)
            
            # Verify first and last bars
            self.assertEqual(bars_by_symbol['AAPL'][0].get_close(), 101.0)
            self.assertEqual(bars_by_symbol['AAPL'][-1].get_close(), 110.0)
            self.assertEqual(bars_by_symbol['MSFT'][0].get_close(), 201.0)
            self.assertEqual(bars_by_symbol['MSFT'][-1].get_close(), 210.0)
    
    def _create_test_data(self, directory):
        """Create test data files."""
        # Create test data for AAPL
        aapl_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
            'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'high': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            'close': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
        
        # Create test data for MSFT
        msft_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
            'open': [200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0, 209.0],
            'high': [202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0, 209.0, 210.0, 211.0],
            'low': [199.0, 200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0],
            'close': [201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0, 209.0, 210.0],
            'volume': [2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900]
        })
        
        # Save as string dates for CSV
        aapl_data_csv = aapl_data.copy()
        aapl_data_csv['date'] = aapl_data_csv['date'].dt.strftime('%Y-%m-%d')
        
        msft_data_csv = msft_data.copy()
        msft_data_csv['date'] = msft_data_csv['date'].dt.strftime('%Y-%m-%d')
        
        # Save the test data
        aapl_file = os.path.join(directory, 'AAPL_1d.csv')
        msft_file = os.path.join(directory, 'MSFT_1d.csv')
        
        aapl_data_csv.to_csv(aapl_file, index=False)
        msft_data_csv.to_csv(msft_file, index=False)


# Need to import types for method binding
import types

if __name__ == '__main__':
    unittest.main()
