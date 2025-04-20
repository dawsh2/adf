import unittest
import datetime
import asyncio
from unittest.mock import MagicMock, patch

from src.core.events.event_types import EventType, Event, BarEvent
from src.core.events.event_bus import EventBus
from src.core.events.event_emitters import EventEmitter, BarEmitter, SignalEmitter


class TestBarEmitter(unittest.TestCase):
    """Test cases for BarEmitter class."""
    
    def setUp(self):
        self.event_bus = MagicMock(spec=EventBus)
        self.emitter = BarEmitter("test_emitter", self.event_bus)
        
    def test_initialization(self):
        """Test emitter initialization."""
        self.assertEqual(self.emitter.name, "test_emitter")
        self.assertEqual(self.emitter.event_bus, self.event_bus)
        self.assertFalse(self.emitter.running)
        self.assertEqual(self.emitter.stats['emitted'], 0)
        self.assertEqual(self.emitter.stats['errors'], 0)
        self.assertEqual(self.emitter.stats['bars_by_symbol'], {})
    
    def test_start_stop(self):
        """Test starting and stopping the emitter."""
        # Initial state
        self.assertFalse(self.emitter.running)
        
        # Start emitter
        self.emitter.start()
        self.assertTrue(self.emitter.running)
        
        # Stop emitter
        self.emitter.stop()
        self.assertFalse(self.emitter.running)
    
    def test_emit_bar_event(self):
        """Test emitting a bar event."""
        # Start emitter
        self.emitter.start()
        
        # Create bar event
        timestamp = datetime.datetime.now()
        bar = BarEvent(
            symbol="AAPL",
            timestamp=timestamp,
            open_price=150.0,
            high_price=152.0,
            low_price=149.0,
            close_price=151.0,
            volume=10000
        )
        
        # Emit event
        result = self.emitter.emit(bar)
        
        # Verify result
        self.assertTrue(result)
        self.event_bus.emit.assert_called_once_with(bar)
        self.assertEqual(self.emitter.stats['emitted'], 1)
        self.assertEqual(self.emitter.stats['bars_by_symbol']['AAPL'], 1)
    
    def test_emit_while_stopped(self):
        """Test emitting while emitter is stopped."""
        # Ensure emitter is stopped
        self.emitter.stop()
        
        # Create bar event
        bar = BarEvent(
            symbol="AAPL",
            timestamp=datetime.datetime.now(),
            open_price=150.0,
            high_price=152.0,
            low_price=149.0,
            close_price=151.0,
            volume=10000
        )
        
        # Emit event
        result = self.emitter.emit(bar)
        
        # Verify result
        self.assertFalse(result)
        self.event_bus.emit.assert_not_called()
        self.assertEqual(self.emitter.stats['emitted'], 0)
    
    def test_emit_multiple_bars(self):
        """Test emitting multiple bar events."""
        # Start emitter
        self.emitter.start()
        
        # Create and emit multiple bars
        symbols = ["AAPL", "MSFT", "AAPL"]
        for symbol in symbols:
            bar = BarEvent(
                symbol=symbol,
                timestamp=datetime.datetime.now(),
                open_price=150.0,
                high_price=152.0,
                low_price=149.0,
                close_price=151.0,
                volume=10000
            )
            self.emitter.emit(bar)
        
        # Verify results
        self.assertEqual(self.emitter.stats['emitted'], 3)
        self.assertEqual(self.emitter.stats['bars_by_symbol']['AAPL'], 2)
        self.assertEqual(self.emitter.stats['bars_by_symbol']['MSFT'], 1)
        self.assertEqual(self.event_bus.emit.call_count, 3)
    
    def test_emit_error_handling(self):
        """Test error handling during emission."""
        # Start emitter
        self.emitter.start()
        
        # Make event_bus.emit raise an exception
        self.event_bus.emit.side_effect = Exception("Test error")
        
        # Create bar event
        bar = BarEvent(
            symbol="AAPL",
            timestamp=datetime.datetime.now(),
            open_price=150.0,
            high_price=152.0,
            low_price=149.0,
            close_price=151.0,
            volume=10000
        )
        
        # Emit event
        result = self.emitter.emit(bar)
        
        # Verify result
        self.assertFalse(result)
        self.event_bus.emit.assert_called_once_with(bar)
        self.assertEqual(self.emitter.stats['emitted'], 0)
        self.assertEqual(self.emitter.stats['errors'], 1)
    
    def test_reset_stats(self):
        """Test resetting statistics."""
        # Start emitter and emit events
        self.emitter.start()
        bar = BarEvent(
            symbol="AAPL",
            timestamp=datetime.datetime.now(),
            open_price=150.0,
            high_price=152.0,
            low_price=149.0,
            close_price=151.0,
            volume=10000
        )
        self.emitter.emit(bar)
        
        # Verify stats are updated
        self.assertEqual(self.emitter.stats['emitted'], 1)
        self.assertEqual(self.emitter.stats['bars_by_symbol']['AAPL'], 1)
        
        # Reset stats
        self.emitter.reset_stats()
        
        # Verify stats are reset
        self.assertEqual(self.emitter.stats['emitted'], 0)
        self.assertEqual(self.emitter.stats['errors'], 0)
        self.assertEqual(self.emitter.stats['bars_by_symbol'], {})


class TestSignalEmitter(unittest.TestCase):
    """Test cases for SignalEmitter class."""
    
    def setUp(self):
        self.event_bus = MagicMock(spec=EventBus)
        self.emitter = SignalEmitter("test_signal_emitter", self.event_bus)
    
    def test_initialization(self):
        """Test emitter initialization."""
        self.assertEqual(self.emitter.name, "test_signal_emitter")
        self.assertEqual(self.emitter.event_bus, self.event_bus)
        self.assertFalse(self.emitter.running)
        self.assertEqual(self.emitter.stats['emitted'], 0)
        self.assertEqual(self.emitter.stats['errors'], 0)
    
    def test_start_stop(self):
        """Test starting and stopping the emitter."""
        # Initial state
        self.assertFalse(self.emitter.running)
        
        # Start emitter
        self.emitter.start()
        self.assertTrue(self.emitter.running)
        
        # Stop emitter
        self.emitter.stop()
        self.assertFalse(self.emitter.running)
    
    def test_emit_signal(self):
        """Test emitting a signal."""
        # Start emitter
        self.emitter.start()
        
        # Create mock signal
        signal = MagicMock()
        
        # Emit signal
        result = self.emitter.emit_signal(signal)
        
        # Verify result
        self.assertTrue(result)
        self.event_bus.emit.assert_called_once_with(signal)
        self.assertEqual(self.emitter.stats['emitted'], 1)
    
    def test_emit_signal_while_stopped(self):
        """Test emitting signal while emitter is stopped."""
        # Ensure emitter is stopped
        self.emitter.stop()
        
        # Create mock signal
        signal = MagicMock()
        
        # Emit signal
        result = self.emitter.emit_signal(signal)
        
        # Verify result
        self.assertFalse(result)
        self.event_bus.emit.assert_not_called()
        self.assertEqual(self.emitter.stats['emitted'], 0)


if __name__ == '__main__':
    unittest.main()
