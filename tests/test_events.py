import unittest
import datetime
from core.events.event_types import EventType, Event, BarEvent, SignalEvent
from core.events.event_bus import EventBus
from core.events.event_manager import EventManager
from core.events.event_utils import create_bar_event, create_signal_event
from core.events.event_handlers import LoggingHandler

class TestEvent(unittest.TestCase):
    """Test cases for Event class."""
    
    def test_event_creation(self):
        """Test event creation and accessors."""
        event = Event(EventType.BAR, {'test': 'data'})
        
        self.assertEqual(event.get_type(), EventType.BAR)
        self.assertIsInstance(event.get_timestamp(), datetime.datetime)
        self.assertIsInstance(event.get_id(), str)
        self.assertEqual(event.data, {'test': 'data'})

class TestBarEvent(unittest.TestCase):
    """Test cases for BarEvent class."""
    
    def test_bar_event_creation(self):
        """Test bar event creation and accessors."""
        timestamp = datetime.datetime.now()
        bar = BarEvent('AAPL', timestamp, 150.0, 152.0, 149.0, 151.0, 10000)
        
        self.assertEqual(bar.get_type(), EventType.BAR)
        self.assertEqual(bar.get_symbol(), 'AAPL')
        self.assertEqual(bar.get_timestamp(), timestamp)
        self.assertEqual(bar.get_open(), 150.0)
        self.assertEqual(bar.get_high(), 152.0)
        self.assertEqual(bar.get_low(), 149.0)
        self.assertEqual(bar.get_close(), 151.0)
        self.assertEqual(bar.get_volume(), 10000)

class TestSignalEvent(unittest.TestCase):
    """Test cases for SignalEvent class."""
    
    def test_signal_event_creation(self):
        """Test signal event creation and accessors."""
        signal = SignalEvent(SignalEvent.BUY, 150.0, 'AAPL', 'test_rule')
        
        self.assertEqual(signal.get_type(), EventType.SIGNAL)
        self.assertEqual(signal.get_symbol(), 'AAPL')
        self.assertEqual(signal.get_signal_value(), SignalEvent.BUY)
        self.assertEqual(signal.get_price(), 150.0)
        self.assertTrue(signal.is_buy())
        self.assertFalse(signal.is_sell())
    
    def test_signal_validation(self):
        """Test signal value validation."""
        # Valid signal values
        SignalEvent(SignalEvent.BUY, 150.0, 'AAPL')
        SignalEvent(SignalEvent.SELL, 150.0, 'AAPL')
        SignalEvent(SignalEvent.NEUTRAL, 150.0, 'AAPL')
        
        # Invalid signal value
        with self.assertRaises(ValueError):
            SignalEvent(5, 150.0, 'AAPL')

class TestEventBus(unittest.TestCase):
    """Test cases for EventBus class."""
    
    def setUp(self):
        # Create event bus with strong references for testing
        self.event_bus = EventBus(use_weak_refs=False)
        self.events_received = []
    
    def handler(self, event):
        self.events_received.append(event)
    
    def test_register_handler(self):
        """Test handler registration."""
        self.event_bus.register(EventType.BAR, self.handler)
        self.assertIn(EventType.BAR, self.event_bus.handlers)
        # Modified test to work with strong refs or weak refs
        handler_exists = False
        for handler_ref in self.event_bus.handlers[EventType.BAR]:
            if handler_ref == self.handler or (hasattr(handler_ref, '__call__') and handler_ref() == self.handler):
                handler_exists = True
                break
        self.assertTrue(handler_exists, "Handler not found in event_bus.handlers")
    
    def test_unregister_handler(self):
        """Test handler unregistration."""
        self.event_bus.register(EventType.BAR, self.handler)
        self.event_bus.unregister(EventType.BAR, self.handler)
        # Check if handler is no longer in the list
        for handler_ref in self.event_bus.handlers[EventType.BAR]:
            if handler_ref == self.handler or (hasattr(handler_ref, '__call__') and handler_ref() == self.handler):
                self.fail("Handler still exists after unregistration")
    
    def test_emit_event(self):
        """Test event emission."""
        self.event_bus.register(EventType.BAR, self.handler)
        event = Event(EventType.BAR, {'test': 'data'})
        self.event_bus.emit(event)
        self.assertEqual(len(self.events_received), 1)
        self.assertEqual(self.events_received[0], event)
    
    def test_event_counts(self):
        """Test event counting."""
        self.event_bus.register(EventType.BAR, self.handler)
        event = Event(EventType.BAR, {'test': 'data'})
        self.event_bus.emit(event)
        self.assertEqual(self.event_bus.event_counts[EventType.BAR], 1)

class TestEventManager(unittest.TestCase):
    """Test cases for EventManager class."""
    
    def setUp(self):
        self.event_bus = EventBus(use_weak_refs=False)  # Use strong refs for testing
        self.event_manager = EventManager(self.event_bus)
        
        # Test component
        class TestComponent:
            def __init__(self):
                self.events = []
                self.bar_events = []
                self.signal_events = []
                self.event_bus = None
            
            def set_event_bus(self, event_bus):
                self.event_bus = event_bus
            
            def handle(self, event):
                self.events.append(event)
            
            def on_bar(self, event):
                self.bar_events.append(event)
            
            def on_signal(self, event):
                self.signal_events.append(event)
            
            def reset(self):
                self.events = []
                self.bar_events = []
                self.signal_events = []
        
        self.component = TestComponent()
    
    def test_register_component(self):
        """Test component registration."""
        self.event_manager.register_component(
            'test', self.component, [EventType.BAR, EventType.SIGNAL]
        )
        
        self.assertIn('test', self.event_manager.components)
        self.assertEqual(self.component.event_bus, self.event_bus)
    
    def test_event_routing(self):
        """Test event routing to components."""
        self.event_manager.register_component(
            'test', self.component, [EventType.BAR, EventType.SIGNAL]
        )
        
        # Create and emit events
        bar_event = create_bar_event(
            'AAPL', datetime.datetime.now(), 150.0, 152.0, 149.0, 151.0, 10000
        )
        signal_event = create_signal_event(
            SignalEvent.BUY, 150.0, 'AAPL', 'test_rule'
        )
        
        self.event_bus.emit(bar_event)
        self.event_bus.emit(signal_event)
        
        # Check that component received events
        self.assertEqual(len(self.component.bar_events), 1)
        self.assertEqual(len(self.component.signal_events), 1)
        self.assertEqual(self.component.bar_events[0], bar_event)
        self.assertEqual(self.component.signal_events[0], signal_event)

if __name__ == '__main__':
    unittest.main()
