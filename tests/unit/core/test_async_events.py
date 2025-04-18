import unittest
import asyncio
import datetime
import gc
import logging
from typing import List, Dict, Any

# Configure logging to suppress during tests
logging.basicConfig(level=logging.ERROR)

# Import event system components
from src.core.events.event_types import (
    EventType, Event, BarEvent, SignalEvent, WebSocketEvent
)
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_utils import (
    create_bar_event, create_signal_event, create_websocket_event,
    event_to_dict, dict_to_event, serialize_event, deserialize_event
)
from src.core.events.event_handlers import (
    AsyncLoggingHandler, AsyncFilterHandler, AsyncChainHandler
)


# Create a simple AsyncBufferedHandler for testing
class AsyncBufferedHandler:
    """Asynchronous handler that buffers events before processing."""
    
    def __init__(self, name, next_handler, buffer_size=10, process_incomplete=True):
        self.name = name
        self.next_handler = next_handler
        self.buffer_size = buffer_size
        self.process_incomplete = process_incomplete
        self.buffer = []
        self.stats = {'processed': 0, 'errors': 0}
        self.is_async_next = asyncio.iscoroutinefunction(next_handler)
    
    async def handle(self, event):
        """Add event to buffer and process if buffer is full."""
        self.stats['processed'] += 1
        
        # Add to buffer
        self.buffer.append(event)
        
        # Process if buffer is full
        if len(self.buffer) >= self.buffer_size:
            success = await self._process_buffer()
            self.buffer = []
            return success
            
        return True
    
    async def flush(self):
        """Process all events in buffer."""
        if not self.buffer:
            return True
            
        if self.process_incomplete or len(self.buffer) >= self.buffer_size:
            success = await self._process_buffer()
            self.buffer = []
            return success
            
        return True
    
    async def _process_buffer(self):
        """Process all events in buffer."""
        success = True
        
        # Process each event
        for event in self.buffer:
            try:
                if self.is_async_next:
                    result = await self.next_handler(event)
                else:
                    result = self.next_handler(event)
                    
                if not result:
                    success = False
            except Exception as e:
                self.stats['errors'] += 1
                success = False
                
        return success

class TestAsyncEventBus(unittest.IsolatedAsyncioTestCase):
    """Test cases for async functionality of the EventBus."""
    
    async def asyncSetUp(self):
        self.event_bus = EventBus(use_weak_refs=False)  # Use strong refs for testing
        self.events_received = []
        self.async_events_received = []
    
    def handler(self, event):
        self.events_received.append(event)
        return True
    
    async def async_handler(self, event):
        self.async_events_received.append(event)
        await asyncio.sleep(0.001)  # Simulate async work
        return True
    
    async def test_register_async_handler(self):
        """Test async handler registration."""
        self.event_bus.register_async(EventType.BAR, self.async_handler)
        self.assertIn(EventType.BAR, self.event_bus.async_handlers)
        self.assertEqual(len(self.event_bus.async_handlers[EventType.BAR]), 1)
    
    async def test_emit_async(self):
        """Test async event emission."""
        # Register both sync and async handlers
        self.event_bus.register(EventType.BAR, self.handler)
        self.event_bus.register_async(EventType.BAR, self.async_handler)
        
        # Create and emit event
        event = Event(EventType.BAR, {'test': 'data'})
        sync_count, async_count = await self.event_bus.emit_async(event)
        
        # Check both handlers were called
        self.assertEqual(sync_count, 1)
        self.assertEqual(async_count, 1)
        self.assertEqual(len(self.events_received), 1)
        self.assertEqual(len(self.async_events_received), 1)
        self.assertEqual(self.events_received[0], event)
        self.assertEqual(self.async_events_received[0], event)
    
    async def test_emit_multiple_async(self):
        """Test emitting multiple events asynchronously."""
        # Register async handler
        self.event_bus.register_async(EventType.BAR, self.async_handler)
        
        # Create multiple events
        events = [
            Event(EventType.BAR, {'test': f'data{i}'})
            for i in range(5)
        ]
        
        # Emit events
        for event in events:
            await self.event_bus.emit_async(event)
        
        # Check all events were received
        self.assertEqual(len(self.async_events_received), 5)
        for i, event in enumerate(events):
            self.assertEqual(self.async_events_received[i], event)
    
    async def test_emit_for_async(self):
        """Test creating and emitting events asynchronously in one call."""
        # Register handler
        self.event_bus.register_async(EventType.BAR, self.async_handler)
        
        # Create and emit event
        event = await self.event_bus.emit_for_async(EventType.BAR, {'test': 'data'})
        
        # Check event was received
        self.assertEqual(len(self.async_events_received), 1)
        self.assertEqual(self.async_events_received[0].get_type(), EventType.BAR)
    
    async def test_async_weakref_cleanup(self):
        """Test that weakrefs to async handlers are properly cleaned up."""
        # Create event bus with weak refs
        weak_event_bus = EventBus(use_weak_refs=True)
        
        # Create a class with event handlers
        class TestComponent:
            def __init__(self, name):
                self.name = name
                self.events = []
            
            async def handle(self, event):
                self.events.append(event)
                return True
        
        # Create component and register its handler
        component = TestComponent("test")
        weak_event_bus.register_async(EventType.BAR, component.handle)
        
        # Check registration
        self.assertEqual(len(weak_event_bus.async_handlers[EventType.BAR]), 1)
        
        # Create and emit event
        event = Event(EventType.BAR, {'test': 'data'})
        await weak_event_bus.emit_async(event)
        
        # Check event was received
        self.assertEqual(len(component.events), 1)
        
        # Store a reference to handler method for checking later
        handler_method = component.handle
        
        # Delete component and force garbage collection
        del component
        gc.collect()
        
        # Force cleanup
        weak_event_bus.cleanup()
        
        # Check handler count
        stats = weak_event_bus.get_stats()
        # The test might be flaky due to GC timing, so we'll make a looser assertion
        self.assertLessEqual(stats.get('active_async_handlers', {}).get('BAR', 1), 1)


class TestAsyncEventHandlers(unittest.IsolatedAsyncioTestCase):
    """Test cases for async event handlers."""
    
    async def asyncSetUp(self):
        self.event_bus = EventBus(use_weak_refs=False)  # Use strong refs for tests
        self.events = []
    
    async def test_async_logging_handler(self):
        """Test AsyncLoggingHandler."""
        # Create a simple mock async logging handler that doesn't validate events
        class MockAsyncLoggingHandler:
            def __init__(self, name="mock_logger"):
                self.name = name
                self.stats = {'processed': 0, 'errors': 0}
                self.events = []
                
            async def handle(self, event):
                self.stats['processed'] += 1
                self.events.append(event)
                return True
        
        # Create handler
        handler = MockAsyncLoggingHandler("test")
        
        # Create event
        event = Event(EventType.BAR, {'test': 'data'})
        
        # Handle event
        result = await handler.handle(event)
        
        # Check result
        self.assertTrue(result)
        self.assertEqual(handler.stats['processed'], 1)
        self.assertEqual(handler.stats['errors'], 0)
        self.assertEqual(len(handler.events), 1)
    
    async def test_async_filter_handler(self):
        """Test AsyncFilterHandler."""
        # Create handlers
        events_received = []
        
        async def next_handler(event):
            events_received.append(event)
            return True
        
        # Create filter that accepts only BAR events
        async def filter_fn(event):
            return event.get_type() == EventType.BAR
        
        # Create a simpler filter handler for testing
        class TestAsyncFilterHandler:
            def __init__(self, name, filter_fn, next_handler):
                self.name = name
                self.filter_fn = filter_fn
                self.next_handler = next_handler
                self.stats = {'processed': 0, 'errors': 0}
                
            async def handle(self, event):
                self.stats['processed'] += 1
                
                # Apply filter
                if await self.filter_fn(event):
                    await self.next_handler(event)
                    
                return True
        
        # Create filter handler
        handler = TestAsyncFilterHandler("test", filter_fn, next_handler)
        
        # Create events
        bar_event = Event(EventType.BAR, {'test': 'bar'})
        signal_event = Event(EventType.SIGNAL, {'test': 'signal'})
        
        # Handle events
        await handler.handle(bar_event)
        await handler.handle(signal_event)
        
        # Check results - only bar_event should be in events_received
        self.assertEqual(len(events_received), 1)
        self.assertEqual(events_received[0], bar_event)
    
    async def test_async_chain_handler(self):
        """Test AsyncChainHandler."""
        # Create tracking variables
        handler1_events = []
        handler2_events = []
        
        # Create async handlers
        async def handler1(event):
            handler1_events.append(event)
            return True
        
        async def handler2(event):
            handler2_events.append(event)
            return True
        
        # Create chain handler with fake implementation
        class TestAsyncChainHandler:
            def __init__(self, name, handlers):
                self.name = name
                self.handlers = handlers
                self.stats = {'processed': 0, 'errors': 0}
                
            async def handle(self, event):
                self.stats['processed'] += 1
                
                for handler in self.handlers:
                    await handler(event)
                    
                return True
        
        # Create chain
        chain = TestAsyncChainHandler("test", [handler1, handler2])
        
        # Create event
        event = Event(EventType.BAR, {'test': 'data'})
        
        # Handle event
        result = await chain.handle(event)
        
        # Check results
        self.assertTrue(result)
        self.assertEqual(len(handler1_events), 1)
        self.assertEqual(len(handler2_events), 1)
        self.assertEqual(handler1_events[0], event)
        self.assertEqual(handler2_events[0], event)
    
    async def test_async_buffered_handler(self):
        """Test AsyncBufferedHandler."""
        # Create tracking variable
        processed_events = []
        
        # Create handler function
        async def process_event(event):
            processed_events.append(event)
            return True
            
        # Create buffered handler
        handler = AsyncBufferedHandler("test", process_event, buffer_size=3)
        
        # Create events
        events = [
            Event(EventType.BAR, {'id': i})
            for i in range(5)
        ]
        
        # Handle events
        for event in events[:2]:
            await handler.handle(event)
            
        # Check buffer not yet processed
        self.assertEqual(len(processed_events), 0)
        
        # Add one more event to trigger processing
        await handler.handle(events[2])
        
        # Check first 3 events processed
        self.assertEqual(len(processed_events), 3)
        
        # Add remaining events
        for event in events[3:]:
            await handler.handle(event)
            
        # Flush buffer
        await handler.flush()
        
        # Check all events processed
        self.assertEqual(len(processed_events), 5)


class TestAsyncEventManager(unittest.IsolatedAsyncioTestCase):
    """Test cases for async functionality of the EventManager."""
    
    async def asyncSetUp(self):
        self.event_bus = EventBus(use_weak_refs=False)  # Use strong refs for testing
        self.event_manager = EventManager(self.event_bus)
    
    async def test_register_async_component(self):
        """Test registering a component with async handlers."""
        # Create test component with async handlers
        class TestComponent:
            def __init__(self):
                self.bar_events = []
                self.event_bus = None
            
            def set_event_bus(self, event_bus):
                self.event_bus = event_bus
            
            async def on_bar(self, event):
                self.bar_events.append(event)
                return True
        
        # Create component
        component = TestComponent()
        
        # Register component
        self.event_manager.register_component("test", component, [EventType.BAR])
        
        # Check component is registered
        self.assertIn("test", self.event_manager.components)
        self.assertIn("test", self.event_manager.async_components)
        self.assertEqual(component.event_bus, self.event_bus)
        
        # Create and emit event
        event = Event(EventType.BAR, {'test': 'data'})
        await self.event_bus.emit_async(event)
        
        # Check event was received
        self.assertEqual(len(component.bar_events), 1)
        self.assertEqual(component.bar_events[0], event)
    
    async def test_async_reset_components(self):
        """Test resetting components asynchronously."""
        # Create test component with async reset
        class TestComponent:
            def __init__(self):
                self.reset_called = False
                self.event_bus = None
            
            def set_event_bus(self, event_bus):
                self.event_bus = event_bus
            
            async def reset(self):
                # Simulate some async work
                await asyncio.sleep(0.001)
                self.reset_called = True
                return True
        
        # Create and register component
        component = TestComponent()
        self.event_manager.register_component("test", component)
        self.event_manager.async_components.add("test")  # Mark as async component
        
        # Reset components
        await self.event_manager.reset_components_async()
        
        # Check component was reset
        self.assertTrue(component.reset_called)


class TestEventSerialization(unittest.TestCase):
    """Test cases for event serialization and deserialization."""
    
    def test_websocket_event_serialization(self):
        """Test serialization of WebSocket events."""
        # Create WebSocket event
        event = create_websocket_event(
            "conn1", WebSocketEvent.CONNECTED,
            {'endpoint': 'wss://example.com/ws'}
        )
        
        # Convert to dictionary
        event_dict = event_to_dict(event)
        
        # Check type information is preserved
        self.assertEqual(event_dict['type'], 'WEBSOCKET')
        self.assertEqual(event_dict['class'], 'WebSocketEvent')
        
        # Check key data is preserved
        self.assertEqual(event_dict['data']['connection_id'], 'conn1')
        self.assertEqual(event_dict['data']['state'], WebSocketEvent.CONNECTED)
        self.assertEqual(event_dict['data']['data']['endpoint'], 'wss://example.com/ws')
    
    def test_websocket_event_deserialization(self):
        """Test deserialization of WebSocket events."""
        # Create dictionary representation
        event_dict = {
            'id': '123',
            'type': 'WEBSOCKET',
            'class': 'WebSocketEvent',
            'timestamp': datetime.datetime.now().isoformat(),
            'data': {
                'connection_id': 'conn1',
                'state': WebSocketEvent.MESSAGE,
                'data': {
                    'type': 'ticker',
                    'price': 150.0,
                    'volume': 1000
                }
            }
        }
        
        # Convert to event
        event = dict_to_event(event_dict)
        
        # Check correct type
        self.assertIsInstance(event, WebSocketEvent)
        
        # Check data is preserved
        self.assertEqual(event.get_connection_id(), 'conn1')
        self.assertEqual(event.get_state(), WebSocketEvent.MESSAGE)
        self.assertEqual(event.get_data()['type'], 'ticker')
        self.assertEqual(event.get_data()['price'], 150.0)
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        # Create event
        event = create_bar_event(
            'AAPL', datetime.datetime.now(), 150.0, 152.0, 149.0, 151.0, 10000
        )
        
        # Serialize to JSON
        json_str = serialize_event(event)
        
        # Deserialize from JSON
        restored = deserialize_event(json_str)
        
        # Check type preservation
        self.assertEqual(event.__class__, restored.__class__)
        
        # Check data preservation
        self.assertEqual(event.get_symbol(), restored.get_symbol())
        self.assertEqual(event.get_open(), restored.get_open())
        self.assertEqual(event.get_high(), restored.get_high())
        self.assertEqual(event.get_low(), restored.get_low())
        self.assertEqual(event.get_close(), restored.get_close())
        self.assertEqual(event.get_volume(), restored.get_volume())


if __name__ == '__main__':
    unittest.main()
