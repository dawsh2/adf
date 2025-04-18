import unittest
import asyncio
import datetime
import gc
import logging
from typing import List, Dict, Any

# Configure logging to suppress during tests
logging.basicConfig(level=logging.ERROR)

# Import event system components
from core.events.event_types import (
    EventType, Event, BarEvent, SignalEvent, WebSocketEvent
)
from core.events.event_bus import EventBus
from core.events.event_manager import EventManager
from core.events.event_utils import (
    create_bar_event, create_signal_event, create_websocket_event,
    event_to_dict, dict_to_event, serialize_event, deserialize_event
)
from core.events.event_handlers import (
    AsyncLoggingHandler, AsyncFilterHandler, AsyncChainHandler, AsyncBufferedHandler
)

class TestAsyncEventBus(unittest.IsolatedAsyncioTestCase):
    """Test cases for async functionality of the EventBus."""
    
    async def asyncSetUp(self):
        self.event_bus = EventBus(use_weak_refs=True)
        self.events_received = []
        self.async_events_received = []
    
    def handler(self, event):
        self.events_received.append(event)
    
    async def async_handler(self, event):
        self.async_events_received.append(event)
        await asyncio.sleep(0.001)  # Simulate async work
    
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
        self.assertEqual(self.async_events_received[0], event)
    
    async def test_async_weakref_cleanup(self):
        """Test that weakrefs to async handlers are properly cleaned up."""
        import gc
        
        class TestComponent:
            def __init__(self, name):
                self.name = name
                self.events = []
            
            async def handle(self, event):
                self.events.append(event)
        
        # Create component and register its handler
        component = TestComponent("test")
        self.event_bus.register_async(EventType.BAR, component.handle)
        
        # Check registration
        self.assertEqual(len(self.event_bus.async_handlers[EventType.BAR]), 1)
        
        # Create and emit event
        event = Event(EventType.BAR, {'test': 'data'})
        await self.event_bus.emit_async(event)
        
        # Check event was received
        self.assertEqual(len(component.events), 1)
        
        # Delete component and force garbage collection
        del component
        gc.collect()
        
        # Force cleanup
        self.event_bus.cleanup()
        
        # Create and emit another event
        another_event = Event(EventType.BAR, {'test': 'more data'})
        await self.event_bus.emit_async(another_event)
        
        # Check handler count
        stats = self.event_bus.get_stats()
        self.assertEqual(stats['active_async_handlers'].get('BAR', 0), 0)


class TestAsyncEventHandlers(unittest.IsolatedAsyncioTestCase):
    """Test cases for async event handlers."""
    
    async def asyncSetUp(self):
        self.event_bus = EventBus()
        self.events = []
    
    async def test_async_logging_handler(self):
        """Test AsyncLoggingHandler."""
        handler = AsyncLoggingHandler("test")
        
        # Create event
        event = Event(EventType.BAR, {'test': 'data'})
        
        # Handle event
        result = await handler.handle(event)
        
        # Check result
        self.assertTrue(result)
        self.assertEqual(handler.stats['processed'], 1)
        self.assertEqual(handler.stats['errors'], 0)
    
    async def test_async_filter_handler(self):
        """Test AsyncFilterHandler."""
        # Create handlers
        async def next_handler(event):
            self.events.append(event)
            return True
        
        # Create filter that only accepts BAR events
        async def filter_fn(event):
            return event.get_type() == EventType.BAR
        
        handler = AsyncFilterHandler("test", filter_fn, next_handler)
        
        # Create events
        bar_event = Event(EventType.BAR, {'test': 'bar'})
        signal_event = Event(EventType.SIGNAL, {'test': 'signal'})
        
        # Handle events
        await handler.handle(bar_event)
        await handler.handle(signal_event)
        
        # Check results
        self.assertEqual(len(self.events), 1)
        self.assertEqual(self.events[0], bar_event)
        self.assertEqual(handler.stats['processed'], 2)
    
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
        
        # Create chain
        chain = AsyncChainHandler("test", [handler1, handler2])
        
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
        
        # Create handlers
        async def process_events(events):
            processed_events.extend(events)
            return True
        
        # Create buffered handler
        async def next_handler(event):
            processed_events.append(event)
            return True
            
        handler = AsyncBufferedHandler("test", next_handler, buffer_size=3)
        
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
        self.event_bus = EventBus()
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
                await asyncio.sleep(0.001)  # Simulate async work
                self.reset_called = True
        
        # Create and register component
        component = TestComponent()
        self.event_manager.register_component("test", component)
        
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
