# Event-Driven Trading System Usage Guide

This guide explains how to use the event-driven architecture in your algorithmic trading system.

## System Overview

The event system serves as the central nervous system of your trading application, allowing components to communicate without direct dependencies. It supports both synchronous and asynchronous operations, making it suitable for real-time market data processing via WebSockets.

## Ready-to-Use Examples

This project includes fully functional examples in the `examples/` directory:

- `event_system_demo.py`: Demonstrates synchronous event handling
- `async_event_system_demo.py`: Shows asynchronous operations with WebSockets

These examples show complete implementations of strategies, executors, and WebSocket handling.

## Basic Usage

### 1. Setting Up the Event System

```python
from core.events import EventBus, EventManager, EventType

# Create the event bus
event_bus = EventBus(use_weak_refs=True)  # Use weak references for memory safety

# Create the event manager
event_manager = EventManager(event_bus)
```

### 2. Creating Components

Components need to handle events. Here's a simple strategy component:

```python
from core.events import Event, BarEvent, SignalEvent, create_signal_event

class SimpleStrategy:
    def __init__(self, symbol):
        self.symbol = symbol
        self.event_bus = None
    
    def set_event_bus(self, event_bus):
        self.event_bus = event_bus
    
    def on_bar(self, event):
        if not isinstance(event, BarEvent) or event.get_symbol() != self.symbol:
            return
            
        # Process bar data and generate signals
        price = event.get_close()
        
        # Example signal generation
        signal = create_signal_event(
            SignalEvent.BUY, price, self.symbol, 'simple_strategy'
        )
        
        # Emit signal event
        if self.event_bus:
            self.event_bus.emit(signal)
```

### 3. Registering Components

```python
# Create components
strategy = SimpleStrategy('AAPL')
executor = SimpleExecutor()

# Register components with event manager
event_manager.register_component('strategy', strategy, [EventType.BAR])
event_manager.register_component('executor', executor, [EventType.SIGNAL])
```

### 4. Emitting Events

```python
from core.events import create_bar_event
import datetime

# Create a bar event
bar = create_bar_event(
    'AAPL', 
    datetime.datetime.now(), 
    150.0,  # Open
    152.0,  # High
    149.0,  # Low
    151.0,  # Close
    10000   # Volume
)

# Emit the event
event_bus.emit(bar)
```

## Asynchronous Usage

### 1. Setting Up Async Components

```python
import asyncio
from core.events import EventBus, EventManager, EventType

async def setup_async_system():
    # Create the event bus
    event_bus = EventBus()
    
    # Create the event manager
    event_manager = EventManager(event_bus)
    
    # Create async components
    strategy = AsyncStrategy('AAPL')
    executor = AsyncExecutor()
    
    # Register components
    event_manager.register_component('strategy', strategy, [EventType.BAR])
    event_manager.register_component('executor', executor, [EventType.SIGNAL])
    
    return event_bus, event_manager
```

### 2. Creating Async Components

```python
from core.events import BarEvent, SignalEvent, create_signal_event

class AsyncStrategy:
    def __init__(self, symbol):
        self.symbol = symbol
        self.event_bus = None
    
    def set_event_bus(self, event_bus):
        self.event_bus = event_bus
    
    async def on_bar(self, event):
        if not isinstance(event, BarEvent) or event.get_symbol() != self.symbol:
            return
            
        # Process bar data and generate signals
        price = event.get_close()
        
        # Example signal generation
        signal = create_signal_event(
            SignalEvent.BUY, price, self.symbol, 'async_strategy'
        )
        
        # Emit signal event
        if self.event_bus:
            await self.event_bus.emit_async(signal)
```

### 3. Using WebSocket for Market Data

```python
from core.events import WebSocketEmitter, EventType, WebSocketHandler

# Create WebSocket handler
ws_handler = WebSocketHandler("market_data_handler")

# Register handler with event bus
event_bus.register_async(EventType.WEBSOCKET, ws_handler.handle)

# Create WebSocket emitter for your exchange
ws_emitter = WebSocketEmitter(
    "exchange_feed",
    "wss://exchange.com/ws/market-data",
    event_bus=event_bus
)

# Start the WebSocket connection
await ws_emitter.start()
```

## Advanced Features

### Event Filtering

```python
from core.events import FilterHandler, EventType

# Create a filter that only passes AAPL events
def is_apple(event):
    return hasattr(event, 'get_symbol') and event.get_symbol() == 'AAPL'

# Create handler for filtered events
def handle_apple_events(event):
    print(f"Apple event: {event.get_close()}")
    return True

# Create filter handler
apple_filter = FilterHandler("apple_filter", is_apple, handle_apple_events)

# Register filter with event bus
event_bus.register(EventType.BAR, apple_filter.handle)
```

### Chaining Handlers

```python
from core.events import ChainHandler, FilterHandler, EventType

# Create multiple filters
apple_filter = FilterHandler("apple_filter", is_apple, handle_apple)
high_price_filter = FilterHandler("high_price", price_above_150, handle_high_price)

# Chain them together
chain = ChainHandler("filter_chain", [apple_filter, high_price_filter])

# Register chain with event bus
event_bus.register(EventType.BAR, chain.handle)
```

### Memory Management

The system uses weak references by default to prevent memory leaks:

```python
# Create event bus with weak references
event_bus = EventBus(use_weak_refs=True)

# Force cleanup of dead references
event_bus.cleanup()

# Get statistics
stats = event_bus.get_stats()
print(f"Active handlers: {stats['active_handlers']}")
```

## Testing Your Components

```python
import unittest
from core.events import EventBus, Event, EventType

class TestMyStrategy(unittest.TestCase):
    def setUp(self):
        self.event_bus = EventBus()
        self.strategy = MyStrategy('AAPL')
        self.strategy.set_event_bus(self.event_bus)
        
        # Track emitted events
        self.signal_events = []
        self.event_bus.register(EventType.SIGNAL, lambda e: self.signal_events.append(e))
    
    def test_strategy_generates_signal(self):
        # Create a bar event
        bar = create_bar_event('AAPL', datetime.datetime.now(), 150, 152, 149, 151, 10000)
        
        # Process the bar
        self.strategy.on_bar(bar)
        
        # Check if signal was generated
        self.assertEqual(len(self.signal_events), 1)
        self.assertEqual(self.signal_events[0].get_symbol(), 'AAPL')
```

For async tests, use `IsolatedAsyncioTestCase`:

```python
import unittest
import asyncio

class TestAsyncStrategy(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.event_bus = EventBus()
        self.strategy = AsyncStrategy('AAPL')
        self.strategy.set_event_bus(self.event_bus)
        
        # Track emitted events
        self.signal_events = []
        self.event_bus.register_async(EventType.SIGNAL, self.collect_signal)
        
    async def collect_signal(self, event):
        self.signal_events.append(event)
        
    async def test_strategy_generates_signal(self):
        # Create a bar event
        bar = create_bar_event('AAPL', datetime.datetime.now(), 150, 152, 149, 151, 10000)
        
        # Process the bar
        await self.strategy.on_bar(bar)
        
        # Check if signal was generated
        self.assertEqual(len(self.signal_events), 1)
```

## Running the Examples

Check out the examples in the `examples/` directory for complete implementations:

```bash
# Run the synchronous demo
python examples/event_system_demo.py

# Run the asynchronous demo
python examples/async_event_system_demo.py
```

## Best Practices

1. **Component Independence**: Design components to be independent, communicating only via events.

2. **Event Granularity**: Define events at the right level of granularity - too fine-grained creates overhead, too coarse reduces flexibility.

3. **Handler Efficiency**: Keep event handlers efficient to avoid blocking the system.

4. **Error Handling**: Use try/except in event handlers to prevent errors from crashing the system.

5. **Memory Management**: Use weak references and periodically call `cleanup()` to prevent memory leaks.

6. **Testing**: Create unit tests for components that simulate event flows.

7. **Logging**: Add logging to track event flow for debugging and monitoring.

8. **Async Awareness**: Be careful with mixing sync and async operations - use the appropriate event emission methods.

## Troubleshooting

- **Events Not Reaching Handlers**: Check that handlers are registered for the correct event types and that the event bus is properly shared.

- **Memory Leaks**: Ensure components are properly unregistered when no longer needed, or use weak references.

- **Performance Issues**: Profile handler execution times and consider batching events or optimizing handler code.

- **Async Errors**: Make sure all async functions properly use `await` and are called from an async context.

Happy trading!
