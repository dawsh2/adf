import json
import datetime
import asyncio
import inspect
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Coroutine

from collections import defaultdict


from .event_types import (
    Event, EventType, BarEvent, SignalEvent, OrderEvent, FillEvent,
    WebSocketEvent, LifecycleEvent, ErrorEvent
)

logger = logging.getLogger(__name__)
# Event creation utility functions

# Add to src/core/events/event_utils.py

class EventTracker:
    """Utility to track events passing through the system."""
    
    def __init__(self, name: str = "event_tracker", verbose: bool = False):
        """
        Initialize the event tracker.
        
        Args:
            name: Tracker name
            verbose: Whether to log detailed event information
        """
        self.name = name
        self.verbose = verbose
        self.events = defaultdict(list)
        self.event_counts = defaultdict(int)
        # Initialize logger inside the class
        self.logger = logging.getLogger(__name__)
    
    def track_event(self, event):
        """
        Track an event.
        
        Args:
            event: Event to track
        """
        event_type = event.get_type()
        self.events[event_type].append(event)
        self.event_counts[event_type] += 1
        
        # Log the event if verbose
        if self.verbose:
            self._log_event(event)
    
    def _log_event(self, event):
        """
        Log detailed event information.
        
        Args:
            event: Event to log
        """
        event_type = event.get_type()
        event_id = len(self.events[event_type])
        
        if event_type == EventType.SIGNAL:
            symbol = event.get_symbol()
            signal_value = event.get_signal_value()
            direction = "BUY" if signal_value == SignalEvent.BUY else "SELL" if signal_value == SignalEvent.SELL else "NEUTRAL"
            self.logger.debug(f"Signal [{event_id}]: {symbol} {direction}")
            
        elif event_type == EventType.ORDER:
            symbol = event.get_symbol()
            direction = event.get_direction()
            quantity = event.get_quantity()
            self.logger.debug(f"Order [{event_id}]: {symbol} {direction} {quantity}")
            
        elif event_type == EventType.FILL:
            symbol = event.get_symbol()
            direction = event.get_direction()
            quantity = event.get_quantity()
            price = event.get_price()
            self.logger.debug(f"Fill [{event_id}]: {symbol} {direction} {quantity} @ {price:.2f}")
            
        elif event_type == EventType.BAR:
            symbol = event.get_symbol()
            timestamp = event.get_timestamp()
            close = event.get_close()
            self.logger.debug(f"Bar [{event_id}]: {symbol} @ {timestamp} - Close: {close:.2f}")
            
        else:
            self.logger.debug(f"Event [{event_type.name}]: ID={event.get_id()}")
    
    def get_summary(self):
        """
        Get a summary of tracked events.
        
        Returns:
            dict: Summary of tracked events by type
        """
        return {
            event_type.name: len(events)
            for event_type, events in self.events.items()
        }
        
    def get_event_count(self, event_type):
        """
        Get the count of events of a specific type.
        
        Args:
            event_type: Type of event to count
            
        Returns:
            int: Number of events of the specified type
        """
        return self.event_counts.get(event_type, 0)
    
    def get_events(self, event_type):
        """
        Get all events of a specific type.
        
        Args:
            event_type: Type of events to retrieve
            
        Returns:
            list: Events of the specified type
        """
        return self.events.get(event_type, [])
    
    def get_last_event(self, event_type):
        """
        Get the last event of a specific type.
        
        Args:
            event_type: Type of event to retrieve
            
        Returns:
            Event or None: Last event of the specified type
        """
        events = self.events.get(event_type, [])
        return events[-1] if events else None
    
    def reset(self):
        """Reset the tracker state."""
        self.events.clear()
        self.event_counts.clear()



        
        
def create_bar_event(symbol, timestamp, open_price, high_price, 
                     low_price, close_price, volume):
    """Create a standardized bar event."""
    return BarEvent(symbol, timestamp, open_price, high_price, 
                    low_price, close_price, volume)

def create_signal_event(signal_value, price, symbol, rule_id=None, 
                        confidence=1.0, metadata=None, timestamp=None):
    """Create a standardized signal event."""
    return SignalEvent(signal_value, price, symbol, rule_id, 
                       confidence, metadata, timestamp)

def create_order_event(symbol, order_type, direction, quantity, 
                      price=None, timestamp=None):
    """Create a standardized order event."""
    return OrderEvent(symbol, order_type, direction, quantity, 
                     price, timestamp)

def create_fill_event(symbol, direction, quantity, price, 
                     commission=0.0, timestamp=None):
    """Create a standardized fill event."""
    return FillEvent(symbol, direction, quantity, price, 
                    commission, timestamp)

def create_websocket_event(connection_id, state, data=None, timestamp=None):
    """Create a standardized WebSocket event."""
    return WebSocketEvent(connection_id, state, data, timestamp)

def create_lifecycle_event(state, component=None, data=None, timestamp=None):
    """Create a standardized lifecycle event."""
    return LifecycleEvent(state, component, data, timestamp)

def create_error_event(error_type, message, source=None, exception=None, timestamp=None):
    """Create a standardized error event."""
    return ErrorEvent(error_type, message, source, exception, timestamp)

# Event serialization/deserialization functions

def event_to_dict(event):
    """Convert an event to a dictionary representation."""
    if not isinstance(event, Event):
        raise TypeError("Expected Event object")
        
    # Create base dictionary with common fields
    result = {
        'id': event.get_id(),
        'type': event.get_type().name,
        'class': event.__class__.__name__,  # Store the actual class name
        'timestamp': event.get_timestamp().isoformat(),  # Always serialize datetime
        'data': event.data
    }
    
    # Add class-specific attributes if any specific event class
    if isinstance(event, BarEvent):
        # No additional fields needed - all in data
        pass
    elif isinstance(event, SignalEvent):
        # No additional fields needed - all in data
        pass
    elif isinstance(event, OrderEvent):
        # No additional fields needed - all in data
        pass
    elif isinstance(event, FillEvent):
        # No additional fields needed - all in data
        pass
    elif isinstance(event, WebSocketEvent):
        # No additional fields needed - all in data
        pass
    elif isinstance(event, LifecycleEvent):
        # No additional fields needed - all in data
        pass
    elif isinstance(event, ErrorEvent):
        # No additional fields needed - all in data
        pass
    
    return result

def dict_to_event(event_dict):
    """Convert a dictionary to an event."""
    # Check for required fields
    event_type_name = event_dict.get('type')
    if not event_type_name:
        raise ValueError("Missing event type in dictionary")
    
    class_name = event_dict.get('class')
    if not class_name:
        # Handle backward compatibility - infer class from type
        class_name = f"{event_type_name}Event" if event_type_name != "EVENT" else "Event"
        
    # Get event type
    try:
        event_type = EventType[event_type_name]
    except KeyError:
        raise ValueError(f"Unknown event type: {event_type_name}")
    
    # Parse timestamp
    timestamp_str = event_dict.get('timestamp')
    timestamp = None
    if timestamp_str:
        if isinstance(timestamp_str, str):
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp_str)
            except ValueError:
                # Fallback for other ISO formats
                timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f")
        else:
            timestamp = timestamp_str
    
    # Get data
    data = event_dict.get('data', {})
    
    # Create appropriate event object based on class name
    if class_name == 'BarEvent':
        return BarEvent(
            symbol=data.get('symbol'),
            timestamp=timestamp,
            open_price=data.get('open'),
            high_price=data.get('high'),
            low_price=data.get('low'),
            close_price=data.get('close'),
            volume=data.get('volume')
        )
    elif class_name == 'SignalEvent':
        return SignalEvent(
            signal_value=data.get('signal_value'),
            price=data.get('price'),
            symbol=data.get('symbol'),
            rule_id=data.get('rule_id'),
            confidence=data.get('confidence', 1.0),
            metadata=data.get('metadata'),
            timestamp=timestamp
        )
    elif class_name == 'OrderEvent':
        return OrderEvent(
            symbol=data.get('symbol'),
            order_type=data.get('order_type'),
            direction=data.get('direction'),
            quantity=data.get('quantity'),
            price=data.get('price'),
            timestamp=timestamp
        )
    elif class_name == 'FillEvent':
        return FillEvent(
            symbol=data.get('symbol'),
            direction=data.get('direction'),
            quantity=data.get('quantity'),
            price=data.get('price'),
            commission=data.get('commission', 0.0),
            timestamp=timestamp
        )
    elif class_name == 'WebSocketEvent':
        return WebSocketEvent(
            connection_id=data.get('connection_id'),
            state=data.get('state'),
            data=data.get('data'),
            timestamp=timestamp
        )
    elif class_name == 'LifecycleEvent':
        return LifecycleEvent(
            state=data.get('state'),
            component=data.get('component'),
            data=data.get('data'),
            timestamp=timestamp
        )
    elif class_name == 'ErrorEvent':
        return ErrorEvent(
            error_type=data.get('error_type'),
            message=data.get('message'),
            source=data.get('source'),
            exception=data.get('exception'),
            timestamp=timestamp
        )
    else:
        # Generic event
        return Event(
            event_type=event_type,
            data=data,
            timestamp=timestamp
        )

def serialize_event(event):
    """Serialize an event to JSON string."""
    event_dict = event_to_dict(event)
    
    # Custom JSON encoder for handling datetime
    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime.datetime):
                return obj.isoformat()
            return super().default(obj)
    
    return json.dumps(event_dict, cls=DateTimeEncoder)

def deserialize_event(json_str):
    """Deserialize a JSON string to an event."""
    event_dict = json.loads(json_str)
    return dict_to_event(event_dict)

# Event filtering functions

def filter_events_by_type(events, event_type):
    """Filter a list of events by type."""
    return [event for event in events if event.get_type() == event_type]

def filter_events_by_symbol(events, symbol):
    """Filter a list of events by symbol."""
    return [
        event for event in events 
        if hasattr(event, 'get_symbol') and event.get_symbol() == symbol
    ]

def filter_events_by_time(events, start_time=None, end_time=None):
    """Filter a list of events by time range."""
    filtered = events
    
    if start_time:
        filtered = [event for event in filtered if event.get_timestamp() >= start_time]
    
    if end_time:
        filtered = [event for event in filtered if event.get_timestamp() <= end_time]
    
    return filtered

# Async helper functions

async def emit_event_async(event_bus, event):
    """Emit an event asynchronously."""
    if hasattr(event_bus, 'emit_async'):
        return await event_bus.emit_async(event)
    else:
        return event_bus.emit(event)

async def emit_events_async(event_bus, events):
    """Emit multiple events asynchronously."""
    results = []
    for event in events:
        if hasattr(event_bus, 'emit_async'):
            result = await event_bus.emit_async(event)
        else:
            result = event_bus.emit(event)
        results.append(result)
    return results

def is_async_handler(handler):
    """Check if a handler is asynchronous."""
    return inspect.iscoroutinefunction(handler) or (
        hasattr(handler, '__call__') and 
        inspect.iscoroutinefunction(handler.__call__)
    )

def wrap_sync_handler(handler):
    """Wrap a synchronous handler to be used asynchronously."""
    async def async_wrapper(event):
        return handler(event)
    return async_wrapper

def wrap_async_handler(handler, loop=None):
    """Wrap an asynchronous handler to be used synchronously."""
    def sync_wrapper(event):
        loop = loop or asyncio.get_event_loop()
        return loop.run_until_complete(handler(event))
    return sync_wrapper

# Event transformation functions

def transform_events(events, transform_fn):
    """
    Transform a list of events using a function.
    
    Args:
        events: List of events to transform
        transform_fn: Function to apply to each event
        
    Returns:
        List of transformed events
    """
    return [transform_fn(event) for event in events]

async def transform_events_async(events, transform_fn):
    """
    Transform a list of events using an async function.
    
    Args:
        events: List of events to transform
        transform_fn: Async function to apply to each event
        
    Returns:
        List of transformed events
    """
    results = []
    for event in events:
        result = await transform_fn(event)
        results.append(result)
    return results

# Event processing pipeline functions

def process_events(events, processors):
    """
    Process events through a pipeline of processors.
    
    Args:
        events: List of events to process
        processors: List of processor functions
        
    Returns:
        List of processed events
    """
    result = events
    for processor in processors:
        result = processor(result)
    return result

async def process_events_async(events, processors):
    """
    Process events through a pipeline of async processors.
    
    Args:
        events: List of events to process
        processors: List of processor functions (sync or async)
        
    Returns:
        List of processed events
    """
    result = events
    for processor in processors:
        if is_async_handler(processor):
            result = await processor(result)
        else:
            result = processor(result)
    return result
