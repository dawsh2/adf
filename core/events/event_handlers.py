import logging
import asyncio
import inspect
from typing import Dict, Any, Optional, List, Callable, Union, Coroutine
from abc import ABC, abstractmethod

from .event_types import Event, EventType
from .event_schema import SchemaValidator

logger = logging.getLogger(__name__)

class EventHandler(ABC):
    """Abstract base class for event handlers."""
    
    def __init__(self, name):
        self.name = name
        self.validator = SchemaValidator()
        self.stats = {
            'processed': 0,
            'errors': 0
        }
    
    @abstractmethod
    def handle(self, event):
        """Handle an event."""
        pass
    
    def validate_event(self, event):
        """Validate an event."""
        is_valid, error = self.validator.validate(event.get_type(), event.data)
        return is_valid, error
    
    def reset_stats(self):
        """Reset handler statistics."""
        self.stats = {
            'processed': 0,
            'errors': 0
        }
    
    def get_stats(self):
        """Get handler statistics."""
        return self.stats


class AsyncEventHandler(EventHandler):
    """Abstract base class for asynchronous event handlers."""
    
    @abstractmethod
    async def handle(self, event):
        """Handle an event asynchronously."""
        pass
    
    async def handle_many(self, events):
        """Handle multiple events asynchronously."""
        results = []
        for event in events:
            result = await self.handle(event)
            results.append(result)
        return results


class LoggingHandler(EventHandler):
    """Handler that logs all events."""
    
    def __init__(self, name="logging_handler", log_level=logging.INFO):
        super().__init__(name)
        self.log_level = log_level
    
    def handle(self, event):
        """Log event details."""
        self.stats['processed'] += 1
        
        # Validate event
        is_valid, error = self.validate_event(event)
        if not is_valid:
            logger.warning(f"Invalid event: {error}")
            self.stats['errors'] += 1
            
        # Log event
        log_msg = f"Event: {event.get_type().name}, ID: {event.get_id()}, Time: {event.get_timestamp()}"
        
        # Add more details for specific event types
        if event.get_type() == EventType.BAR:
            symbol = event.data.get('symbol')
            close = event.data.get('close')
            log_msg += f", Symbol: {symbol}, Close: {close}"
        elif event.get_type() == EventType.SIGNAL:
            symbol = event.data.get('symbol')
            signal = event.data.get('signal_value')
            signal_name = "BUY" if signal == 1 else "SELL" if signal == -1 else "NEUTRAL"
            log_msg += f", Symbol: {symbol}, Signal: {signal_name}"
        elif event.get_type() == EventType.ORDER:
            symbol = event.data.get('symbol')
            direction = event.data.get('direction')
            quantity = event.data.get('quantity')
            log_msg += f", Symbol: {symbol}, Direction: {direction}, Quantity: {quantity}"
        elif event.get_type() == EventType.WEBSOCKET:
            connection_id = event.data.get('connection_id')
            state = event.data.get('state')
            log_msg += f", Connection: {connection_id}, State: {state}"
            
        logger.log(self.log_level, log_msg)
        return True

class AsyncLoggingHandler(AsyncEventHandler):
    """Asynchronous handler that logs all events."""
    
    def __init__(self, name="async_logging_handler", log_level=logging.INFO):
        super().__init__(name)
        self.log_level = log_level
    
    async def handle(self, event):
        """Log event details asynchronously."""
        self.stats['processed'] += 1
        
        try:
            # Validate event
            is_valid, error = self.validate_event(event)
            if not is_valid:
                logger.warning(f"Invalid event: {error}")
                self.stats['errors'] += 1
                
            # Log event
            log_msg = f"Async Event: {event.get_type().name}, ID: {event.get_id()}, Time: {event.get_timestamp()}"
            
            # Add more details for specific event types
            if event.get_type() == EventType.BAR:
                symbol = event.data.get('symbol')
                close = event.data.get('close')
                log_msg += f", Symbol: {symbol}, Close: {close}"
            elif event.get_type() == EventType.SIGNAL:
                symbol = event.data.get('symbol')
                signal = event.data.get('signal_value')
                signal_name = "BUY" if signal == 1 else "SELL" if signal == -1 else "NEUTRAL"
                log_msg += f", Symbol: {symbol}, Signal: {signal_name}"
            elif event.get_type() == EventType.ORDER:
                symbol = event.data.get('symbol')
                direction = event.data.get('direction')
                quantity = event.data.get('quantity')
                log_msg += f", Symbol: {symbol}, Direction: {direction}, Quantity: {quantity}"
                
            # Use run_in_executor to prevent blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: logger.log(self.log_level, log_msg))
            
            return True
            
        except Exception as e:
            logger.error(f"Error in async logging handler: {e}", exc_info=True)
            self.stats['errors'] += 1
            return False


class FilterHandler(EventHandler):
    """Handler that filters events based on criteria."""
    
    def __init__(self, name, filter_fn, next_handler):
        super().__init__(name)
        self.filter_fn = filter_fn
        self.next_handler = next_handler
    
    def handle(self, event):
        """Filter events and pass to next handler if they match criteria."""
        self.stats['processed'] += 1
        
        # Validate event
        is_valid, error = self.validate_event(event)
        if not is_valid:
            logger.warning(f"Invalid event: {error}")
            self.stats['errors'] += 1
            return False
            
        # Apply filter
        if self.filter_fn(event):
            # Check if next_handler is a function or an object with handle method
            if hasattr(self.next_handler, 'handle'):
                return self.next_handler.handle(event)
            else:
                # Assume it's a callable function
                return self.next_handler(event)
        return True        

class AsyncFilterHandler(AsyncEventHandler):
    """Asynchronous handler that filters events based on criteria."""
    
    def __init__(self, name, filter_fn, next_handler):
        super().__init__(name)
        self.filter_fn = filter_fn
        self.next_handler = next_handler
        
        # Check if filter function is async
        self.is_async_filter = asyncio.iscoroutinefunction(filter_fn)
        
        # Check if next handler is async
        self.is_async_next = asyncio.iscoroutinefunction(next_handler)
    
    async def handle(self, event):
        """Filter events and pass to next handler if they match criteria."""
        self.stats['processed'] += 1
        
        try:
            # Validate event
            is_valid, error = self.validate_event(event)
            if not is_valid:
                logger.warning(f"Invalid event: {error}")
                self.stats['errors'] += 1
                return False
                
            # Apply filter
            if self.is_async_filter:
                filter_result = await self.filter_fn(event)
            else:
                filter_result = self.filter_fn(event)
                
            if filter_result:
                if self.is_async_next:
                    return await self.next_handler(event)
                else:
                    return self.next_handler(event)
                    
            return True
            
        except Exception as e:
            logger.error(f"Error in async filter handler: {e}", exc_info=True)
            self.stats['errors'] += 1
            return False


class ChainHandler(EventHandler):
    """Handler that chains multiple handlers together."""
    
    def __init__(self, name, handlers=None):
        super().__init__(name)
        self.handlers = handlers or []
    
    def add_handler(self, handler):
        """Add a handler to the chain."""
        self.handlers.append(handler)
    
    def handle(self, event):
        """Pass event through all handlers in chain."""
        self.stats['processed'] += 1
        
        # Validate event
        is_valid, error = self.validate_event(event)
        if not is_valid:
            logger.warning(f"Invalid event: {error}")
            self.stats['errors'] += 1
            return False
            
        # Process through all handlers
        for handler in self.handlers:
            try:
                # Check if handler is a function or an object with handle method
                if hasattr(handler, 'handle'):
                    success = handler.handle(event)
                else:
                    # Assume it's a callable function
                    success = handler(event)
                    
                if not success:
                    return False
            except Exception as e:
                logger.error(f"Error in handler {handler.__class__.__name__ if hasattr(handler, '__class__') else type(handler).__name__}: {e}", exc_info=True)
                self.stats['errors'] += 1
                return False
                
        return True        

class AsyncChainHandler(AsyncEventHandler):
    """Asynchronous handler that chains multiple handlers together."""
    
    def __init__(self, name, handlers=None):
        super().__init__(name)
        self.handlers = handlers or []
        
        # Track which handlers are async
        self.async_handlers = {}
        for i, handler in enumerate(self.handlers):
            self.async_handlers[i] = asyncio.iscoroutinefunction(handler)
    
    def add_handler(self, handler):
        """Add a handler to the chain."""
        self.handlers.append(handler)
        self.async_handlers[len(self.handlers) - 1] = asyncio.iscoroutinefunction(handler)
    
    async def handle(self, event):
        """Pass event through all handlers in chain asynchronously."""
        self.stats['processed'] += 1
        
        try:
            # Validate event
            is_valid, error = self.validate_event(event)
            if not is_valid:
                logger.warning(f"Invalid event: {error}")
                self.stats['errors'] += 1
                return False
                
            # Process through all handlers
            for i, handler in enumerate(self.handlers):
                try:
                    if self.async_handlers[i]:
                        success = await handler(event)
                    else:
                        success = handler(event)
                        
                    if not success:
                        return False
                except Exception as e:
                    logger.error(f"Error in handler {i}: {e}", exc_info=True)
                    self.stats['errors'] += 1
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error in async chain handler: {e}", exc_info=True)
            self.stats['errors'] += 1
            return False    





class BufferedHandler(EventHandler):
    """Handler that buffers events before processing."""
    
    def __init__(self, name, next_handler, buffer_size=10, process_incomplete=True):
        super().__init__(name)
        self.next_handler = next_handler
        self.buffer_size = buffer_size
        self.process_incomplete = process_incomplete
        self.buffer = []
    
    def handle(self, event):
        """Add event to buffer and process if buffer is full."""
        self.stats['processed'] += 1
        
        # Validate event
        is_valid, error = self.validate_event(event)
        if not is_valid:
            logger.warning(f"Invalid event: {error}")
            self.stats['errors'] += 1
            return False
            
        # Add to buffer
        self.buffer.append(event)
        
        # Process if buffer is full
        if len(self.buffer) >= self.buffer_size:
            success = self._process_buffer()
            self.buffer = []
            return success
            
        return True
    
    def flush(self):
        """Process all events in buffer."""
        if not self.buffer:
            return True
            
        if self.process_incomplete or len(self.buffer) >= self.buffer_size:
            success = self._process_buffer()
            self.buffer = []
            return success
            
        return True
    
    def _process_buffer(self):
        """Process all events in buffer."""
        success = True
        for event in self.buffer:
            try:
                result = self.next_handler.handle(event)
                success = success and result
            except Exception as e:
                logger.error(f"Error in handler {self.next_handler.name}: {e}", exc_info=True)
                self.stats['errors'] += 1
                success = False
                
        return success


class AsyncBufferedHandler(AsyncEventHandler):
    """Asynchronous handler that buffers events before processing."""
    
    def __init__(self, name, next_handler, buffer_size=10, process_incomplete=True):
        super().__init__(name)
        self.next_handler = next_handler
        self.buffer_size = buffer_size
        self.process_incomplete = process_incomplete
        self.buffer = []
        self.is_async_next = isinstance(next_handler, AsyncEventHandler)
    
    async def handle(self, event):
        """Add event to buffer and process if buffer is full."""
        self.stats['processed'] += 1
        
        # Validate event
        is_valid, error = self.validate_event(event)
        if not is_valid:
            logger.warning(f"Invalid event: {error}")
            self.stats['errors'] += 1
            return False
            
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
        
        if self.is_async_next:
            # For async handlers, process events concurrently
            tasks = []
            for event in self.buffer:
                try:
                    task = asyncio.create_task(self.next_handler.handle(event))
                    tasks.append(task)
                except Exception as e:
                    logger.error(f"Error creating task: {e}", exc_info=True)
                    self.stats['errors'] += 1
                    success = False
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Error in async handler: {result}")
                        self.stats['errors'] += 1
                        success = False
                    elif not result:
                        success = False
        else:
            # For sync handlers, process events sequentially
            for event in self.buffer:
                try:
                    result = self.next_handler.handle(event)
                    success = success and result
                except Exception as e:
                    logger.error(f"Error in handler {self.next_handler.name}: {e}", exc_info=True)
                    self.stats['errors'] += 1
                    success = False
                
        return success


class WebSocketHandler(AsyncEventHandler):
    """Handler for WebSocket events."""
    
    def __init__(self, name):
        super().__init__(name)
        self.connections = {}  # connection_id -> WebSocket
    
    async def handle(self, event):
        """Handle WebSocket events."""
        self.stats['processed'] += 1
        
        # Validate event
        is_valid, error = self.validate_event(event)
        if not is_valid:
            logger.warning(f"Invalid event: {error}")
            self.stats['errors'] += 1
            return False
            
        # Check if it's a WebSocket event
        if event.get_type() != EventType.WEBSOCKET:
            logger.warning(f"Not a WebSocket event: {event.get_type().name}")
            return False
            
        # Extract connection info
        try:
            connection_id = event.data['connection_id']
            state = event.data['state']
            
            # Handle connection state changes
            if state == 'CONNECTED':
                self.connections[connection_id] = event.data.get('connection')
                logger.info(f"WebSocket connected: {connection_id}")
                
            elif state == 'DISCONNECTED':
                if connection_id in self.connections:
                    del self.connections[connection_id]
                logger.info(f"WebSocket disconnected: {connection_id}")
                
            elif state == 'MESSAGE':
                # Process message based on type
                data = event.data.get('data', {})
                await self._process_message(connection_id, data)
                
            return True
            
        except Exception as e:
            logger.error(f"Error handling WebSocket event: {e}", exc_info=True)
            self.stats['errors'] += 1
            return False
    
    async def _process_message(self, connection_id, data):
        """
        Process a WebSocket message.
        
        Override this method in subclasses to handle specific message types.
        
        Args:
            connection_id: WebSocket connection ID
            data: Message data
        """
        logger.debug(f"WebSocket message from {connection_id}: {data}")
        return True
