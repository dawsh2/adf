import logging
import datetime
import json
import asyncio
import time
from typing import Dict, Any, Optional, List, Callable, Union, Coroutine
from abc import ABC, abstractmethod

from .event_types import (
    Event, EventType, BarEvent, WebSocketEvent, ErrorEvent
)
from .event_utils import create_bar_event

logger = logging.getLogger(__name__)

class EventEmitter(ABC):
    """Abstract base class for event emitters."""
    
    def __init__(self, name, event_bus=None):
        self.name = name
        self.event_bus = event_bus
        self.stats = {
            'emitted': 0,
            'errors': 0
        }
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
    
    @abstractmethod
    def start(self):
        """Start emitting events."""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop emitting events."""
        pass
    
    def emit(self, event):
        """Emit an event to the event bus."""
        if not self.event_bus:
            logger.warning(f"No event bus set for emitter {self.name}")
            self.stats['errors'] += 1
            return False
            
        try:
            self.event_bus.emit(event)
            self.stats['emitted'] += 1
            return True
        except Exception as e:
            logger.error(f"Error emitting event: {e}", exc_info=True)
            self.stats['errors'] += 1
            return False
    
    async def emit_async(self, event):
        """Emit an event to the event bus asynchronously."""
        if not self.event_bus:
            logger.warning(f"No event bus set for emitter {self.name}")
            self.stats['errors'] += 1
            return False
            
        try:
            if hasattr(self.event_bus, 'emit_async'):
                await self.event_bus.emit_async(event)
            else:
                self.event_bus.emit(event)
            self.stats['emitted'] += 1
            return True
        except Exception as e:
            logger.error(f"Error emitting event asynchronously: {e}", exc_info=True)
            self.stats['errors'] += 1
            return False
    
    def reset_stats(self):
        """Reset emitter statistics."""
        self.stats = {
            'emitted': 0,
            'errors': 0
        }
    
    def get_stats(self):
        """Get emitter statistics."""
        return self.stats

class BarEmitter(EventEmitter):
    """Concrete implementation of EventEmitter for bar events."""
    
    def __init__(self, name, event_bus=None):
        """
        Initialize the bar emitter.
        
        Args:
            name: Emitter name
            event_bus: Event bus to emit events to
        """
        super().__init__(name, event_bus)
        self.running = False
        self.stats.update({
            'bars_by_symbol': {}
        })
    
    def start(self):
        """Start the bar emitter."""
        self.running = True
        logger.info(f"Bar emitter {self.name} started")
    
    def stop(self):
        """Stop the bar emitter."""
        self.running = False
        logger.info(f"Bar emitter {self.name} stopped")
    
    def emit(self, event):
        """
        Emit a bar event with additional tracking.
        
        Args:
            event: Bar event to emit
            
        Returns:
            True if emitted successfully, False otherwise
        """
        # Skip if not running
        if not self.running:
            return False
            
        # Track statistics for bar events
        if isinstance(event, BarEvent):
            symbol = event.get_symbol()
            
            # Update symbol-specific stats
            if symbol not in self.stats['bars_by_symbol']:
                self.stats['bars_by_symbol'][symbol] = 0
            self.stats['bars_by_symbol'][symbol] += 1
        
        # Call parent implementation
        return super().emit(event)
    


class SignalEmitter(EventEmitter):
    """Emitter for trading signal events."""
    
    def __init__(self, name, event_bus=None):
        super().__init__(name, event_bus)
        self.running = False
    
    def start(self):
        """Start emitting signal events."""
        if self.running:
            logger.warning(f"Signal emitter {self.name} already running")
            return
            
        self.running = True
        logger.info(f"Signal emitter {self.name} started")
    
    def stop(self):
        """Stop emitting signal events."""
        self.running = False
        logger.info(f"Signal emitter {self.name} stopped")
    
    def emit_signal(self, signal):
        """
        Emit a signal to the event bus.
        
        Args:
            signal: Signal to emit
            
        Returns:
            True if signal was emitted, False otherwise
        """
        if not self.running:
            logger.warning(f"Signal emitter {self.name} not running")
            return False
            
        return self.emit(signal)

class HistoricalDataEmitter(EventEmitter):
    """Emitter for historical market data."""
    
    def __init__(self, name, data_source, symbols, start_date=None, end_date=None, 
                 timeframe='1d', event_bus=None):
        super().__init__(name, event_bus)
        self.data_source = data_source
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.running = False
        self.data_frames = {}  # symbol -> DataFrame
        self.current_index = {}  # symbol -> current row index
    
    def start(self):
        """Start emitting historical data events."""
        if self.running:
            logger.warning(f"Emitter {self.name} already running")
            return
            
        self.running = True
        
        # Load data for all symbols
        for symbol in self.symbols:
            try:
                # Get data from source
                df = self.data_source.get_data(
                    symbol, self.start_date, self.end_date, self.timeframe
                )
                
                if df.empty:
                    logger.warning(f"No data for symbol {symbol}")
                    continue
                    
                # Store data and initialize index
                self.data_frames[symbol] = df
                self.current_index[symbol] = 0
                
                logger.info(f"Loaded {len(df)} bars for {symbol}")
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}", exc_info=True)
                self.stats['errors'] += 1
    
    def stop(self):
        """Stop emitting events."""
        self.running = False
    
    def emit_next(self):
        """Emit the next bar for all symbols."""
        if not self.running:
            logger.warning(f"Emitter {self.name} not running")
            return False
            
        events_emitted = 0
        
        for symbol in self.symbols:
            if symbol not in self.data_frames:
                continue
                
            df = self.data_frames[symbol]
            index = self.current_index[symbol]
            
            if index >= len(df):
                continue  # No more data for this symbol
                
            # Get row data
            row = df.iloc[index]
            timestamp = row.name if isinstance(row.name, datetime.datetime) else pd.to_datetime(row.name)
            
            # Create and emit bar event
            event = create_bar_event(
                symbol=symbol,
                timestamp=timestamp,
                open_price=row['open'] if 'open' in row else row['Open'],
                high_price=row['high'] if 'high' in row else row['High'],
                low_price=row['low'] if 'low' in row else row['Low'],
                close_price=row['close'] if 'close' in row else row['Close'],
                volume=row['volume'] if 'volume' in row else row['Volume']
            )
            
            success = self.emit(event)
            if success:
                events_emitted += 1
                
            # Increment index
            self.current_index[symbol] = index + 1
        
        return events_emitted > 0
    
    def has_more_data(self):
        """Check if there is more data to emit."""
        if not self.running:
            return False
            
        for symbol in self.symbols:
            if symbol not in self.data_frames:
                continue
                
            if self.current_index[symbol] < len(self.data_frames[symbol]):
                return True
                
        return False
    
    def reset(self):
        """Reset the emitter state."""
        self.current_index = {symbol: 0 for symbol in self.symbols if symbol in self.data_frames}
        self.reset_stats()


class EventGeneratorEmitter(EventEmitter):
    """Emitter that generates events based on a function."""
    
    def __init__(self, name, generator_fn, event_bus=None, interval=1.0):
        super().__init__(name, event_bus)
        self.generator_fn = generator_fn
        self.interval = interval
        self.running = False
        self.timer = None
    
    def start(self):
        """Start generating events."""
        if self.running:
            logger.warning(f"Emitter {self.name} already running")
            return
            
        self.running = True
        self._schedule_next()
    
    def stop(self):
        """Stop generating events."""
        self.running = False
        if self.timer:
            self.timer.cancel()
            self.timer = None
    
    def _schedule_next(self):
        """Schedule the next event."""
        if not self.running:
            return
            
        try:
            # Generate event
            event = self.generator_fn()
            if event:
                self.emit(event)
                
            # Schedule next event
            import threading
            self.timer = threading.Timer(self.interval, self._schedule_next)
            self.timer.daemon = True
            self.timer.start()
        except Exception as e:
            logger.error(f"Error generating event: {e}", exc_info=True)
            self.stats['errors'] += 1
            # Continue scheduling despite error
            import threading
            self.timer = threading.Timer(self.interval, self._schedule_next)
            self.timer.daemon = True
            self.timer.start()
    
    def reset(self):
        """Reset the emitter state."""
        self.stop()
        self.reset_stats()


class AsyncEventEmitter(EventEmitter):
    """Base class for asynchronous event emitters."""
    
    def __init__(self, name, event_bus=None):
        super().__init__(name, event_bus)
        self.running = False
        self.tasks = []  # Track async tasks
        self.loop = None
    
    async def start(self):
        """Start the asynchronous emitter."""
        if self.running:
            logger.warning(f"Async emitter {self.name} already running")
            return
            
        self.running = True
        self.loop = asyncio.get_running_loop()
    
    async def stop(self):
        """Stop the asynchronous emitter."""
        self.running = False
        
        # Cancel any pending tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
        self.tasks = []
    
    def create_task(self, coro):
        """Create and track an async task."""
        if not self.loop:
            self.loop = asyncio.get_event_loop()
            
        task = self.loop.create_task(coro)
        self.tasks.append(task)
        
        # Clean up task when done
        task.add_done_callback(self._task_done)
        
        return task
    
    def _task_done(self, task):
        """Handle task completion."""
        if task in self.tasks:
            self.tasks.remove(task)
            
        # Check for exceptions
        try:
            if task.exception():
                logger.error(f"Task error in {self.name}: {task.exception()}")
                self.stats['errors'] += 1
        except (asyncio.CancelledError, asyncio.InvalidStateError):
            # Task was cancelled or is still running
            pass


class WebSocketEmitter(AsyncEventEmitter):
    """Emitter for WebSocket data streams."""
    
    def __init__(self, name, url, event_parsers=None, 
                 auth_handler=None, event_bus=None):
        super().__init__(name, event_bus)
        self.url = url
        self.event_parsers = event_parsers or {}
        self.auth_handler = auth_handler
        self.connection = None
        self.reconnect_delay = 1.0  # Initial reconnect delay
        self.max_reconnect_delay = 60.0  # Maximum reconnect delay
        self.stats.update({
            'messages_received': 0,
            'messages_parsed': 0,
            'reconnects': 0
        })
    
    async def start(self):
        """Start the WebSocket connection."""
        await super().start()
        
        if not self.running:
            return
            
        # Start connection task
        self.create_task(self._maintain_connection())
    
    async def _maintain_connection(self):
        """Maintain the WebSocket connection, reconnecting as needed."""
        while self.running:
            try:
                # Emit connecting event
                await self.emit_async(WebSocketEvent(
                    connection_id=self.name,
                    state=WebSocketEvent.CONNECTING
                ))
                
                # Connect to WebSocket
                import websockets
                headers = {}
                
                # Apply authentication if needed
                if self.auth_handler:
                    auth_result = await self.auth_handler()
                    if isinstance(auth_result, dict):
                        headers.update(auth_result)
                
                self.connection = await websockets.connect(self.url, extra_headers=headers)
                
                # Emit connected event
                await self.emit_async(WebSocketEvent(
                    connection_id=self.name,
                    state=WebSocketEvent.CONNECTED
                ))
                
                # Reset reconnect delay after successful connection
                self.reconnect_delay = 1.0
                
                # Start listening for messages
                await self._listen_for_messages()
                
                # If we get here, the connection was closed
                if self.running:
                    logger.info(f"WebSocket connection {self.name} closed, reconnecting...")
                    self.stats['reconnects'] += 1
                    await asyncio.sleep(self.reconnect_delay)
                    
                    # Increase reconnect delay with backoff
                    self.reconnect_delay = min(self.reconnect_delay * 1.5, self.max_reconnect_delay)
            except Exception as e:
                # Emit error event
                await self.emit_async(WebSocketEvent(
                    connection_id=self.name,
                    state=WebSocketEvent.ERROR,
                    data={'error': str(e)}
                ))
                
                logger.error(f"WebSocket error for {self.name}: {e}", exc_info=True)
                self.stats['errors'] += 1
                
                if self.running:
                    # Wait and reconnect
                    await asyncio.sleep(self.reconnect_delay)
                    
                    # Increase reconnect delay with backoff
                    self.reconnect_delay = min(self.reconnect_delay * 1.5, self.max_reconnect_delay)
    
    async def _listen_for_messages(self):
        """Listen for WebSocket messages."""
        if not self.connection:
            return
            
        try:
            async for message in self.connection:
                if not self.running:
                    break
                    
                self.stats['messages_received'] += 1
                
                # Emit raw message event
                await self.emit_async(WebSocketEvent(
                    connection_id=self.name,
                    state=WebSocketEvent.MESSAGE,
                    data={'raw': message}
                ))
                
                # Parse and emit events
                try:
                    events = await self._parse_message(message)
                    for event in events:
                        if event:
                            await self.emit_async(event)
                            self.stats['messages_parsed'] += 1
                except Exception as e:
                    logger.error(f"Error parsing WebSocket message: {e}", exc_info=True)
                    self.stats['errors'] += 1
                    
                    # Emit error event
                    await self.emit_async(ErrorEvent(
                        error_type="PARSE_ERROR",
                        message=f"Error parsing WebSocket message: {e}",
                        source=self.name,
                        exception=e
                    ))
        except Exception as e:
            if self.running:
                logger.error(f"WebSocket listen error for {self.name}: {e}", exc_info=True)
                self.stats['errors'] += 1
                
                # Emit error event
                await self.emit_async(ErrorEvent(
                    error_type="WEBSOCKET_ERROR",
                    message=f"WebSocket error: {e}",
                    source=self.name,
                    exception=e
                ))
        finally:
            # Emit disconnected event
            await self.emit_async(WebSocketEvent(
                connection_id=self.name,
                state=WebSocketEvent.DISCONNECTED
            ))
    
    async def _parse_message(self, message):
        """
        Parse a WebSocket message into events.
        
        Args:
            message: Raw WebSocket message
            
        Returns:
            list: List of parsed events
        """
        events = []
        
        try:
            # Try to parse as JSON
            data = json.loads(message)
            
            # Determine message type and apply appropriate parser
            message_type = self._determine_message_type(data)
            
            if message_type in self.event_parsers:
                parser = self.event_parsers[message_type]
                parsed_events = parser(data)
                
                # Handle both single events and lists
                if isinstance(parsed_events, list):
                    events.extend(parsed_events)
                elif parsed_events:
                    events.append(parsed_events)
            else:
                # Default to generic WebSocket event if no parser
                events.append(WebSocketEvent(
                    connection_id=self.name,
                    state=WebSocketEvent.MESSAGE,
                    data=data
                ))
                
        except json.JSONDecodeError:
            # Not JSON, use raw message
            events.append(WebSocketEvent(
                connection_id=self.name,
                state=WebSocketEvent.MESSAGE,
                data={'raw': message, 'format': 'text'}
            ))
            
        return events
    
    def _determine_message_type(self, data):
        """
        Determine the type of WebSocket message.
        
        This can be overridden by subclasses to provide custom type detection.
        
        Args:
            data: Parsed JSON data
            
        Returns:
            str: Message type identifier
        """
        # Default implementation looks for 'type' field
        if isinstance(data, dict) and 'type' in data:
            return data['type']
        
        return 'unknown'
    
    async def send(self, message):
        """
        Send a message through the WebSocket connection.
        
        Args:
            message: Message to send (string or dict that will be JSON encoded)
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.connection:
            logger.warning(f"Cannot send message, WebSocket {self.name} not connected")
            return False
            
        try:
            # Convert dict to JSON if needed
            if isinstance(message, dict):
                message = json.dumps(message)
                
            await self.connection.send(message)
            return True
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}", exc_info=True)
            self.stats['errors'] += 1
            return False
    
    async def stop(self):
        """Stop the WebSocket connection."""
        self.running = False
        
        # Close connection
        if self.connection:
            try:
                await self.connection.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket connection: {e}", exc_info=True)
                
        self.connection = None
        
        # Stop all tasks
        await super().stop()


