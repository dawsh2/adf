import logging
import weakref
import inspect
import asyncio
from typing import Dict, List, Callable, Any, Set, Union, Optional, Coroutine

from .event_types import Event, EventType

logger = logging.getLogger(__name__)

class EventBus:
    """Central event bus for routing events between system components."""
    
    def __init__(self, use_weak_refs=True):
        self.handlers = {}  # EventType -> list of handlers
        self.async_handlers = {}  # EventType -> list of async handlers
        self.event_counts = {}  # For tracking event statistics
        self.use_weak_refs = use_weak_refs
        self.strong_refs = set()  # Keep strong references when needed
    
    def register(self, event_type, handler, weak_ref=None):
        """
        Register a handler for an event type.
        
        Args:
            event_type: EventType to register for
            handler: Callable to handle event
            weak_ref: Override default weak_ref behavior (True/False)
        """
        if not callable(handler):
            logger.error(f"Cannot register non-callable handler: {handler}")
            return
        
        # Check if this is an async handler
        if inspect.iscoroutinefunction(handler) or (
            hasattr(handler, '__call__') and 
            inspect.iscoroutinefunction(handler.__call__)
        ):
            return self.register_async(event_type, handler, weak_ref)
            
        if event_type not in self.handlers:
            self.handlers[event_type] = []
            self.event_counts[event_type] = 0
        
        # Determine if we should use weak reference
        use_weak = self.use_weak_refs if weak_ref is None else weak_ref
        
        if use_weak:
            # Check if it's a bound method (has __self__ attribute)
            if hasattr(handler, '__self__') and not isinstance(handler.__self__, type):
                # Create a weakref that automatically unregisters when garbage collected
                weak_handler = weakref.WeakMethod(handler, self._create_cleanup(event_type, handler))
                self.handlers[event_type].append(weak_handler)
            else:
                # Regular function or class method
                weak_handler = weakref.ref(handler, self._create_cleanup(event_type, handler))
                self.handlers[event_type].append(weak_handler)
        else:
            # Store strong reference
            self.handlers[event_type].append(handler)
            # Keep a strong reference to prevent garbage collection
            self.strong_refs.add(handler)
            
        logger.debug(f"Registered handler for {event_type.name}")
    
    def register_async(self, event_type, handler, weak_ref=None):
        """
        Register an async handler for an event type.
        
        Args:
            event_type: EventType to register for
            handler: Async callable to handle event
            weak_ref: Override default weak_ref behavior (True/False)
        """
        if not inspect.iscoroutinefunction(handler) and not (
            hasattr(handler, '__call__') and 
            inspect.iscoroutinefunction(handler.__call__)
        ):
            logger.error(f"Cannot register non-async handler as async: {handler}")
            return
            
        if event_type not in self.async_handlers:
            self.async_handlers[event_type] = []
            if event_type not in self.event_counts:
                self.event_counts[event_type] = 0
        
        # Determine if we should use weak reference
        use_weak = self.use_weak_refs if weak_ref is None else weak_ref
        
        if use_weak:
            # Check if it's a bound method (has __self__ attribute)
            if hasattr(handler, '__self__') and not isinstance(handler.__self__, type):
                # Create a weakref that automatically unregisters when garbage collected
                weak_handler = weakref.WeakMethod(
                    handler, 
                    self._create_cleanup(event_type, handler, is_async=True)
                )
                self.async_handlers[event_type].append(weak_handler)
            else:
                # Regular function or class method
                weak_handler = weakref.ref(
                    handler, 
                    self._create_cleanup(event_type, handler, is_async=True)
                )
                self.async_handlers[event_type].append(weak_handler)
        else:
            # Store strong reference
            self.async_handlers[event_type].append(handler)
            # Keep a strong reference to prevent garbage collection
            self.strong_refs.add(handler)
            
        logger.debug(f"Registered async handler for {event_type.name}")
    
    def _create_cleanup(self, event_type, handler, is_async=False):
        """Create a cleanup function for when weakref is garbage collected."""
        def cleanup(weak_ref):
            handlers_dict = self.async_handlers if is_async else self.handlers
            logger.debug(f"Handler for {event_type.name} garbage collected, cleaning up")
            try:
                handlers = handlers_dict.get(event_type, [])
                # Find and remove any dead weakrefs
                handlers_dict[event_type] = [h for h in handlers if not 
                    (isinstance(h, weakref.ref) and h() is None)]
            except Exception as e:
                logger.error(f"Error in weakref cleanup: {e}")
                
        return cleanup
    
    def unregister(self, event_type, handler, is_async=False):
        """Unregister a handler for an event type."""
        handlers_dict = self.async_handlers if is_async else self.handlers
        
        if event_type not in handlers_dict:
            return
            
        # Check if we're using weak references
        if self.use_weak_refs:
            # Remove matching weakrefs
            new_handlers = []
            for existing_handler in handlers_dict[event_type]:
                if isinstance(existing_handler, weakref.ref):
                    # Get the actual handler from weakref
                    actual_handler = existing_handler()
                    if actual_handler is not None and actual_handler != handler:
                        new_handlers.append(existing_handler)
                elif existing_handler != handler:
                    new_handlers.append(existing_handler)
            handlers_dict[event_type] = new_handlers
        else:
            # Direct removal for strong references
            if handler in handlers_dict[event_type]:
                handlers_dict[event_type].remove(handler)
                # Also remove from strong_refs set
                if handler in self.strong_refs:
                    self.strong_refs.remove(handler)
                    
        logger.debug(f"Unregistered {'async ' if is_async else ''}handler for {event_type.name}")
    
    def unregister_async(self, event_type, handler):
        """Unregister an async handler for an event type."""
        return self.unregister(event_type, handler, is_async=True)
    
    def emit(self, event):
        """
        Emit an event to registered handlers.
        
        Args:
            event: Event to emit
            
        Returns:
            int: Number of handlers that processed the event
        """
        event_type = event.get_type()
        handlers_called = 0
        
        # Track event count
        if event_type in self.event_counts:
            self.event_counts[event_type] += 1
        else:
            self.event_counts[event_type] = 1
            
        logger.debug(f"Emitting {event_type.name} event (ID: {event.get_id()})")
        
        # Process handlers
        if event_type in self.handlers:
            # Make a copy to avoid modification during iteration
            handlers_copy = list(self.handlers[event_type])
            dead_refs = []
            
            for handler_ref in handlers_copy:
                try:
                    # Get actual handler from weakref if needed
                    if isinstance(handler_ref, weakref.ref):
                        handler = handler_ref()
                        if handler is None:
                            # Reference is dead, mark for cleanup
                            dead_refs.append(handler_ref)
                            continue
                    else:
                        handler = handler_ref
                        
                    # Call the handler
                    handler(event)
                    handlers_called += 1
                except Exception as e:
                    logger.error(f"Error in handler: {e}", exc_info=True)
            
            # Clean up any dead references
            if dead_refs and event_type in self.handlers:
                self.handlers[event_type] = [
                    h for h in self.handlers[event_type] 
                    if h not in dead_refs
                ]
        
        return handlers_called
    
    async def emit_async(self, event):
        """
        Emit an event to registered handlers, both sync and async.
        
        This will call sync handlers immediately and wait for all async
        handlers to complete.
        
        Args:
            event: Event to emit
            
        Returns:
            tuple: (sync_handlers_called, async_handlers_called)
        """
        event_type = event.get_type()
        sync_handlers_called = 0
        async_handlers_called = 0
        
        # First, call synchronous handlers
        sync_handlers_called = self.emit(event)
        
        # Process async handlers
        if event_type in self.async_handlers:
            # Make a copy to avoid modification during iteration
            handlers_copy = list(self.async_handlers[event_type])
            dead_refs = []
            tasks = []
            
            for handler_ref in handlers_copy:
                try:
                    # Get actual handler from weakref if needed
                    if isinstance(handler_ref, weakref.ref):
                        handler = handler_ref()
                        if handler is None:
                            # Reference is dead, mark for cleanup
                            dead_refs.append(handler_ref)
                            continue
                    else:
                        handler = handler_ref
                        
                    # Schedule the handler
                    task = asyncio.create_task(handler(event))
                    tasks.append(task)
                except Exception as e:
                    logger.error(f"Error scheduling async handler: {e}", exc_info=True)
            
            # Wait for all tasks to complete
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Error in async handler: {result}")
                    else:
                        async_handlers_called += 1
            
            # Clean up any dead references
            if dead_refs and event_type in self.async_handlers:
                self.async_handlers[event_type] = [
                    h for h in self.async_handlers[event_type] 
                    if h not in dead_refs
                ]
        
        return (sync_handlers_called, async_handlers_called)
    
    def emit_for(self, event_type, data=None, timestamp=None):
        """
        Create and emit an event of the specified type.
        
        Args:
            event_type: Type of event to emit
            data: Event data
            timestamp: Event timestamp
            
        Returns:
            Event: The created and emitted event
        """
        event = Event(event_type, data, timestamp)
        self.emit(event)
        return event
    
    async def emit_for_async(self, event_type, data=None, timestamp=None):
        """
        Create and emit an event of the specified type asynchronously.
        
        Args:
            event_type: Type of event to emit
            data: Event data
            timestamp: Event timestamp
            
        Returns:
            Event: The created and emitted event
        """
        event = Event(event_type, data, timestamp)
        await self.emit_async(event)
        return event
    
    def reset(self):
        """Reset event counts."""
        self.event_counts = {
            event_type: 0 
            for event_type in set(
                list(self.handlers.keys()) + list(self.async_handlers.keys())
            )
        }

    def get_stats(self):
        """Get event statistics."""
        # Count active handlers
        active_handlers = {}
        for event_type, handlers in self.handlers.items():
            # Count non-dead weak references
            active_count = 0
            for handler in handlers:
                if isinstance(handler, weakref.ref):
                    if handler() is not None:
                        active_count += 1
                else:
                    active_count += 1
            active_handlers[event_type.name] = active_count
        
        # Count active async handlers
        active_async_handlers = {}
        for event_type, handlers in self.async_handlers.items():
            # Count non-dead weak references
            active_count = 0
            for handler in handlers:
                if isinstance(handler, weakref.ref):
                    if handler() is not None:
                        active_count += 1
                else:
                    active_count += 1
            active_async_handlers[event_type.name] = active_count
            
        return {
            'total_events': sum(self.event_counts.values()),
            'events_by_type': {
                event_type.name: count 
                for event_type, count in self.event_counts.items()
            },
            'active_handlers': active_handlers,
            'active_async_handlers': active_async_handlers,
            'strong_refs_count': len(self.strong_refs)
        }
        
    def cleanup(self):
        """Clean up dead weak references."""
        # Clean up sync handlers
        for event_type in list(self.handlers.keys()):
            if event_type in self.handlers:
                self.handlers[event_type] = [
                    h for h in self.handlers[event_type]
                    if not (isinstance(h, weakref.ref) and h() is None)
                ]
        
        # Clean up async handlers
        for event_type in list(self.async_handlers.keys()):
            if event_type in self.async_handlers:
                self.async_handlers[event_type] = [
                    h for h in self.async_handlers[event_type]
                    if not (isinstance(h, weakref.ref) and h() is None)
                ]
