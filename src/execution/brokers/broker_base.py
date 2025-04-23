"""
Fixed implementation of BrokerBase class with improved emit_fill functionality.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from src.core.events.event_types import OrderEvent, FillEvent

logger = logging.getLogger(__name__)

class BrokerBase:
    """
    Abstract base class for all broker interfaces.
    Defines common operations for order execution.
    """
    
    def __init__(self, fill_emitter=None):
        """
        Initialize the broker base.
        
        Args:
            fill_emitter: Optional component to emit fill events
        """
        self.fill_emitter = fill_emitter
        self.event_bus = None
        
        if fill_emitter:
            logger.debug(f"BrokerBase initialized with fill_emitter: {fill_emitter}")
    
    def set_event_bus(self, event_bus):
        """
        Set the event bus.
        
        Args:
            event_bus: Event bus for emitting events
            
        Returns:
            self: For method chaining
        """
        self.event_bus = event_bus
        logger.debug(f"Event bus set for {self.__class__.__name__}")
        return self
    
    @abstractmethod
    def place_order(self, order):
        """
        Place an order with the broker.
        
        Args:
            order: Order to place
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id):
        """
        Cancel an existing order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            bool: Success of cancellation
        """
        pass
    
    def emit_fill(self, fill):
        """
        Emit a fill event.
        
        Args:
            fill: Fill event to emit
            
        Returns:
            bool: True if successfully emitted, False otherwise
        """
        if not isinstance(fill, FillEvent):
            logger.error(f"Invalid fill event type: {type(fill)}")
            return False
        
        # Flag to track success
        success = False
            
        # Try using fill_emitter first
        if self.fill_emitter:
            try:
                # Handle case where fill_emitter is an event bus or similar with emit method
                if hasattr(self.fill_emitter, 'emit'):
                    self.fill_emitter.emit(fill)
                    logger.debug(f"Fill event emitted via fill_emitter.emit")
                    success = True
                # Handle case where fill_emitter might be a callable/function
                elif callable(self.fill_emitter):
                    self.fill_emitter(fill)
                    logger.debug(f"Fill event emitted via callable fill_emitter")
                    success = True
            except Exception as e:
                logger.error(f"Error emitting fill event via fill_emitter: {e}")
                # Continue to try event_bus as fallback
        
        # Fall back to event_bus if fill_emitter failed or is not set
        if not success and self.event_bus:
            try:
                self.event_bus.emit(fill)
                logger.debug(f"Fill event emitted via event_bus")
                success = True
            except Exception as e:
                logger.error(f"Error emitting fill event via event_bus: {e}")
        
        if not success:
            logger.error(f"Failed to emit fill event: No valid emitter available")
            
        return success

