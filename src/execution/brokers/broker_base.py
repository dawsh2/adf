from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from src.core.events.event_types import OrderEvent, FillEvent



class BrokerBase:
    """
    Abstract base class for all broker interfaces.
    Defines common operations for order execution.
    """
    
    def __init__(self, fill_emitter=None):
        self.fill_emitter = fill_emitter
        self.event_bus = None
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
        return self
    
    def place_order(self, order):
        """Place an order with the broker."""
        pass
    
    def cancel_order(self, order_id):
        """Cancel an existing order."""
        pass
    
    def emit_fill(self, fill):
        """Emit a fill event."""
        if not isinstance(fill, FillEvent):
            return
            
        # Try using fill_emitter first, then fall back to event_bus
        if self.fill_emitter:
            # This could be an event bus or a dedicated emitter
            if hasattr(self.fill_emitter, 'emit'):
                self.fill_emitter.emit(fill)
            # Handle case where fill_emitter might be an event bus directly
        elif self.event_bus:
            self.event_bus.emit(fill)
