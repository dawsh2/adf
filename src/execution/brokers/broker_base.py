from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from src.core.events.event_types import OrderEvent, FillEvent

class BrokerBase(ABC):
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
    
    @abstractmethod
    def place_order(self, order):
        """Place an order with the broker."""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id):
        """Cancel an existing order."""
        pass
    
    def emit_fill(self, fill):
        """Emit a fill event."""
        if not isinstance(fill, FillEvent):
            return
            
        if self.fill_emitter:
            self.fill_emitter.emit(fill)
        elif self.event_bus:
            self.event_bus.emit(fill)
