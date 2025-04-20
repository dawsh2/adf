from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from src.core.events.event_types import OrderEvent, FillEvent, EventType

class ExecutionEngineBase(ABC):
    """Abstract base class for execution engines."""
    
    @abstractmethod
    def on_order(self, event):
        """Process order events."""
        pass
    
    @abstractmethod
    def place_order(self, order):
        """Place an order with the appropriate broker."""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id):
        """Cancel an existing order."""
        pass


class ExecutionEngine(ExecutionEngineBase):
    """
    Routes order instructions to the appropriate broker interface.
    Acts as middleware between risk management and broker execution.
    """
    
    def __init__(self, broker_interface, event_bus=None):
        self.broker_interface = broker_interface
        self.event_bus = event_bus
        self.orders = {}  # order_id -> order
        self.order_status = {}  # order_id -> status
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
        return self
    
    def on_order(self, event):
        """Process order events."""
        if not isinstance(event, OrderEvent):
            return
        
        # Track the order
        order_id = event.get_id()
        self.orders[order_id] = event
        self.order_status[order_id] = 'PENDING'
        
        # Place the order with the broker
        self.place_order(event)
    
    def place_order(self, order):
        """Place an order with the broker."""
        try:
            self.broker_interface.place_order(order)
            self.order_status[order.get_id()] = 'PLACED'
        except Exception as e:
            self.order_status[order.get_id()] = 'FAILED'
            # Log error
    
    def cancel_order(self, order_id):
        """Cancel an existing order."""
        if order_id not in self.orders:
            return False
        
        try:
            self.broker_interface.cancel_order(order_id)
            self.order_status[order_id] = 'CANCELLING'
            return True
        except Exception as e:
            # Log error
            return False
    
    def on_order_status(self, event):
        """Update order status."""
        order_id = event.get_order_id()
        status = event.get_status()
        
        if order_id in self.order_status:
            self.order_status[order_id] = status
    
    def get_order_status(self, order_id):
        """Get the status of an order."""
        return self.order_status.get(order_id, 'UNKNOWN')
    
    def get_open_orders(self, symbol=None):
        """Get all open orders, optionally filtered by symbol."""
        open_orders = {}
        
        for order_id, status in self.order_status.items():
            if status in ['PENDING', 'PLACED', 'PARTIAL_FILL']:
                order = self.orders[order_id]
                if symbol is None or order.get_symbol() == symbol:
                    open_orders[order_id] = order
        
        return open_orders
