class BrokerBase(ABC):
    """
    Abstract base class for all broker interfaces.
    """
    
    def __init__(self, fill_emitter=None):
        self.fill_emitter = fill_emitter
        self.event_bus = None
    
    def set_event_bus(self, event_bus):
        self.event_bus = event_bus
    
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
        if self.fill_emitter:
            self.fill_emitter.emit(fill)
        elif self.event_bus:
            self.event_bus.emit(fill)
