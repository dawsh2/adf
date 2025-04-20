class ExecutionEngine:
    """
    Execution engine that processes signals and routes them through the system.
    Acts as middleware between strategies and broker execution.
    """
    
    def __init__(self, risk_manager, broker_interface, order_emitter=None):
        self.risk_manager = risk_manager
        self.broker_interface = broker_interface
        self.order_emitter = order_emitter
        self.event_bus = None
    
    def set_event_bus(self, event_bus):
        self.event_bus = event_bus
    
    def on_signal(self, event):
        """Process signal events and create orders."""
        # Extract signal details
        symbol = event.get_symbol()
        signal_value = event.get_signal_value()
        price = event.get_price()
        
        # Determine order direction
        if signal_value == SignalEvent.BUY:
            direction = OrderEvent.BUY
        elif signal_value == SignalEvent.SELL:
            direction = OrderEvent.SELL
        else:
            return  # Ignore neutral signals
        
        # Get position sizing from risk manager (externalized logic)
        order_details = self.risk_manager.calculate_order_details(
            symbol, direction, price, event
        )
        
        if not order_details or order_details.get('quantity', 0) <= 0:
            return  # Risk manager rejected trade or sized to zero
        
        # Create order event
        order = OrderEvent(
            symbol=symbol,
            order_type=order_details.get('order_type', OrderEvent.MARKET),
            direction=direction,
            quantity=order_details.get('quantity'),
            price=order_details.get('price', price),
            timestamp=event.get_timestamp()
        )
        
        # Emit order event
        if self.order_emitter:
            self.order_emitter.emit(order)
        elif self.event_bus:
            self.event_bus.emit(order)
            
        # Send to broker interface
        self.broker_interface.place_order(order)
