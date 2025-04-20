class Portfolio:
    """
    Tracks the current state of the portfolio.
    Listens for fill events to update positions.
    """
    
    def __init__(self, initial_cash=0.0):
        self.cash = initial_cash
        self.positions = {}  # symbol -> Position
        self.order_history = []
        self.fill_history = []
        self.event_bus = None
    
    def set_event_bus(self, event_bus):
        self.event_bus = event_bus
    
    def on_fill(self, event):
        """Update portfolio state based on fill events."""
        if not isinstance(event, FillEvent):
            return
            
        symbol = event.get_symbol()
        direction = event.get_direction()
        quantity = event.get_quantity()
        price = event.get_price()
        commission = event.get_commission()
        
        # Update position
        self._update_position(symbol, direction, quantity, price)
        
        # Update cash
        self._update_cash(direction, quantity, price, commission)
        
        # Record fill
        self.fill_history.append(event)
    
    def _update_position(self, symbol, direction, quantity, price):
        """Update or create position based on fill."""
        # Create position if it doesn't exist
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        
        # Update position quantity and cost basis
        position = self.positions[symbol]
        if direction == OrderEvent.BUY:
            position.add_quantity(quantity, price)
        elif direction == OrderEvent.SELL:
            position.reduce_quantity(quantity, price)
    
    def _update_cash(self, direction, quantity, price, commission):
        """Update cash balance based on fill."""
        # Reduce cash for buys, increase for sells
        trade_value = quantity * price
        if direction == OrderEvent.BUY:
            self.cash -= trade_value + commission
        elif direction == OrderEvent.SELL:
            self.cash += trade_value - commission
    
    def get_equity(self, market_prices=None):
        """Calculate total portfolio equity."""
        # Cash plus total position values
        position_value = sum(position.market_value(market_prices)
                            for position in self.positions.values())
        return self.cash + position_value
