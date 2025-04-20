from typing import Dict, List, Any, Optional
import datetime

from src.core.events.event_types import FillEvent, EventType
from src.execution.position import Position


class PortfolioManager:
    """
    Tracks the current state of the portfolio.
    Listens for fill events to update positions and cash.
    """
    
    def __init__(self, initial_cash=0.0, event_bus=None):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}  # symbol -> Position
        self.fill_history = []
        self.event_bus = event_bus
        self.last_update_time = datetime.datetime.now()
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
        return self
    
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
        
        # Update timestamp
        self.last_update_time = event.get_timestamp() or datetime.datetime.now()
    
    def _update_position(self, symbol, direction, quantity, price):
        """Update or create position based on fill."""
        # Create position if it doesn't exist
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        
        # Update position quantity and cost basis
        position = self.positions[symbol]
        if direction == 'BUY':
            position.add_quantity(quantity, price)
        elif direction == 'SELL':
            position.reduce_quantity(quantity, price)
    
    def _update_cash(self, direction, quantity, price, commission):
        """Update cash balance based on fill."""
        # Calculate trade value
        trade_value = quantity * price
        
        # Update cash: reduce for buys, increase for sells
        if direction == 'BUY':
            self.cash -= trade_value + commission
        elif direction == 'SELL':
            self.cash += trade_value - commission
    
    def get_equity(self, market_prices=None):
        """Calculate total portfolio equity."""
        # Cash plus position values
        position_value = sum(position.market_value(
            market_prices.get(position.symbol) if market_prices else None
        ) for position in self.positions.values())
        
        return self.cash + position_value
    
    def get_position(self, symbol):
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def get_all_positions(self):
        """Get all positions."""
        return self.positions
    
    def get_position_value(self, market_prices=None):
        """Get total value of all positions."""
        return sum(position.market_value(
            market_prices.get(position.symbol) if market_prices else None
        ) for position in self.positions.values())
    
    def get_portfolio_summary(self, market_prices=None):
        """Get summary of portfolio state."""
        return {
            'cash': self.cash,
            'position_value': self.get_position_value(market_prices),
            'equity': self.get_equity(market_prices),
            'positions': len(self.positions),
            'fills': len(self.fill_history),
            'last_update': self.last_update_time
        }
    
    def reset(self):
        """Reset portfolio to initial state."""
        self.cash = self.initial_cash
        self.positions = {}
        self.fill_history = []
        self.last_update_time = datetime.datetime.now()
