import datetime
from typing import Optional, Union, Dict, Any

class Position:
    """
    Represents a position in a single instrument.
    Tracks quantity, cost basis, and P&L.
    """
    
    def __init__(self, symbol):
        self.symbol = symbol
        self.quantity = 0
        self.cost_basis = 0.0
        self.realized_pnl = 0.0
        # Add additional state as needed
        self.last_update_time = None
    
    def add_quantity(self, quantity, price):
        """Add to the position."""
        if quantity <= 0:
            return
        
        # Calculate new cost basis
        old_value = self.quantity * self.cost_basis
        new_value = quantity * price
        new_quantity = self.quantity + quantity
        
        if new_quantity > 0:
            self.cost_basis = (old_value + new_value) / new_quantity
        
        self.quantity = new_quantity
        self.last_update_time = datetime.datetime.now()
    
    def reduce_quantity(self, quantity, price):
        """Reduce the position and calculate realized P&L."""
        if quantity <= 0 or self.quantity <= 0:
            return
        
        # Cannot reduce by more than current quantity
        actual_reduction = min(quantity, self.quantity)
        
        # Calculate realized profit/loss
        transaction_pnl = actual_reduction * (price - self.cost_basis)
        self.realized_pnl += transaction_pnl
        
        # Reduce quantity
        self.quantity -= actual_reduction
        
        # If position is now zero, reset cost basis
        if self.quantity == 0:
            self.cost_basis = 0.0
            
        self.last_update_time = datetime.datetime.now()
    
    def market_value(self, market_price=None):
        """Calculate current market value of position."""
        if market_price is None or self.quantity == 0:
            return 0.0
        
        return self.quantity * market_price
    
    def unrealized_pnl(self, market_price=None):
        """Calculate unrealized profit/loss."""
        if market_price is None or self.quantity == 0:
            return 0.0
        
        return self.quantity * (market_price - self.cost_basis)
    
    def total_pnl(self, market_price=None):
        """Calculate total profit/loss (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl(market_price)
    
    def is_long(self):
        """Check if position is long."""
        return self.quantity > 0
    
    def is_short(self):
        """Check if position is short."""
        return self.quantity < 0
    
    def is_flat(self):
        """Check if position is flat (no holdings)."""
        return self.quantity == 0
