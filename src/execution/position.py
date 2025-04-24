from typing import Dict, List, Any, Optional
import datetime

from src.core.events.event_types import FillEvent, EventType
import logging

"""
Fixed Position class with proper short selling support.
"""


# Add proper logger initialization
logger = logging.getLogger(__name__)

"""
Position tracking module for the trading system.
Contains the Position class for accurate position tracking.
"""


class Position:
    """
    Track an individual position with accurate cost basis and P&L tracking.
    """
    
    def __init__(self, symbol: str, quantity: int = 0, cost_basis: float = 0.0):
        """
        Initialize the position.
        
        Args:
            symbol: Position symbol
            quantity: Position quantity
            cost_basis: Position cost basis
        """
        self.symbol = symbol
        self.quantity = quantity
        self.cost_basis = cost_basis
        self.realized_pnl = 0.0
        
        # Track accumulated costs for accurate cost basis
        self._total_cost = 0.0
        
        # Track transaction history for debugging
        self.transactions = []
    
    def add_quantity(self, quantity: int, price: float):
        """
        Add to position quantity and update cost basis.
        
        Args:
            quantity: Quantity to add (positive for buys, negative for shorts)
            price: Price per unit
        """
        # Track transaction
        self.transactions.append({
            'type': 'BUY' if quantity > 0 else 'SHORT',
            'quantity': abs(quantity),
            'price': price
        })
        
        # For a new position, simply set values
        if self.quantity == 0:
            self.quantity = quantity
            self.cost_basis = price
            self._total_cost = abs(quantity) * price
            return
            
        # For adding to existing same-direction position
        if (self.quantity > 0 and quantity > 0) or (self.quantity < 0 and quantity < 0):
            # Same direction - adding to existing position
            new_total_quantity = self.quantity + quantity
            
            # Calculate new total cost
            existing_cost = abs(self.quantity) * self.cost_basis
            additional_cost = abs(quantity) * price
            self._total_cost = existing_cost + additional_cost
            
            # Update position
            self.quantity = new_total_quantity
            # Calculate new weighted average cost basis
            self.cost_basis = self._total_cost / abs(self.quantity) if self.quantity != 0 else 0.0
        else:
            # Opposite direction - reducing or flipping position
            remaining_quantity = self.quantity + quantity
            
            if (remaining_quantity * self.quantity) > 0:
                # Position reduced but same direction maintained
                # Calculate P&L for the portion that was reduced
                reduced_amount = min(abs(self.quantity), abs(quantity))
                
                # For long positions being reduced
                if self.quantity > 0:
                    pnl = reduced_amount * (price - self.cost_basis)
                # For short positions being reduced
                else:
                    pnl = reduced_amount * (self.cost_basis - price)
                    
                self.realized_pnl += pnl
                
                # Update position
                self.quantity = remaining_quantity
                # Cost basis remains the same
            else:
                # Position flipped direction (long->short or short->long)
                # First calculate P&L for closing the entire existing position
                if self.quantity > 0:
                    pnl = self.quantity * (price - self.cost_basis)
                else:
                    pnl = abs(self.quantity) * (self.cost_basis - price)
                    
                self.realized_pnl += pnl
                
                # Then create a new position in the opposite direction with the remainder
                if remaining_quantity != 0:
                    self.quantity = remaining_quantity
                    self.cost_basis = price
                    self._total_cost = abs(remaining_quantity) * price
                else:
                    # Position closed completely
                    self.quantity = 0
                    self.cost_basis = 0.0
                    self._total_cost = 0.0
        
        logger.debug(f"{self.symbol}: Added {quantity} shares at {price:.2f}, new position: {self.quantity} @ {self.cost_basis:.2f}")
    
    def reduce_quantity(self, quantity: int, price: float):
        """
        Reduce position quantity and calculate realized P&L.
        
        Args:
            quantity: Quantity to reduce (always positive)
            price: Price per unit
        """
        if quantity <= 0 or self.quantity == 0:
            return
            
        # For long positions
        if self.quantity > 0:
            # Cap reduction at current quantity
            actual_quantity = min(quantity, self.quantity)
            
            # Track transaction
            self.transactions.append({
                'type': 'SELL',
                'quantity': actual_quantity,
                'price': price
            })
            
            # Calculate realized P&L from this sale
            cost_of_sold = actual_quantity * self.cost_basis
            proceeds = actual_quantity * price
            transaction_pnl = proceeds - cost_of_sold
            
            # Update realized P&L
            self.realized_pnl += transaction_pnl
            
            # Update quantities
            self.quantity -= actual_quantity
            
            # If quantity is zero, reset cost basis
            if self.quantity <= 0:
                self._total_cost = 0.0
                self.cost_basis = 0.0
                self.quantity = 0
            else:
                # Cost basis remains the same for remaining shares
                self._total_cost = self.quantity * self.cost_basis
        
        # For short positions
        elif self.quantity < 0:
            # Cap reduction at current quantity
            actual_quantity = min(quantity, abs(self.quantity))
            
            # Track transaction
            self.transactions.append({
                'type': 'COVER',
                'quantity': actual_quantity,
                'price': price
            })
            
            # Calculate realized P&L from covering this short
            # For shorts: profit = sell high, buy low
            cost_of_covered = actual_quantity * self.cost_basis  # Original short price
            cost_to_cover = actual_quantity * price              # Price to cover
            transaction_pnl = cost_of_covered - cost_to_cover    # Profit if positive
            
            # Update realized P&L
            self.realized_pnl += transaction_pnl
            
            # Update quantities
            self.quantity += actual_quantity  # Reduce short position (add to negative quantity)
            
            # If quantity is zero, reset cost basis
            if self.quantity == 0:
                self._total_cost = 0.0
                self.cost_basis = 0.0
            else:
                # Cost basis remains the same for remaining short position
                self._total_cost = abs(self.quantity) * self.cost_basis
            
        logger.debug(f"{self.symbol}: Reduced by {quantity} shares at {price:.2f}, " +
                   f"realized P&L: {transaction_pnl:.2f}, remaining: {self.quantity}")
    
    def market_value(self, current_price=None):
        """
        Calculate current market value of the position.
        
        Args:
            current_price: Optional current market price
            
        Returns:
            float: Current market value
        """
        # If no price provided and we have a cost basis, use that
        if current_price is None:
            current_price = self.cost_basis
            
        return self.quantity * current_price
    
    def unrealized_pnl(self, current_price):
        """
        Calculate unrealized P&L at a given price.
        
        Args:
            current_price: Current market price
            
        Returns:
            float: Unrealized P&L
        """
        if self.quantity == 0 or current_price is None:
            return 0.0
            
        # Calculate unrealized P&L
        if self.quantity > 0:
            # Long position: profit if price > cost basis
            return self.quantity * (current_price - self.cost_basis)
        else:
            # Short position: profit if price < cost basis
            return abs(self.quantity) * (self.cost_basis - current_price)
    
    def total_pnl(self, current_price):
        """
        Calculate total P&L (realized + unrealized).
        
        Args:
            current_price: Current market price
            
        Returns:
            float: Total P&L
        """
        return self.realized_pnl + self.unrealized_pnl(current_price)
