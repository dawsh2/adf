"""
Complete fixed portfolio manager with proper position tracking and equity calculation.
"""
import datetime
import logging
from typing import Dict, List, Any, Optional

from src.core.events.event_types import FillEvent

logger = logging.getLogger(__name__)

class Position:
    """
    Position class for tracking security positions and P&L.
    """
    
    def __init__(self, symbol):
        self.symbol = symbol
        self.quantity = 0
        self.cost_basis = 0.0
        self.realized_pnl = 0.0
        self.trades = []
    
    def add_quantity(self, quantity, price):
        """Add to position quantity."""
        if quantity <= 0:
            return
            
        # Calculate new cost basis
        if self.quantity > 0:
            # Update cost basis as weighted average
            total_cost = self.quantity * self.cost_basis
            new_cost = quantity * price
            self.quantity += quantity
            self.cost_basis = (total_cost + new_cost) / self.quantity
        else:
            # New position
            self.quantity = quantity
            self.cost_basis = price
        
        # Record trade
        self.trades.append({
            'action': 'BUY',
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.datetime.now()
        })
        
        logger.debug(f"Added {quantity} shares of {self.symbol} @ {price:.2f}, new position: {self.quantity} @ {self.cost_basis:.2f}")
    
    def reduce_quantity(self, quantity, price):
        """Reduce position quantity."""
        if quantity <= 0 or self.quantity <= 0:
            return
            
        # Cap reduction at current quantity
        quantity = min(quantity, self.quantity)
        
        # Calculate realized P&L
        trade_pnl = (price - self.cost_basis) * quantity
        self.realized_pnl += trade_pnl
        
        # Update position
        self.quantity -= quantity
        
        # When fully closed, reset cost basis
        if self.quantity <= 0:
            self.cost_basis = 0.0
        
        # Record trade
        self.trades.append({
            'action': 'SELL',
            'quantity': quantity, 
            'price': price,
            'pnl': trade_pnl,
            'timestamp': datetime.datetime.now()
        })
        
        logger.debug(f"Reduced {quantity} shares of {self.symbol} @ {price:.2f}, realized P&L: {trade_pnl:.2f}, remaining: {self.quantity}")
    
    def market_value(self, price=None):
        """Calculate market value of position."""
        if self.quantity == 0:
            return 0.0
            
        # If no price provided, use cost basis (this is crucial for backtesting)
        if price is None:
            price = self.cost_basis
            
        return self.quantity * price
    
    def unrealized_pnl(self, price=None):
        """Calculate unrealized P&L of position."""
        if self.quantity == 0:
            return 0.0
            
        # If no price provided, use cost basis (no unrealized P&L)
        if price is None:
            return 0.0
            
        return (price - self.cost_basis) * self.quantity
    
    def total_pnl(self, price=None):
        """Calculate total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl(price)


class PortfolioManager:
    """
    Tracks the current state of the portfolio with proper P&L accounting.
    Listens for fill events to update positions and cash.
    """
    
    def __init__(self, initial_cash=0.0, event_bus=None):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}  # symbol -> Position
        self.fill_history = []
        self.event_bus = event_bus
        self.last_update_time = datetime.datetime.now()
        
        # Track equity history
        self.equity_history = [{
            'timestamp': datetime.datetime.now(),
            'cash': initial_cash,
            'position_value': 0.0,
            'equity': initial_cash
        }]
        
        # Track total commissions and fees
        self.total_commission = 0.0
        
        # Track all transactions for debugging
        self.transactions = []
        
        # Current market data for valuation
        self.market_data = {}  # symbol -> current price
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
        return self
    
    def on_fill(self, event):
        """Update portfolio state based on fill events."""
        if not isinstance(event, FillEvent):
            logger.warning(f"Non-FillEvent received by portfolio: {type(event)}")
            return
        
        try:
            # Extract fill details
            symbol = event.get_symbol()
            direction = event.get_direction()
            quantity = event.get_quantity()
            price = event.get_price()
            commission = event.get_commission() if hasattr(event, 'get_commission') else 0.0
            
            # Log fill event for debugging
            logger.debug(f"Processing fill: {symbol} {direction} {quantity} @ {price}")
            
            # Track transaction
            self.transactions.append({
                'type': 'FILL',
                'symbol': symbol,
                'direction': direction,
                'quantity': quantity,
                'price': price,
                'commission': commission,
                'timestamp': event.get_timestamp() or datetime.datetime.now()
            })
            
            # Update position
            self._update_position(symbol, direction, quantity, price)
            
            # Update cash
            self._update_cash(direction, quantity, price, commission)
            
            # Update market data with fill price
            self.market_data[symbol] = price
            
            # Add commission to total
            self.total_commission += commission
            
            # Record fill
            self.fill_history.append(event)
            
            # Update timestamp
            self.last_update_time = event.get_timestamp() or datetime.datetime.now()
            
            # Update equity history
            self._update_equity_history()
            
            # Log the updated state
            position_value = self.get_position_value()
            logger.debug(f"Fill processed: {symbol} {direction} {quantity} @ {price:.2f}, " +
                       f"Cash: {self.cash:.2f}, Position Value: {position_value:.2f}")
        except Exception as e:
            logger.error(f"Error processing fill: {e}", exc_info=True)

    def _update_position(self, symbol, direction, quantity, price):
        """Update or create position based on fill."""
        # Create position if it doesn't exist
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)

        # Get the position
        position = self.positions[symbol]

        # Update position based on direction
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
    
    def _update_equity_history(self):
        """Update the equity history."""
        equity = self.get_equity()
        position_value = self.get_position_value()
        
        self.equity_history.append({
            'timestamp': self.last_update_time,
            'cash': self.cash,
            'position_value': position_value,
            'equity': equity
        })
    
    def update_market_data(self, market_prices):
        """
        Update market data for position valuation.
        
        Args:
            market_prices: Dict mapping symbols to current prices
        """
        # Store market data
        if market_prices:
            self.market_data.update(market_prices)
            
            # Update equity history with new prices
            self._update_equity_history()
    
    def get_equity(self, market_prices=None):
        """Calculate total portfolio equity."""
        # If market prices provided, update internal market data
        if market_prices:
            self.update_market_data(market_prices)
            
        # Cash plus position values
        position_value = self.get_position_value()
        return self.cash + position_value
    
    def get_position(self, symbol):
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def get_all_positions(self):
        """Get all positions."""
        return self.positions
    
    def get_position_value(self, market_prices=None):
        """Get total value of all positions."""
        # If market prices provided, update internal market data
        if market_prices:
            self.update_market_data(market_prices)
            
        total_value = 0.0
        
        for symbol, position in self.positions.items():
            # Get current price for the symbol
            price = self.market_data.get(symbol)
                
            # Add position value
            position_value = position.market_value(price)
            total_value += position_value
            
        return total_value
    
    def get_portfolio_summary(self, market_prices=None):
        """Get summary of portfolio state."""
        # If market prices provided, update internal market data
        if market_prices:
            self.update_market_data(market_prices)
            
        # Calculate total realized and unrealized P&L
        realized_pnl = sum(position.realized_pnl for position in self.positions.values())
        
        unrealized_pnl = 0.0
        for symbol, position in self.positions.items():
            price = self.market_data.get(symbol)
            unrealized_pnl += position.unrealized_pnl(price)
        
        return {
            'cash': self.cash,
            'position_value': self.get_position_value(),
            'equity': self.get_equity(),
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': realized_pnl + unrealized_pnl,
            'total_commission': self.total_commission,
            'positions': len(self.positions),
            'fills': len(self.fill_history),
            'last_update': self.last_update_time
        }
    
    def get_position_details(self, market_prices=None):
        """Get detailed position information."""
        # If market prices provided, update internal market data
        if market_prices:
            self.update_market_data(market_prices)
            
        details = []
        
        for symbol, position in self.positions.items():
            # Get current price for the symbol
            price = self.market_data.get(symbol)
                
            details.append({
                'symbol': symbol,
                'quantity': position.quantity,
                'cost_basis': position.cost_basis,
                'market_price': price,
                'market_value': position.market_value(price),
                'realized_pnl': position.realized_pnl,
                'unrealized_pnl': position.unrealized_pnl(price),
                'total_pnl': position.total_pnl(price)
            })
            
        return details
    
    def reset(self):
        """Reset portfolio to initial state."""
        self.cash = self.initial_cash
        self.positions = {}
        self.fill_history = []
        self.transactions = []
        self.total_commission = 0.0
        self.last_update_time = datetime.datetime.now()
        self.market_data = {}
        
        # Reset equity history
        self.equity_history = [{
            'timestamp': datetime.datetime.now(),
            'cash': self.initial_cash,
            'position_value': 0.0,
            'equity': self.initial_cash
        }]
