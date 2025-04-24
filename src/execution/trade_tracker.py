# src/execution/trade_tracker.py

import datetime
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class TradeTracker:
    """
    Tracks complete trades (entry and exit) rather than individual fills.
    """
    
    def __init__(self, initial_cash=10000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}  # symbol -> {'quantity': int, 'direction': str, 'entry_price': float, 'entry_time': timestamp}
        self.closed_trades = []  # List of complete trades
        self.open_fills = []  # List of individual fills before position closure
        self.equity_history = []  # Track equity over time
        
    def process_fill(self, fill):
        """Process a fill and track as part of a complete trade."""
        symbol = fill['symbol']
        direction = fill['direction']
        quantity = fill['quantity']
        price = fill['price']
        timestamp = fill['timestamp']
        commission = fill.get('commission', 0.0)
        
        # Update cash immediately
        if direction == 'BUY':
            self.cash -= (quantity * price + commission)
        else:  # SELL
            self.cash += (quantity * price - commission)
        
        # Track as open fill
        self.open_fills.append(fill)
        
        # Check if we have a position for this symbol
        if symbol not in self.positions:
            # Opening a new position
            self.positions[symbol] = {
                'quantity': quantity if direction == 'BUY' else -quantity,
                'direction': direction,
                'entry_price': price,
                'entry_time': timestamp
            }
            # No P&L to calculate when opening a position
            return 0.0
        
        # Existing position
        position = self.positions[symbol]
        current_qty = position['quantity']
        
        # Check if this fill closes or reduces the position
        if (current_qty > 0 and direction == 'SELL') or (current_qty < 0 and direction == 'BUY'):
            # This is closing or reducing a position - calculate P&L
            entry_price = position['entry_price']
            close_qty = min(abs(current_qty), quantity)
            
            # Calculate P&L
            if current_qty > 0:  # Long position
                pnl = (price - entry_price) * close_qty - commission
            else:  # Short position
                pnl = (entry_price - price) * close_qty - commission
            
            # Update position
            new_qty = current_qty + (quantity if direction == 'BUY' else -quantity)
            
            # Record complete trade if position fully closed
            if (current_qty > 0 and new_qty <= 0) or (current_qty < 0 and new_qty >= 0):
                # Complete trade
                self.closed_trades.append({
                    'symbol': symbol,
                    'entry_direction': position['direction'],
                    'entry_price': position['entry_price'],
                    'entry_time': position['entry_time'],
                    'exit_price': price,
                    'exit_time': timestamp,
                    'quantity': close_qty,
                    'pnl': pnl,
                    'commission': commission
                })
                
                # Clear position if fully closed
                if new_qty == 0:
                    del self.positions[symbol]
                else:
                    # Position flipped - record the new position
                    self.positions[symbol] = {
                        'quantity': new_qty,
                        'direction': 'BUY' if new_qty > 0 else 'SELL',
                        'entry_price': price,
                        'entry_time': timestamp
                    }
            else:
                # Position partially closed - update quantity
                self.positions[symbol]['quantity'] = new_qty
            
            return pnl
        else:
            # Adding to existing position
            old_value = abs(current_qty) * position['entry_price']
            new_value = quantity * price
            total_qty = current_qty + (quantity if direction == 'BUY' else -quantity)
            
            # Update position with new average entry price
            self.positions[symbol] = {
                'quantity': total_qty,
                'direction': position['direction'],
                'entry_price': (old_value + new_value) / abs(total_qty),
                'entry_time': position['entry_time']  # Keep original entry time
            }
            
            # No P&L for adding to a position
            return 0.0
    
    def update_equity(self, timestamp, market_prices):
        """Update equity history with current values."""
        position_value = 0.0
        
        # Calculate position values using market prices
        for symbol, position in self.positions.items():
            price = market_prices.get(symbol)
            if price is not None:
                position_value += position['quantity'] * price
        
        # Calculate total equity
        equity = self.cash + position_value
        
        # Add to history
        self.equity_history.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'position_value': position_value,
            'equity': equity
        })
        
        return equity
    
    def get_open_positions(self):
        """Get all open positions."""
        return self.positions
    
    def get_closed_trades(self):
        """Get all completed trades."""
        return self.closed_trades
    
    def get_equity_curve(self):
        """Get equity curve as DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.equity_history)
    
    def get_trade_statistics(self):
        """Calculate statistics on completed trades."""
        if not self.closed_trades:
            return {
                'total_trades': 0,
                'win_count': 0,
                'loss_count': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'average_pnl': 0
            }
        
        win_trades = [t for t in self.closed_trades if t['pnl'] > 0]
        loss_trades = [t for t in self.closed_trades if t['pnl'] <= 0]
        
        total_pnl = sum(t['pnl'] for t in self.closed_trades)
        
        return {
            'total_trades': len(self.closed_trades),
            'win_count': len(win_trades),
            'loss_count': len(loss_trades),
            'win_rate': len(win_trades) / len(self.closed_trades) * 100 if self.closed_trades else 0,
            'total_pnl': total_pnl,
            'average_pnl': total_pnl / len(self.closed_trades) if self.closed_trades else 0
        }
    
    def liquidate_positions(self, timestamp, prices):
        """Liquidate all open positions at specified prices."""
        liquidation_pnl = 0.0
        
        for symbol, position in list(self.positions.items()):
            price = prices.get(symbol)
            if price is None:
                continue
            
            quantity = abs(position['quantity'])
            direction = 'SELL' if position['quantity'] > 0 else 'BUY'
            
            # Create liquidation fill
            fill = {
                'timestamp': timestamp,
                'symbol': symbol,
                'direction': direction,
                'quantity': quantity,
                'price': price,
                'commission': 0.0  # Often no commission on liquidation in backtest
            }
            
            # Process the fill
            pnl = self.process_fill(fill)
            liquidation_pnl += pnl
        
        return liquidation_pnl
