"""
Backtest Validation Script

This script validates a backtesting system by comparing it with an independent
implementation that tracks trades and P&L separately.
"""
import logging
import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from typing import Dict, List, Any, Optional, Tuple, Union

# Import necessary components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType, BarEvent, SignalEvent, OrderEvent
from src.core.events.event_utils import create_bar_event, create_signal_event, create_order_event, create_fill_event
from src.core.events.event_emitters import BarEmitter
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler
from src.strategy.strategies.mean_reversion import MeanReversionStrategy
from src.analytics.performance import PerformanceAnalytics
from src.execution.portfolio import PortfolioManager
from src.execution.brokers.simulated import SimulatedBroker
from src.strategy.risk.risk_manager import SimplePassthroughRiskManager
from src.execution.backtest.backtest import run_backtest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
        
        # Initialize equity history with starting point
        self.equity_history.append({
            'timestamp': datetime.datetime.now(),
            'cash': self.cash,
            'position_value': 0.0,
            'equity': self.cash
        })
        
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
    
    def get_equity_curve(self):
        """Get equity curve as DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.equity_history)
    
    def get_open_positions(self):
        """Get all open positions."""
        return self.positions
    
    def get_closed_trades(self):
        """Get all completed trades."""
        return self.closed_trades
    
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

class IndependentPositionTracker:
    """
    Manually tracks positions and calculates P&L independently from the system's components.
    This allows validation of the backtest logic by comparing results.
    """
    
    def __init__(self, initial_cash=10000.0):
        """Initialize tracker with starting cash"""
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}  # symbol -> {'quantity': int, 'cost_basis': float}
        self.trades = []  # List of trade details
        self.realized_pnl = 0.0  # Cumulative realized P&L
        self.equity_history = []  # List of equity points
        
        # Add initial equity point
        self.equity_history.append({
            'timestamp': datetime.datetime.now(),
            'cash': self.cash,
            'position_value': 0.0,
            'equity': self.cash
        })


    def process_fill(self, fill):
        """
        Process a fill event and update position and P&L

        Args:
            fill: Dictionary with fill information
                {timestamp, symbol, direction, quantity, price}

        Returns:
            float: Trade P&L
        """
        timestamp = fill['timestamp']
        symbol = fill['symbol']
        direction = fill['direction']
        quantity = fill['quantity']
        price = fill['price']
        commission = fill.get('commission', 0.0)  # Optional commission

        # Initialize position for symbol if not exists
        if symbol not in self.positions:
            self.positions[symbol] = {'quantity': 0, 'cost_basis': 0.0}

        # Previous position info for P&L calculation
        old_quantity = self.positions[symbol]['quantity']
        old_cost_basis = self.positions[symbol]['cost_basis']
        trade_pnl = 0.0

        # Calculate new position after the trade
        if direction == 'BUY':
            # For consistency with the system backtest, limit the position to +1 or -1
            # If buying to cover, just reduce position toward 0
            if old_quantity < 0:
                new_quantity = max(-1, old_quantity + quantity)

                # Calculate P&L for the portion that's covered
                covered_quantity = min(abs(old_quantity), quantity)
                trade_pnl = (old_cost_basis - price) * covered_quantity
                self.realized_pnl += trade_pnl

                # If position closed completely
                if new_quantity == 0:
                    new_cost_basis = 0.0
                else:
                    # Still short, maintain cost basis
                    new_cost_basis = old_cost_basis

                logger.info(f"COVERING SHORT: {covered_quantity} shares @ {price:.2f}, " +
                           f"Entry: {old_cost_basis:.2f}, P&L: {trade_pnl:.2f}")
            else:
                # If already long or neutral, limit to +1
                new_quantity = min(1, old_quantity + quantity)

                # Calculate new cost basis
                if old_quantity == 0:
                    new_cost_basis = price
                else:
                    # For an existing long position, maintain cost basis
                    new_cost_basis = old_cost_basis

                logger.info(f"LONG POSITION: Adding {quantity} shares @ {price:.2f}, " +
                           f"New position: {new_quantity} @ {new_cost_basis:.2f}")

            # Update cash
            self.cash -= (quantity * price + commission)

        elif direction == 'SELL':
            # For consistency with the system backtest, limit the position to +1 or -1
            # If selling from a long position, reduce position toward 0
            if old_quantity > 0:
                new_quantity = min(1, max(0, old_quantity - quantity))

                # Calculate P&L for the portion that's sold
                sold_quantity = min(old_quantity, quantity)
                trade_pnl = (price - old_cost_basis) * sold_quantity
                self.realized_pnl += trade_pnl

                # If position closed completely
                if new_quantity == 0:
                    new_cost_basis = 0.0
                else:
                    # Still long, maintain cost basis
                    new_cost_basis = old_cost_basis

                logger.info(f"SELLING LONG: {sold_quantity} shares @ {price:.2f}, " +
                           f"Entry: {old_cost_basis:.2f}, P&L: {trade_pnl:.2f}")
            else:
                # If already short or neutral, limit to -1
                new_quantity = max(-1, old_quantity - quantity)

                # Calculate new cost basis
                if old_quantity == 0:
                    new_cost_basis = price
                else:
                    # For an existing short position, maintain cost basis
                    new_cost_basis = old_cost_basis

                logger.info(f"SHORT POSITION: Adding {quantity} shares @ {price:.2f}, " +
                           f"New position: {new_quantity} @ {new_cost_basis:.2f}")

            # Update cash
            self.cash += (quantity * price - commission)

        # Check for position flips
        position_flipped = (old_quantity > 0 and new_quantity < 0) or (old_quantity < 0 and new_quantity > 0)

        # Update position
        self.positions[symbol] = {
            'quantity': new_quantity,
            'cost_basis': new_cost_basis
        }

        # Store trade
        self.trades.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'position_after': new_quantity,
            'cash_after': self.cash,
            'pnl': trade_pnl,
            'position_flipped': position_flipped
        })

        # Return calculated P&L for this trade
        return trade_pnl
        
  
    
    def update_equity(self, timestamp, market_prices):
        """
        Update equity history with current position values
        
        Args:
            timestamp: Current timestamp
            market_prices: Dict of symbol -> current price
        """
        # Calculate position value
        position_value = 0.0
        for symbol, position in self.positions.items():
            price = market_prices.get(symbol)
            if price is not None:
                position_value += position['quantity'] * price
        
        # Calculate equity
        equity = self.cash + position_value
        
        # Add to history
        self.equity_history.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'position_value': position_value,
            'equity': equity
        })
        
        # Return calculated equity
        return equity
    
    def get_unrealized_pnl(self, market_prices):
        """
        Calculate current unrealized P&L based on market prices
        
        Args:
            market_prices: Dict of symbol -> current price
            
        Returns:
            float: Total unrealized P&L
        """
        unrealized_pnl = 0.0
        
        for symbol, position in self.positions.items():
            quantity = position['quantity']
            cost_basis = position['cost_basis']
            price = market_prices.get(symbol)
            
            if price is not None and quantity != 0:
                if quantity > 0:
                    # Long position
                    unrealized_pnl += (price - cost_basis) * quantity
                else:
                    # Short position
                    unrealized_pnl += (cost_basis - price) * abs(quantity)
        
        return unrealized_pnl
    
    def liquidate_position(self, timestamp, symbol, price):
        """
        Liquidate a specific position at the given price
        
        Args:
            timestamp: Current timestamp
            symbol: Symbol to liquidate
            price: Price to use for liquidation
            
        Returns:
            float: P&L from liquidation
        """
        if symbol not in self.positions or self.positions[symbol]['quantity'] == 0:
            return 0.0
        
        position = self.positions[symbol]
        quantity = position['quantity']
        
        # Create a fill event for liquidation
        fill = {
            'timestamp': timestamp,
            'symbol': symbol,
            'direction': 'SELL' if quantity > 0 else 'BUY',
            'quantity': abs(quantity),
            'price': price
        }
        
        # Process the fill and return P&L
        pnl = self.process_fill(fill)
        return pnl
    
    def liquidate_all_positions(self, timestamp, market_prices):
        """
        Liquidate all positions at current market prices
        
        Args:
            timestamp: Current timestamp
            market_prices: Dict of symbol -> current price
            
        Returns:
            float: Total P&L from liquidation
        """
        total_pnl = 0.0
        
        for symbol in list(self.positions.keys()):
            if self.positions[symbol]['quantity'] != 0:
                price = market_prices.get(symbol)
                if price is not None:
                    pnl = self.liquidate_position(timestamp, symbol, price)
                    total_pnl += pnl
                    logger.info(f"Liquidated {symbol} position at {price:.2f}, P&L: {pnl:.2f}")
        
        return total_pnl
    
    def get_equity_curve(self):
        """
        Return equity curve as DataFrame
        
        Returns:
            DataFrame: Equity curve with timestamp and equity columns
        """
        df = pd.DataFrame(self.equity_history)
        return df
    
    def get_trade_list(self):
        """
        Return trade list
        
        Returns:
            list: List of trade dictionaries
        """
        return self.trades
    
    def get_performance_summary(self):
        """
        Get a summary of performance metrics
        
        Returns:
            dict: Performance metrics dictionary
        """
        if not self.equity_history:
            return {}
            
        initial_equity = self.initial_cash
        final_equity = self.equity_history[-1]['equity']
        total_return = ((final_equity / initial_equity) - 1) * 100
        
        # Calculate drawdown
        equity_values = [point['equity'] for point in self.equity_history]
        max_drawdown = 0.0
        peak = equity_values[0]
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            else:
                drawdown = (peak - equity) / peak * 100 if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate win rate
        trades = self.trades
        pnl_trades = [t for t in trades if t['pnl'] != 0.0]
        win_count = sum(1 for t in pnl_trades if t['pnl'] > 0.0)
        loss_count = sum(1 for t in pnl_trades if t['pnl'] < 0.0)
        win_rate = (win_count / len(pnl_trades) * 100) if pnl_trades else 0.0
        
        # Count trades
        buy_trades = [t for t in trades if t['direction'] == 'BUY']
        sell_trades = [t for t in trades if t['direction'] == 'SELL']
        
        # Count position flips
        position_flips = sum(1 for t in trades if t.get('position_flipped', False))
        
        return {
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'win_count': win_count,
            'loss_count': loss_count,
            'total_trades': len(pnl_trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'position_flips': position_flips,
            'realized_pnl': self.realized_pnl
        }


# Add the TradeTracker class to your latest.py file
# (Add the full class definition from my previous response)

def run_independent_tracking(data_handler, strategy, symbol, lookback=20, z_threshold=1.5, initial_cash=10000.0):
    """
    Run a backtest with proper trade tracking
    """
    logger.info("=== RUNNING INDEPENDENT TRACKING BACKTEST ===")
    
    # Create event system
    event_bus = EventBus()
    event_manager = EventManager(event_bus)
    
    # Create trade tracker instead of position tracker
    tracker = TradeTracker(initial_cash=initial_cash)
    
    # Create regular components
    portfolio = PortfolioManager(initial_cash=initial_cash)
    portfolio.set_event_bus(event_bus)
    
    broker = SimulatedBroker(fill_emitter=event_bus)
    broker.set_event_bus(event_bus)
    
    risk_manager = SimplePassthroughRiskManager(
        portfolio=portfolio,
        event_bus=event_bus,
    )
    risk_manager.broker = broker
    
    # Reset and connect strategy
    strategy.reset()
    strategy.set_event_bus(event_bus)
    
    # Event tracking
    events = {
        'bar': 0,
        'signal': 0,
        'order': 0,
        'fill': 0
    }
    
    # Track market data
    market_data = {}
    
    # Event handlers
    def on_bar(event):
        """Track bar events and update market data"""
        events['bar'] += 1
        symbol = event.get_symbol()
        price = event.get_close()
        timestamp = event.get_timestamp()
        
        # Update market data
        market_data[symbol] = price
        
        # Update broker's market data
        broker.update_market_data(symbol, {
            'price': price,
            'timestamp': timestamp
        })
        
        # Update portfolio's market data
        portfolio.update_market_data({symbol: price})
        
        # Update trade tracker's equity
        tracker.update_equity(timestamp, market_data)
        
        # Log periodic updates
        if events['bar'] % 100 == 0:
            logger.info(f"Processing bar {events['bar']}: {symbol} @ {price:.2f}")
    
    def on_signal(event):
        """Track signal events"""
        events['signal'] += 1
        symbol = event.get_symbol()
        signal_value = event.get_signal_value()
        price = event.get_price()
        logger.debug(f"Signal: {symbol} {signal_value} @ {price:.2f}")
    
    def on_order(event):
        """Track order events"""
        events['order'] += 1
        symbol = event.get_symbol()
        direction = event.get_direction()
        quantity = event.get_quantity()
        price = event.get_price()
        logger.debug(f"Order: {symbol} {direction} {quantity} @ {price:.2f}")
    
    def on_fill(event):
        """Track fill events with proper trade tracking"""
        events['fill'] += 1
        symbol = event.get_symbol()
        direction = event.get_direction()
        quantity = event.get_quantity()
        price = event.get_price()
        commission = event.get_commission() if hasattr(event, 'get_commission') else 0.0
        timestamp = event.get_timestamp()
        
        logger.info(f"Fill: {symbol} {direction} {quantity} @ {price:.2f}")
        
        # Process with trade tracker
        fill_data = {
            'timestamp': timestamp,
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'price': price,
            'commission': commission
        }
        pnl = tracker.process_fill(fill_data)
        
        # Log P&L if significant
        if abs(pnl) > 0.01:
            logger.info(f"Trade P&L: {pnl:.2f}")
    
    # Register event handlers
    event_bus.register(EventType.BAR, on_bar)
    event_bus.register(EventType.SIGNAL, on_signal)
    event_bus.register(EventType.ORDER, on_order)
    event_bus.register(EventType.FILL, on_fill)
    
    # Register system components
    event_manager.register_component('strategy', strategy, [EventType.BAR])
    event_manager.register_component('risk_manager', risk_manager, [EventType.SIGNAL])
    event_manager.register_component('portfolio', portfolio, [EventType.FILL])
    event_bus.register(EventType.ORDER, broker.place_order)
    
    # Reset data handler
    data_handler.reset()
    
    # Process each bar
    logger.info("Starting backtest...")
    
    last_bar = None
    
    while True:
        # Get next bar
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break
            
        # Store reference to the last valid bar
        last_bar = bar
            
        # Process bar through event system
        event_bus.emit(bar)
    
    # Get last bar information for liquidation
    last_timestamp = datetime.datetime.now()
    
    # Use the last valid bar's timestamp if available
    if last_bar is not None:
        last_timestamp = last_bar.get_timestamp()
    
    # Liquidate all positions
    logger.info("Liquidating all positions...")
    liquidation_pnl = tracker.liquidate_positions(last_timestamp, market_data)
    logger.info(f"Liquidation P&L: {liquidation_pnl:.2f}")
    
    # Get trade statistics
    trade_stats = tracker.get_trade_statistics()
    logger.info(f"Trade Statistics: {trade_stats}")
    
    # Get equity curve
    equity_curve = tracker.get_equity_curve()
    
    # Return results with keys matching what compare_results expects
    return {
        'events': events,
        'tracker_metrics': trade_stats,  # Key name that compare_results expects
        'tracker_equity': equity_curve,  # Key name that compare_results expects
        'tracker_trades': tracker.get_closed_trades(),  # Key name that compare_results expects
        'market_data': market_data
    }

def run_system_backtest(data_handler, strategy, symbol, initial_cash=10000.0):
    """
    Run a backtest using the system's own backtest function
    
    Args:
        data_handler: HistoricalDataHandler with data
        strategy: Trading strategy to test
        symbol: Symbol to trade
        initial_cash: Initial portfolio cash
        
    Returns:
        dict: Backtest results from system
    """
    logger.info("=== RUNNING SYSTEM BACKTEST ===")
    
    # Reset components
    strategy.reset()
    data_handler.reset()
    
    try:
        # Run backtest
        equity_curve, trades = run_backtest(
            component=strategy,
            data_handler=data_handler,
            initial_cash=initial_cash
        )
        
        # Check results
        if equity_curve is None or len(equity_curve) == 0:
            logger.warning("System backtest returned empty equity curve")
            equity_curve = pd.DataFrame({
                'timestamp': [datetime.datetime.now()],
                'equity': [initial_cash]
            })
        
        if trades is None:
            logger.warning("System backtest returned None for trades")
            trades = []
            
        # Log trades count
        logger.info(f"System backtest completed - {len(trades)} trades executed")
        
        # Calculate metrics
        metrics = PerformanceAnalytics.calculate_metrics(equity_curve, trades)
        
        # Add trade count explicitly
        metrics['trade_count'] = len(trades)
        
        logger.info("=== SYSTEM BACKTEST COMPLETE ===")
        logger.info(f"Initial equity: ${metrics['initial_equity']:.2f}")
        logger.info(f"Final equity: ${metrics['final_equity']:.2f}")
        logger.info(f"Total return: {metrics['total_return']:.2f}%")
        logger.info(f"Trade count: {metrics['trade_count']}")
        
        return {
            'equity_curve': equity_curve,
            'trades': trades,
            'metrics': metrics
        }
    except Exception as e:
        logger.exception(f"Error in system backtest: {e}")
        # Create minimal results to avoid errors in comparison
        return {
            'equity_curve': pd.DataFrame({
                'timestamp': [datetime.datetime.now()],
                'equity': [initial_cash]
            }),
            'trades': [],
            'metrics': {
                'initial_equity': initial_cash,
                'final_equity': initial_cash,
                'total_return': 0.0,
                'trade_count': 0
            }
        }


def log_system_metrics(metrics):
    """
    Log system metrics in a readable format
    
    Args:
        metrics: Dictionary of metrics
    """
    logger.info("=== SYSTEM METRICS DETAILS ===")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        else:
            logger.info(f"{key}: {type(value)}")
    
    # Special handling for trade-related info
    if 'trade_count' in metrics:
        logger.info(f"trade_count: {metrics['trade_count']}")
    elif 'fills' in metrics:
        logger.info(f"fills: {metrics['fills']}")
    
    # Check for nested structures
    for key, value in metrics.items():
        if isinstance(value, dict) and value:
            logger.info(f"{key} contents: {list(value.keys())}")



def compare_results(independent_results, system_results):
    """Compare results from independent tracking and system backtest."""
    logger.info("=== COMPARING BACKTEST RESULTS ===")
    
    # Extract metrics
    ind_metrics = independent_results['tracker_metrics']
    
    # For system metrics, prioritize trade statistics over return calculation
    if 'metrics' in system_results and 'total_pnl' in system_results['metrics']:
        sys_metrics = system_results['metrics']
        sys_trade_count = sys_metrics.get('trade_count', 0)
        sys_total_pnl = sys_metrics.get('total_pnl', 0)
    else:
        # Fallback: calculate from trade list
        sys_trades = system_results.get('trades', [])
        sys_trade_count = len(sys_trades)
        sys_total_pnl = sum(t.get('pnl', 0) for t in sys_trades if isinstance(t, dict))
    
    # Extract independent metrics
    ind_trade_count = ind_metrics.get('total_trades', 0)
    ind_total_pnl = ind_metrics.get('total_pnl', 0)
    
    # Compare metrics
    results = {
        'metrics_compared': {
            'trade_count': {
                'independent': ind_trade_count,
                'system': sys_trade_count,
                'match': ind_trade_count == sys_trade_count
            },
            'total_pnl': {
                'independent': ind_total_pnl,
                'system': sys_total_pnl,
                'match': abs(ind_total_pnl - sys_total_pnl) < 0.01
            }
        }
    }
    
    # Check overall match
    validation_passed = all(item['match'] for item in results['metrics_compared'].values())
    results['validation_passed'] = validation_passed
    
    # Log results
    logger.info("Validation Results:")
    for metric, data in results['metrics_compared'].items():
        match_str = "✓" if data['match'] else "✗"
        logger.info(f"{match_str} {metric}: Independent={data['independent']}, System={data['system']}")
    
    if validation_passed:
        logger.info("✓ VALIDATION PASSED - Independent tracking matches system results")
    else:
        logger.warning("✗ VALIDATION FAILED - Discrepancies detected")
    
    return results            

def plot_comparison(independent_results, system_results, symbol, save_path='backtest_validation_comparison.png'):
    """
    Plot equity curves from both tracking methods for comparison
    
    Args:
        independent_results: Results from independent tracking
        system_results: Results from system backtest
        symbol: Symbol traded
        save_path: Path to save the plot
        
    Returns:
        True if successful, False if error
    """
    try:
        # Extract equity curves
        ind_equity = independent_results['tracker_equity']
        sys_equity = system_results['equity_curve']
        
        # Make sure timestamps are datetime objects
        if 'timestamp' in ind_equity.columns:
            ind_equity['timestamp'] = pd.to_datetime(ind_equity['timestamp'])
        if 'timestamp' in sys_equity.columns:
            sys_equity['timestamp'] = pd.to_datetime(sys_equity['timestamp'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot both equity curves
        if 'timestamp' in ind_equity.columns and 'equity' in ind_equity.columns:
            ax.plot(ind_equity['timestamp'], ind_equity['equity'], 
                   label='Independent Tracking', linewidth=2)
            
        if 'timestamp' in sys_equity.columns and 'equity' in sys_equity.columns:
            ax.plot(sys_equity['timestamp'], sys_equity['equity'], 
                   label='System Backtest', linewidth=2, linestyle='--')
        
        # Set labels and title
        ax.set_title(f'Backtest Validation Comparison - {symbol}', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Equity ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=12)
        
        # Format dates on x-axis
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Add metrics as text
        ind_metrics = independent_results['tracker_metrics']
        sys_metrics = system_results['metrics']
        
        metrics_text = (
            f"Independent: Return={ind_metrics['total_return']:.2f}%, "
            f"Trades={ind_metrics['total_trades']}\n"
            f"System: Return={sys_metrics['total_return']:.2f}%, "
            f"Trades={sys_metrics['trade_count']}"
        )
        
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path)
        logger.info(f"Comparison plot saved to {save_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error creating comparison plot: {e}")
        return False


def generate_trade_comparison_report(independent_results, system_results):
    """
    Generate a detailed report comparing how both systems handle specific trades.
    
    Args:
        independent_results: Results from independent tracking
        system_results: Results from system backtest
        
    Returns:
        str: Detailed comparison report
    """
    # Extract trades from both systems
    independent_trades = independent_results.get('tracker_trades', [])
    system_trades = system_results.get('trades', [])
    
    # Independent trades now have entry_time/exit_time instead of timestamp
    # Sort trades by entry time
    independent_trades = sorted(independent_trades, key=lambda x: x.get('entry_time', datetime.datetime.now()))
    
    # System trades may vary in format
    if system_trades and isinstance(system_trades[0], dict) and 'timestamp' in system_trades[0]:
        system_trades = sorted(system_trades, key=lambda x: x.get('timestamp', datetime.datetime.now()))
    
    # Create report
    report = ["# Trade-by-Trade Comparison Report\n"]
    report.append("This report compares how the independent tracker and system backtest handle trades.\n")
    
    # Add summary statistics
    ind_stats = independent_results.get('tracker_metrics', {})
    
    report.append("## Summary Statistics\n")
    report.append(f"- Independent Tracker: {ind_stats.get('total_trades', 0)} trades")
    report.append(f"- System Backtest: {len(system_trades)} trades")
    
    # Calculate total P&L from both systems
    ind_pnl = ind_stats.get('total_pnl', 0)
    sys_pnl = 0
    for trade in system_trades:
        if isinstance(trade, dict) and 'pnl' in trade:
            sys_pnl += trade['pnl']
    
    report.append(f"- Independent Total P&L: ${ind_pnl:.2f}")
    report.append(f"- System Total P&L: ${sys_pnl:.2f}")
    report.append("")
    
    # Sample detailed trade analysis (first 5 trades from each system)
    report.append("## Sample Trade Analysis\n")
    
    # Analyze first 5 independent trades
    report.append("### Independent Tracker Trades (First 5)\n")
    for i, trade in enumerate(independent_trades[:5]):
        report.append(f"#### Trade {i+1}:")
        report.append(f"- Entry Time: {trade.get('entry_time')}")
        report.append(f"- Exit Time: {trade.get('exit_time')}")
        report.append(f"- Symbol: {trade.get('symbol')}")
        report.append(f"- Direction: {trade.get('entry_direction')}")
        report.append(f"- Quantity: {trade.get('quantity')}")
        report.append(f"- Entry Price: ${trade.get('entry_price', 0):.2f}")
        report.append(f"- Exit Price: ${trade.get('exit_price', 0):.2f}")
        report.append(f"- P&L: ${trade.get('pnl', 0):.2f}")
        report.append(f"- Commission: ${trade.get('commission', 0):.2f}")
        report.append("")
    
    # Analyze first 5 system trades
    report.append("### System Backtest Trades (First 5)\n")
    for i, trade in enumerate(system_trades[:5]):
        report.append(f"#### Trade {i+1}:")
        # Handle different trade formats
        if isinstance(trade, dict):
            if 'timestamp' in trade:
                report.append(f"- Timestamp: {trade.get('timestamp')}")
            if 'symbol' in trade:
                report.append(f"- Symbol: {trade.get('symbol')}")
            if 'direction' in trade:
                report.append(f"- Direction: {trade.get('direction')}")
            if 'quantity' in trade:
                report.append(f"- Quantity: {trade.get('quantity')}")
            if 'price' in trade:
                report.append(f"- Price: ${trade.get('price', 0):.2f}")
            if 'pnl' in trade:
                report.append(f"- P&L: ${trade.get('pnl', 0):.2f}")
            if 'commission' in trade:
                report.append(f"- Commission: ${trade.get('commission', 0):.2f}")
        else:
            # Handle tuple or list format if that's what the system uses
            report.append(f"- Raw trade data: {trade}")
        report.append("")
    
    # Add findings and recommendations
    report.append("\n## Key Findings\n")
    
    # Trade count discrepancy
    if ind_stats.get('total_trades', 0) != len(system_trades):
        report.append(f"1. **Trade Count Discrepancy**: Independent tracker recorded {ind_stats.get('total_trades', 0)} trades while system backtest recorded {len(system_trades)} trades.")
    
    # P&L discrepancy
    if abs(ind_pnl - sys_pnl) > 0.01:
        report.append(f"2. **P&L Discrepancy**: Independent tracker shows ${ind_pnl:.2f} total P&L while system backtest shows ${sys_pnl:.2f}.")
    
    # Final recommendations
    report.append("\n## Recommendations\n")
    
    if ind_stats.get('total_trades', 0) != len(system_trades):
        report.append("1. **Standardize Trade Counting**: Ensure both systems use the same definition of a 'trade' (entry and exit pair).")
    
    if abs(ind_pnl - sys_pnl) > 0.01:
        report.append("2. **Verify P&L Calculation**: Check that both systems calculate P&L using the same method, especially for closing positions.")
    
    return "\n".join(report)    


def run_validation_test(data_path='data/SAMPLE_1m.csv', lookback=20, z_threshold=1.5, initial_cash=10000.0):
    """
    Run a validation test comparing independent tracking with system backtest
    
    Args:
        data_path: Path to CSV data file
        lookback: Lookback period for mean reversion
        z_threshold: Z-score threshold for mean reversion
        initial_cash: Initial portfolio cash
        
    Returns:
        dict: Validation results
    """
    logger.info("=== STARTING BACKTEST VALIDATION TEST ===")
    
    # Check if data file exists
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return {'validation_passed': False, 'error': 'Data file not found'}
    
    try:
        # Set up data source
        data_dir = os.path.dirname(data_path)
        filename = os.path.basename(data_path)
        
        # Extract symbol from filename (assuming format SYMBOL_timeframe.csv)
        symbol = filename.split('_')[0]
        
        # Create data source
        data_source = CSVDataSource(
            data_dir=data_dir, 
            filename_pattern=filename  # Use exact filename
        )
        
        # Create bar emitter
        event_bus = EventBus()
        bar_emitter = BarEmitter(name="bar_emitter", event_bus=event_bus)
        bar_emitter.start()
        
        # Create data handler
        data_handler = HistoricalDataHandler(
            data_source=data_source,
            bar_emitter=bar_emitter
        )
        
        # Load data
        data_handler.load_data(symbols=[symbol])
        
        # Check if data was loaded
        if symbol not in data_handler.data_frames:
            logger.error(f"Failed to load data for {symbol}")
            return {'validation_passed': False, 'error': 'Failed to load data'}
        
        # Log data summary
        df = data_handler.data_frames[symbol]
        logger.info(f"Loaded {len(df)} bars for {symbol} from {df.index.min()} to {df.index.max()}")
        
        # Create strategy for first test
        strategy1 = MeanReversionStrategy(
            name="mean_reversion_1",
            symbols=[symbol],
            lookback=lookback,
            z_threshold=z_threshold
        )
        
        try:
            # Run independent tracking test
            independent_results = run_independent_tracking(
                data_handler=data_handler,
                strategy=strategy1,
                symbol=symbol,
                lookback=lookback,
                z_threshold=z_threshold,
                initial_cash=initial_cash
            )
        except Exception as e:
            logger.exception(f"Error during independent tracking: {e}")
            return {'validation_passed': False, 'error': f'Independent tracking failed: {str(e)}'}
        
        # Create strategy for second test (identical but fresh instance)
        strategy2 = MeanReversionStrategy(
            name="mean_reversion_2",
            symbols=[symbol],
            lookback=lookback,
            z_threshold=z_threshold
        )
        
        try:
            # Run system backtest
            system_results = run_system_backtest(
                data_handler=data_handler,
                strategy=strategy2,
                symbol=symbol,
                initial_cash=initial_cash
            )
        except Exception as e:
            logger.exception(f"Error during system backtest: {e}")
            # We can still report independent results even if system backtest fails
            return {
                'validation_passed': False, 
                'error': f'System backtest failed: {str(e)}',
                'independent_results': independent_results
            }
        
        try:
            # Compare results
            comparison = compare_results(independent_results, system_results)
            
            # Generate detailed trade comparison report
            trade_report = generate_trade_comparison_report(independent_results, system_results)
            
            # Save trade report to file
            report_filename = "trade_comparison_report.md"
            with open(report_filename, 'w') as f:
                f.write(trade_report)
            logger.info(f"Trade comparison report written to {report_filename}")
            
            # Plot comparison
            plot_comparison(independent_results, system_results, symbol)
            
            # Return combined results
            return {
                'validation_passed': comparison['validation_passed'],
                'independent_results': independent_results,
                'system_results': system_results,
                'comparison': comparison,
                'trade_report': trade_report
            }
        except Exception as e:
            logger.exception(f"Error during results comparison: {e}")
            # Return the individual results even if comparison fails
            return {
                'validation_passed': False,
                'error': f'Results comparison failed: {str(e)}',
                'independent_results': independent_results,
                'system_results': system_results
            }
        
    except Exception as e:
        logger.exception(f"Error during validation test: {e}")
        return {'validation_passed': False, 'error': str(e)}



if __name__ == "__main__":
    # Run validation test
    test_results = run_validation_test()
    
    # Display final result
    if test_results.get('validation_passed', False):
        print("\n✓ VALIDATION TEST PASSED!")
        
        # Extract key metrics for display
        if 'comparison' in test_results and 'metrics_compared' in test_results['comparison']:
            metrics = test_results['comparison']['metrics_compared']
            
            print("\nMetrics Comparison:")
            print(f"Trade Count: Independent={metrics['trade_count']['independent']}, "
                 f"System={metrics['trade_count']['system']}")
            print(f"Total P&L: Independent=${metrics['total_pnl']['independent']:.2f}, "
                 f"System=${metrics['total_pnl']['system']:.2f}")
    else:
        print("\n✗ VALIDATION TEST FAILED!")
        if 'error' in test_results:
            print(f"Error: {test_results['error']}")
        
        # If we have partial results, still show them
        if 'independent_results' in test_results and 'tracker_metrics' in test_results['independent_results']:
            ind_metrics = test_results['independent_results']['tracker_metrics']
            
            print("\nIndependent Tracking Results:")
            print(f"Total Trades: {ind_metrics.get('total_trades', 0)}")
            
            if 'total_pnl' in ind_metrics:
                print(f"Total P&L: ${ind_metrics['total_pnl']:.2f}")
            
            if 'win_rate' in ind_metrics:
                print(f"Win Rate: {ind_metrics['win_rate']:.2f}%")
    
    # Display trade report information
    if 'trade_report' in test_results:
        print("\nDetailed trade-by-trade comparison report has been generated!")
        print("Review 'trade_comparison_report.md' for in-depth analysis.")
    
    print("\nCheck logs for details.")    
