#!/usr/bin/env python
"""
Data System Example

This script demonstrates how to use the data module with the event system
to build a simple trading system.

Usage:
    python data_system_example.py path/to/data_dir symbol1 [symbol2 ...]

Example:
    python data_system_example.py ./data AAPL MSFT GOOG
"""

# Add functions to create test data
def create_test_data_directory(output_dir):
    """
    Create a directory with test CSV data files.
    
    Args:
        output_dir: Directory to create data files in
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test symbols
    symbols = ['AAPL', 'MSFT', 'GOOG']
    start_date = datetime.datetime(2020, 1, 1)
    
    for symbol in symbols:
        # Create price time series with some randomness but a trend
        np.random.seed(hash(symbol) % 2**32)  # Different seed for each symbol
        
        days = 252  # One year of trading days
        
        # Start prices
        if symbol == 'AAPL':
            base_price = 100.0
        elif symbol == 'MSFT':
            base_price = 200.0
        else:
            base_price = 1500.0
            
        # Create price series with random walk
        prices = np.zeros(days)
        prices[0] = base_price
        
        # Trending component parameters
        trend = 0.1  # Daily trend percentage
        if symbol == 'MSFT':
            trend = 0.15  # Stronger trend for MSFT
            
        # Volatility
        volatility = 0.01
        if symbol == 'GOOG':
            volatility = 0.02  # Higher volatility for GOOG
            
        # Calculate daily changes with trend and volatility
        for i in range(1, days):
            # Random walk with drift
            change = np.random.normal(trend/100, volatility) * prices[i-1]
            prices[i] = max(0.1, prices[i-1] + change)  # Ensure price doesn't go negative
            
        # Create dataframe with price data
        df = pd.DataFrame(
            index=pd.date_range(start=start_date, periods=days, freq='B'),
            columns=['open', 'high', 'low', 'close', 'volume']
        )
        
        # Fill dataframe with price data
        for i in range(days):
            price = prices[i]
            df.iloc[i]['open'] = price * (1 - np.random.uniform(0, 0.005))
            df.iloc[i]['high'] = price * (1 + np.random.uniform(0, 0.01))
            df.iloc[i]['low'] = price * (1 - np.random.uniform(0, 0.01))
            df.iloc[i]['close'] = price
            df.iloc[i]['volume'] = int(np.random.uniform(1000000, 5000000))
            
        # Save dataframe to CSV
        filename = os.path.join(output_dir, f"{symbol}_1d.csv")
        df.to_csv(filename, index=True, date_format='%Y-%m-%d')

import os
import sys
import logging
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core event system components
from core.events.event_types import EventType, Event, BarEvent, SignalEvent
from core.events.event_bus import EventBus
from core.events.event_manager import EventManager
from core.events.event_utils import create_bar_event, create_signal_event

# Import data module components
from data.sources.csv_handler import CSVDataSource
from data.historical_data_handler import HistoricalDataHandler
from data.transformers.resampler import Resampler
from data.transformers.normalizer import Normalizer

# Import other modules if available
try:
    from strategy.strategy_base import StrategyBase
    from execution.execution_engine import ExecutionEngine
    FULL_SYSTEM = True
except ImportError:
    FULL_SYSTEM = False
    logger.warning("Strategy and execution modules not found. Running with simplified components.")

class SimpleMovingAverageStrategy:
    """Simple moving average crossover strategy."""
    
    def __init__(self, symbols, fast_window=10, slow_window=30):
        """
        Initialize the strategy.
        
        Args:
            symbols: List of symbols to trade
            fast_window: Fast moving average window
            slow_window: Slow moving average window
        """
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.prices = {symbol: [] for symbol in self.symbols}
        self.fast_ma = {symbol: None for symbol in self.symbols}
        self.slow_ma = {symbol: None for symbol in self.symbols}
        self.event_bus = None
        self.signals = []
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
    
    def on_bar(self, event):
        """Handle bar events."""
        if not isinstance(event, BarEvent):
            return
            
        symbol = event.get_symbol()
        if symbol not in self.symbols:
            return
            
        # Add price to history
        price = event.get_close()
        self.prices[symbol].append(price)
        
        # Keep only necessary history
        max_window = max(self.fast_window, self.slow_window)
        if len(self.prices[symbol]) > max_window + 10:
            self.prices[symbol] = self.prices[symbol][-(max_window + 10):]
            
        # Calculate moving averages
        if len(self.prices[symbol]) >= self.slow_window:
            fast_ma = sum(self.prices[symbol][-self.fast_window:]) / self.fast_window
            slow_ma = sum(self.prices[symbol][-self.slow_window:]) / self.slow_window
            
            # Store current MAs
            prev_fast_ma = self.fast_ma[symbol]
            prev_slow_ma = self.slow_ma[symbol]
            
            self.fast_ma[symbol] = fast_ma
            self.slow_ma[symbol] = slow_ma
            
            # Check for crossover
            if prev_fast_ma is not None and prev_slow_ma is not None:
                # Buy signal: fast MA crosses above slow MA
                if fast_ma > slow_ma and prev_fast_ma <= prev_slow_ma:
                    signal = create_signal_event(
                        SignalEvent.BUY, price, symbol, 'ma_crossover',
                        metadata={'fast_ma': fast_ma, 'slow_ma': slow_ma}
                    )
                    
                    logger.info(f"BUY SIGNAL: {symbol} - Fast MA ({fast_ma:.2f}) crossed above Slow MA ({slow_ma:.2f})")
                    self.signals.append(signal)
                    
                    if self.event_bus:
                        self.event_bus.emit(signal)
                
                # Sell signal: fast MA crosses below slow MA
                elif fast_ma < slow_ma and prev_fast_ma >= prev_slow_ma:
                    signal = create_signal_event(
                        SignalEvent.SELL, price, symbol, 'ma_crossover',
                        metadata={'fast_ma': fast_ma, 'slow_ma': slow_ma}
                    )
                    
                    logger.info(f"SELL SIGNAL: {symbol} - Fast MA ({fast_ma:.2f}) crossed below Slow MA ({slow_ma:.2f})")
                    self.signals.append(signal)
                    
                    if self.event_bus:
                        self.event_bus.emit(signal)
    
    def reset(self):
        """Reset the strategy state."""
        self.prices = {symbol: [] for symbol in self.symbols}
        self.fast_ma = {symbol: None for symbol in self.symbols}
        self.slow_ma = {symbol: None for symbol in self.symbols}
        self.signals = []


class SimplePortfolio:
    """Simple portfolio for demonstration."""
    
    def __init__(self, initial_capital=100000.0):
        """
        Initialize the portfolio.
        
        Args:
            initial_capital: Initial capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # symbol -> {'quantity': int, 'entry_price': float}
        self.trades = []  # list of trades
        self.equity_curve = []  # list of (timestamp, equity)
        self.event_bus = None
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
    
    def on_signal(self, event):
        """Handle signal events."""
        if not isinstance(event, SignalEvent):
            return
            
        symbol = event.get_symbol()
        price = event.get_price()
        signal = event.get_signal_value()
        timestamp = event.get_timestamp()
        
        # Process signal
        if signal == SignalEvent.BUY:
            # Calculate position size
            position_size = 0.1  # 10% of capital
            available_capital = self.current_capital * position_size
            quantity = int(available_capital / price)
            
            if quantity > 0:
                # Open or add to position
                if symbol not in self.positions:
                    self.positions[symbol] = {'quantity': 0, 'entry_price': 0.0}
                
                # Update position
                cost = quantity * price
                total_quantity = self.positions[symbol]['quantity'] + quantity
                total_cost = (self.positions[symbol]['quantity'] * self.positions[symbol]['entry_price']) + cost
                
                self.positions[symbol] = {
                    'quantity': total_quantity,
                    'entry_price': total_cost / total_quantity
                }
                
                # Update capital
                self.current_capital -= cost
                
                # Record trade
                trade = {
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': quantity,
                    'price': price,
                    'timestamp': timestamp,
                    'cost': cost
                }
                self.trades.append(trade)
                
                logger.info(f"Executed BUY: {quantity} {symbol} @ {price:.2f} = ${cost:.2f}")
                
        elif signal == SignalEvent.SELL:
            # Check if we have a position
            if symbol in self.positions and self.positions[symbol]['quantity'] > 0:
                # Close position
                quantity = self.positions[symbol]['quantity']
                entry_price = self.positions[symbol]['entry_price']
                proceeds = quantity * price
                pnl = proceeds - (quantity * entry_price)
                
                # Update capital
                self.current_capital += proceeds
                
                # Record trade
                trade = {
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': quantity,
                    'price': price,
                    'timestamp': timestamp,
                    'proceeds': proceeds,
                    'pnl': pnl
                }
                self.trades.append(trade)
                
                # Remove position
                self.positions[symbol] = {'quantity': 0, 'entry_price': 0.0}
                
                logger.info(f"Executed SELL: {quantity} {symbol} @ {price:.2f} = ${proceeds:.2f}, P&L: ${pnl:.2f}")
        
        # Update equity curve
        equity = self.calculate_equity(timestamp, price)
        self.equity_curve.append((timestamp, equity))
    
    def calculate_equity(self, timestamp, last_price):
        """
        Calculate current equity.
        
        Args:
            timestamp: Current timestamp
            last_price: Latest price for calculating position values
            
        Returns:
            Current equity value
        """
        equity = self.current_capital
        
        # Add value of open positions using last price
        for symbol, position in self.positions.items():
            if position['quantity'] > 0:
                equity += position['quantity'] * last_price
        
        return equity
    
    def reset(self):
        """Reset the portfolio state."""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
    
    def get_stats(self):
        """
        Get portfolio statistics.
        
        Returns:
            Dictionary of portfolio statistics
        """
        total_pnl = sum(trade.get('pnl', 0) for trade in self.trades)
        num_trades = len(self.trades)
        win_trades = sum(1 for trade in self.trades if trade.get('pnl', 0) > 0)
        loss_trades = sum(1 for trade in self.trades if trade.get('pnl', 0) < 0)
        win_rate = win_trades / num_trades if num_trades > 0 else 0
        
        return {
            'total_pnl': total_pnl,
            'num_trades': num_trades,
            'win_trades': win_trades,
            'loss_trades': loss_trades,
            'win_rate': win_rate,
            'initial_capital': self.initial_capital,
            'final_equity': self.equity_curve[-1][1] if self.equity_curve else self.initial_capital,
            'return_pct': ((self.equity_curve[-1][1] / self.initial_capital) - 1) * 100 if self.equity_curve else 0
        }
    
    def plot_equity_curve(self):
        """Plot the equity curve."""
        if not self.equity_curve:
            logger.warning("No equity curve data to plot")
            return
            
        timestamps = [item[0] for item in self.equity_curve]
        equity = [item[1] for item in self.equity_curve]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, equity)
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def run_backtest(data_dir, symbols, start_date=None, end_date=None, 
                fast_ma=10, slow_ma=30, timeframe='1d', plot_results=True):
    """
    Run a backtest using the data and event system.
    
    Args:
        data_dir: Directory containing data files
        symbols: Symbols to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        fast_ma: Fast moving average window
        slow_ma: Slow moving average window
        timeframe: Data timeframe
        plot_results: Whether to plot results
        
    Returns:
        Dictionary of backtest results
    """
    # Create event system
    event_bus = EventBus()
    event_manager = EventManager(event_bus)
    
    # Create data source and handler
    data_source = CSVDataSource(data_dir)
    data_handler = HistoricalDataHandler(data_source, event_bus)
    
    # Create strategy and portfolio
    strategy = SimpleMovingAverageStrategy(symbols, fast_ma, slow_ma)
    portfolio = SimplePortfolio(initial_capital=100000.0)
    
    # Register components with event manager
    event_manager.register_component('strategy', strategy, [EventType.BAR])
    event_manager.register_component('portfolio', portfolio, [EventType.SIGNAL])
    
    # Load data
    logger.info(f"Loading data for {symbols} from {data_dir}")
    data_handler.load_data(symbols, start_date, end_date, timeframe)
    
    # Run backtest
    logger.info("Running backtest...")
    for symbol in symbols:
        logger.info(f"Processing data for {symbol}")
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
    
    # Get results
    logger.info("Backtest complete")
    stats = portfolio.get_stats()
    
    # Print results
    logger.info(f"Backtest Results:")
    logger.info(f"  Total P&L: ${stats['total_pnl']:.2f}")
    logger.info(f"  Return: {stats['return_pct']:.2f}%")
    logger.info(f"  Trades: {stats['num_trades']}")
    if stats['num_trades'] > 0:
        logger.info(f"  Win Rate: {stats['win_rate']*100:.2f}%")
        logger.info(f"  Win/Loss: {stats['win_trades']}/{stats['loss_trades']}")
    
    # Plot results
    if plot_results and portfolio.equity_curve:
        logger.info("Plotting equity curve...")
        portfolio.plot_equity_curve()
    
    return {
        'stats': stats,
        'portfolio': portfolio,
        'strategy': strategy,
        'data_handler': data_handler
    }


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Algorithmic Trading Data System Example')
    parser.add_argument('data_dir', help='Directory containing data files')
    parser.add_argument('symbols', nargs='+', help='Symbols to trade')
    parser.add_argument('--fast-ma', type=int, default=10, help='Fast MA window')
    parser.add_argument('--slow-ma', type=int, default=30, help='Slow MA window')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', default='1d', help='Data timeframe')
    parser.add_argument('--no-plot', action='store_true', help='Do not plot results')
    parser.add_argument('--create-test-data', action='store_true', help='Create test data')
    
    args = parser.parse_args()
    
    # Create test data if requested
    if args.create_test_data:
        logger.info(f"Creating test data in {args.data_dir}")
        create_test_data_directory(args.data_dir)
    
    # Run backtest
    run_backtest(
        args.data_dir,
        args.symbols,
        args.start_date,
        args.end_date,
        args.fast_ma,
        args.slow_ma,
        args.timeframe,
        not args.no_plot
    )


if __name__ == "__main__":
    main()
