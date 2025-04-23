#!/usr/bin/env python
"""
Moving Average Crossover Strategy Backtest Demo
with Train-Test Split

This script demonstrates:
1. Loading SPY minute data
2. Creating a train-test split
3. Optimizing the MA parameters on the training set
4. Testing the optimized parameters on the test set
5. Visualizing and reporting performance metrics
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Any, Tuple

# Set pandas display options
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

# ----------- STRATEGY IMPLEMENTATION -----------

class MAStrategy:
    """Simple Moving Average Crossover Strategy"""
    
    def __init__(self, fast_window: int, slow_window: int):
        """
        Initialize MA strategy with window sizes
        
        Args:
            fast_window: Period for fast moving average
            slow_window: Period for slow moving average
        """
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.positions = []
        self.current_position = 0  # 0 = no position, 1 = long, -1 = short
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals using MA crossover
        
        Args:
            data: DataFrame with price data (requires 'Close' column)
            
        Returns:
            DataFrame with added signals
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate moving averages
        df['fast_ma'] = df['Close'].rolling(window=self.fast_window).mean()
        df['slow_ma'] = df['Close'].rolling(window=self.slow_window).mean()
        
        # Initialize signal column
        df['signal'] = 0
        
        # Find crossover points
        # Buy when fast MA crosses above slow MA
        df.loc[(df['fast_ma'] > df['slow_ma']) & 
               (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1)), 'signal'] = 1
        
        # Sell when fast MA crosses below slow MA
        df.loc[(df['fast_ma'] < df['slow_ma']) & 
               (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1)), 'signal'] = -1
        
        return df
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 10000.0) -> pd.DataFrame:
        """
        Backtest the strategy on historical data
        
        Args:
            data: DataFrame with price data
            initial_capital: Starting capital
            
        Returns:
            DataFrame with equity curve and trade data
        """
        # Generate signals
        df = self.generate_signals(data)
        
        # Initialize portfolio columns
        df['position'] = 0
        df['holdings'] = 0.0
        df['cash'] = initial_capital
        df['equity'] = initial_capital
        
        # Reset positions list
        self.positions = []
        self.current_position = 0
        
        # Process each bar
        for i in range(len(df)):
            # Skip if we don't have enough data for moving averages
            if i < self.slow_window:
                continue
                
            # Get current bar data
            current_row = df.iloc[i]
            signal = current_row['signal']
            price = current_row['Close']
            timestamp = df.index[i]
            
            # Update position based on signal
            if signal == 1 and self.current_position <= 0:  # Buy signal
                # Calculate shares to buy with all available cash
                shares_to_buy = df.loc[df.index[i-1], 'cash'] // price
                
                if shares_to_buy > 0:
                    # Update position and record trade
                    self.current_position = 1
                    df.loc[df.index[i]:, 'position'] = shares_to_buy
                    
                    # Record trade
                    self.positions.append({
                        'entry_time': timestamp,
                        'entry_price': price,
                        'direction': 'BUY',
                        'shares': shares_to_buy,
                        'exit_time': None,
                        'exit_price': None,
                        'pnl': None,
                        'return': None
                    })
            
            elif signal == -1 and self.current_position >= 0:  # Sell signal
                if self.current_position > 0 and len(self.positions) > 0:
                    # Sell all shares
                    shares_to_sell = df.loc[df.index[i-1], 'position']
                    
                    if shares_to_sell > 0:
                        # Update position
                        self.current_position = -1
                        df.loc[df.index[i]:, 'position'] = 0
                        
                        # Update last trade with exit details
                        last_trade = self.positions[-1]
                        last_trade['exit_time'] = timestamp
                        last_trade['exit_price'] = price
                        last_trade['pnl'] = (price - last_trade['entry_price']) * last_trade['shares']
                        last_trade['return'] = (price / last_trade['entry_price']) - 1
            
            # Calculate holdings value
            df.loc[df.index[i], 'holdings'] = df.loc[df.index[i], 'position'] * price
            
            # Calculate cash (initial cash - holdings)
            if i > 0:
                prev_position = df.loc[df.index[i-1], 'position']
                curr_position = df.loc[df.index[i], 'position']
                
                # Position increased (bought shares)
                if curr_position > prev_position:
                    shares_bought = curr_position - prev_position
                    df.loc[df.index[i], 'cash'] = df.loc[df.index[i-1], 'cash'] - (shares_bought * price)
                
                # Position decreased (sold shares)
                elif curr_position < prev_position:
                    shares_sold = prev_position - curr_position
                    df.loc[df.index[i], 'cash'] = df.loc[df.index[i-1], 'cash'] + (shares_sold * price)
                
                # Position unchanged
                else:
                    df.loc[df.index[i], 'cash'] = df.loc[df.index[i-1], 'cash']
            
            # Calculate total equity
            df.loc[df.index[i], 'equity'] = df.loc[df.index[i], 'cash'] + df.loc[df.index[i], 'holdings']
        
        # Close any open position at the end using the last price
        if self.current_position > 0 and len(self.positions) > 0:
            last_price = df['Close'].iloc[-1]
            last_time = df.index[-1]
            last_trade = self.positions[-1]
            
            if last_trade['exit_time'] is None:
                last_trade['exit_time'] = last_time
                last_trade['exit_price'] = last_price
                last_trade['pnl'] = (last_price - last_trade['entry_price']) * last_trade['shares']
                last_trade['return'] = (last_price / last_trade['entry_price']) - 1
        
        return df
        
    def get_trades(self) -> List[Dict]:
        """Get list of trades executed in the backtest"""
        return self.positions

# ----------- PERFORMANCE ANALYSIS -----------

def calculate_metrics(equity_curve: pd.Series, trades: List[Dict]) -> Dict[str, float]:
    """
    Calculate performance metrics
    
    Args:
        equity_curve: Series of portfolio equity values
        trades: List of trade dictionaries
        
    Returns:
        Dictionary of performance metrics
    """
    # Return metrics
    initial_value = equity_curve.iloc[0]
    final_value = equity_curve.iloc[-1]
    total_return = (final_value / initial_value) - 1
    
    # Annualized return
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = days / 365.0
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve / rolling_max) - 1
    max_drawdown = drawdown.min()
    
    # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
    returns = equity_curve.pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    
    # Trade statistics
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] <= 0]
    
    win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
    
    avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
    
    profit_factor = abs(sum(t['pnl'] for t in winning_trades) / 
                       sum(t['pnl'] for t in losing_trades)) if losing_trades and sum(t['pnl'] for t in losing_trades) != 0 else float('inf')
    
    return {
        'initial_equity': initial_value,
        'final_equity': final_value,
        'total_return': total_return,
        'total_return_pct': total_return * 100,
        'annual_return': annual_return,
        'annual_return_pct': annual_return * 100,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown * 100,
        'sharpe_ratio': sharpe_ratio,
        'trade_count': len(trades),
        'win_rate': win_rate,
        'win_rate_pct': win_rate * 100,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor
    }

def plot_backtest_results(df: pd.DataFrame, title: str = "Backtest Results"):
    """
    Plot backtest results
    
    Args:
        df: DataFrame with backtest results
        title: Plot title
    """
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Price and Moving Averages
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['Close'], label='Close Price', alpha=0.5)
    # Get moving average windows from column names if possible
    fast_window = strategy.fast_window if 'strategy' in globals() else ""
    slow_window = strategy.slow_window if 'strategy' in globals() else ""
    plt.plot(df.index, df['fast_ma'], label=f'Fast MA ({fast_window})', linewidth=1)
    plt.plot(df.index, df['slow_ma'], label=f'Slow MA ({slow_window})', linewidth=1)
    
    # Plot buy and sell signals
    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]
    
    plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='Buy Signal')
    plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', s=100, label='Sell Signal')
    
    plt.title(f'{title} - Price & Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Position Size
    plt.subplot(3, 1, 2)
    plt.plot(df.index, df['position'], label='Position Size')
    plt.title('Position Size')
    plt.xlabel('Date')
    plt.ylabel('Shares')
    plt.grid(True)
    
    # Plot 3: Equity Curve
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['equity'], label='Portfolio Value')
    
    # Add buy and sell markers on equity curve
    plt.scatter(buy_signals.index, buy_signals['equity'], marker='^', color='green', s=100)
    plt.scatter(sell_signals.index, sell_signals['equity'], marker='v', color='red', s=100)
    
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Value ($)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def print_metrics(metrics: Dict[str, float], title: str = "Backtest Results"):
    """
    Print performance metrics
    
    Args:
        metrics: Dictionary of performance metrics
        title: Title for the metrics report
    """
    print("\n" + "="*50)
    print(f" {title} ")
    print("="*50)
    
    print(f"Initial Capital: ${metrics['initial_equity']:.2f}")
    print(f"Final Capital: ${metrics['final_equity']:.2f}")
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Annual Return: {metrics['annual_return_pct']:.2f}%")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    print("\nTrade Statistics:")
    print(f"Number of Trades: {metrics['trade_count']}")
    print(f"Win Rate: {metrics['win_rate_pct']:.2f}%")
    print(f"Average Win: ${metrics['avg_win']:.2f}")
    print(f"Average Loss: ${metrics['avg_loss']:.2f}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    
    print("="*50)

def optimize_parameters(data: pd.DataFrame, fast_range: List[int], slow_range: List[int]) -> Tuple[int, int, float]:
    """
    Grid search for optimal MA parameters
    
    Args:
        data: DataFrame with price data
        fast_range: List of values to test for fast MA
        slow_range: List of values to test for slow MA
        
    Returns:
        Tuple of (best_fast, best_slow, best_score)
    """
    best_sharpe = -float('inf')
    best_fast = 0
    best_slow = 0
    
    results = []
    
    print(f"Optimizing parameters: testing {len(fast_range) * len(slow_range)} combinations...")
    total_combinations = len(fast_range) * len(slow_range)
    completed = 0
    
    for fast in fast_range:
        for slow in slow_range:
            # Skip invalid combinations
            if fast >= slow:
                continue
                
            completed += 1
            if completed % 10 == 0 or completed == total_combinations:
                print(f"Progress: {completed}/{total_combinations} combinations tested ({completed/total_combinations*100:.1f}%)")
                
            # Initialize and backtest strategy
            strategy = MAStrategy(fast, slow)
            df = strategy.backtest(data)
            trades = strategy.get_trades()
            
            # Calculate metrics
            if len(trades) > 0:
                metrics = calculate_metrics(df['equity'], trades)
                sharpe = metrics['sharpe_ratio']
                
                results.append({
                    'fast': fast,
                    'slow': slow,
                    'sharpe': sharpe,
                    'return': metrics['total_return_pct'],
                    'trades': metrics['trade_count'],
                    'win_rate': metrics['win_rate_pct']
                })
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_fast = fast
                    best_slow = slow
    
    # Print top 5 parameter combinations
    if results:
        results_df = pd.DataFrame(results)
        print("\nTop 5 parameter combinations:")
        print(results_df.sort_values('sharpe', ascending=False).head(5))
    
    return best_fast, best_slow, best_sharpe

# ----------- MAIN BACKTEST SCRIPT -----------

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    data_path = "data/SPY_1m.csv"
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        sys.exit(1)
    
    # Convert timestamp to datetime and set as index
    # Using utc=True to prevent timezone mixing warnings
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.set_index('timestamp', inplace=True)
    
    # Check column names - they might be capitalized or lowercase
    # Map whichever version exists to the standard names
    column_mapping = {}
    for std_name, possible_names in {
        'Open': ['Open', 'open', 'OPEN'],
        'High': ['High', 'high', 'HIGH'],
        'Low': ['Low', 'low', 'LOW'],
        'Close': ['Close', 'close', 'CLOSE'],
        'Volume': ['Volume', 'volume', 'VOLUME']
    }.items():
        for col_name in possible_names:
            if col_name in df.columns:
                column_mapping[col_name] = std_name
                break
    
    # Rename columns for consistency
    if column_mapping:
        df.rename(columns=column_mapping, inplace=True)
    
    # Drop rows with missing values in critical columns
    df.dropna(subset=['Close'], inplace=True)
    
    # Print basic dataset info
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    # Handle timezone information consistently
    if df.index.tzinfo is not None:
        print(f"Data timezone: {df.index.tzinfo}")
    else:
        # Check if individual timestamps have timezone info
        has_tz = False
        if len(df) > 0:
            first_idx = df.index[0]
            if hasattr(first_idx, 'tzinfo') and first_idx.tzinfo is not None:
                has_tz = True
                print(f"Data timezone: {first_idx.tzinfo}")
        
        if not has_tz:
            print("Data has no timezone information, assuming local timezone")
            # Don't force localization - work with naive timestamps to avoid issues
            # If needed, can convert to UTC with: df.index = df.index.tz_localize('UTC')
    
    # Create train-test split (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Training set: {len(train_df)} bars from {train_df.index[0]} to {train_df.index[-1]}")
    print(f"Testing set: {len(test_df)} bars from {test_df.index[0]} to {test_df.index[-1]}")
    
    # Optimize on training set
    print("\nOptimizing parameters on training set...")
    
    # Determine parameter ranges based on data frequency
    # For 1-minute data, use smaller windows than for daily data
    avg_bars_per_day = len(train_df) / (train_df.index[-1] - train_df.index[0]).days
    
    if avg_bars_per_day > 100:  # Likely intraday data
        print(f"Detected high-frequency data (~{avg_bars_per_day:.1f} bars per day)")
        fast_range = list(range(5, 31, 5))  # 5, 10, 15, ..., 30
        slow_range = list(range(20, 121, 20))  # 20, 40, 60, ..., 120
        print(f"Using intraday parameter ranges: Fast MA {fast_range[0]}-{fast_range[-1]}, Slow MA {slow_range[0]}-{slow_range[-1]}")
    else:  # Likely daily data
        print(f"Detected lower-frequency data (~{avg_bars_per_day:.1f} bars per day)")
        fast_range = list(range(5, 51, 5))  # 5, 10, 15, ..., 50
        slow_range = list(range(10, 201, 10))  # 10, 20, 30, ..., 200
        print(f"Using daily parameter ranges: Fast MA {fast_range[0]}-{fast_range[-1]}, Slow MA {slow_range[0]}-{slow_range[-1]}")
    
    best_fast, best_slow, best_sharpe = optimize_parameters(train_df, fast_range, slow_range)
    
    print(f"\nBest parameters found: Fast MA = {best_fast}, Slow MA = {best_slow} (Sharpe = {best_sharpe:.2f})")
    
    # Backtest with optimized parameters on training set
    print("\nRunning backtest on training set with optimized parameters...")
    strategy = MAStrategy(best_fast, best_slow)
    train_results = strategy.backtest(train_df)
    train_trades = strategy.get_trades()
    train_metrics = calculate_metrics(train_results['equity'], train_trades)
    
    # Backtest with optimized parameters on test set
    print("\nRunning backtest on test set with optimized parameters...")
    strategy = MAStrategy(best_fast, best_slow)
    test_results = strategy.backtest(test_df)
    test_trades = strategy.get_trades()
    test_metrics = calculate_metrics(test_results['equity'], test_trades)
    
    # Print performance metrics
    print_metrics(train_metrics, "Training Set Performance")
    print_metrics(test_metrics, "Test Set Performance")
    
    # Plot results
    plot_backtest_results(train_results, "Training Set")
    plot_backtest_results(test_results, "Test Set")
    
    print("\nBacktest completed successfully!")
