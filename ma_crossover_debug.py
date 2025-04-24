"""
Moving Average Crossover Strategy Debugging

This script helps diagnose issues with MA Crossover strategy optimization
by providing detailed logging and running tests with specific parameters.
"""
import logging
import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from typing import Dict, Any, List

# Import necessary components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler
from src.core.events.event_emitters import BarEmitter
from src.strategy.strategies.ma_crossover import MovingAverageCrossoverStrategy
from src.execution.backtest.backtest import run_backtest
from src.analytics.performance import PerformanceAnalytics

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_data_handler(data_path):
    """
    Set up data handler with CSV data.
    
    Args:
        data_path: Path to CSV data file
        
    Returns:
        tuple: (data_handler, symbol)
    """
    # Check if data file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    # Set up data source
    data_dir = os.path.dirname(data_path)
    filename = os.path.basename(data_path)
    
    # Extract symbol from filename
    symbol = filename.split('_')[0]  # Assumes format SYMBOL_timeframe.csv
    
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
        raise ValueError(f"Failed to load data for {symbol}")
    
    # Log data summary
    df = data_handler.data_frames[symbol]
    logger.info(f"Loaded {len(df)} bars for {symbol} from {df.index.min()} to {df.index.max()}")
    logger.info(f"Data frequency: {get_data_frequency(df)}")
    
    return data_handler, symbol

def get_data_frequency(df):
    """Determine approximate data frequency."""
    if len(df) < 2:
        return "Unknown (insufficient data)"
        
    # Calculate average time difference between bars
    if isinstance(df.index, pd.DatetimeIndex):
        diff = pd.Series(df.index).diff().median()
        
        # Convert to human-readable format
        if diff < pd.Timedelta(minutes=1):
            return f"{diff.total_seconds():.1f} seconds"
        elif diff < pd.Timedelta(hours=1):
            return f"{diff.total_seconds()/60:.1f} minutes"
        elif diff < pd.Timedelta(days=1):
            return f"{diff.total_seconds()/3600:.1f} hours"
        else:
            return f"{diff.total_seconds()/(3600*24):.1f} days"
    else:
        return "Unknown (index not datetime)"

def debug_single_backtest(data_handler, symbol, fast_window, slow_window, initial_cash=10000.0):
    """
    Run a detailed backtest with specific parameters.
    
    Args:
        data_handler: Historical data handler
        symbol: Symbol to trade
        fast_window: Fast MA window size
        slow_window: Slow MA window size
        initial_cash: Initial portfolio cash
        
    Returns:
        dict: Backtest results
    """
    logger.info(f"========= DETAILED BACKTEST =========")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Fast window: {fast_window}")
    logger.info(f"Slow window: {slow_window}")
    
    # Create a verbose event tracker to log detailed events
    class VerboseTracker:
        def __init__(self):
            self.events = {
                'bar': 0,
                'signal': 0,
                'order': 0,
                'fill': 0
            }
            self.signals = []
            self.orders = []
            self.fills = []
            
        def track_event(self, event):
            event_type = event.get_type()
            
            if event_type == EventType.BAR:
                self.events['bar'] += 1
                if self.events['bar'] % 100 == 0:
                    logger.debug(f"Processed {self.events['bar']} bars")
                    
            elif event_type == EventType.SIGNAL:
                self.events['signal'] += 1
                signal_value = event.get_signal_value()
                direction = "BUY" if signal_value == 1 else "SELL" if signal_value == -1 else "NEUTRAL"
                timestamp = event.get_timestamp()
                price = event.get_price()
                
                signal_info = {
                    'timestamp': timestamp,
                    'direction': direction,
                    'price': price
                }
                self.signals.append(signal_info)
                
                logger.info(f"SIGNAL: {direction} {symbol} @ {price:.2f} [{timestamp}]")
                
            elif event_type == EventType.ORDER:
                self.events['order'] += 1
                direction = event.get_direction()
                quantity = event.get_quantity()
                price = event.get_price()
                timestamp = event.get_timestamp()
                
                order_info = {
                    'timestamp': timestamp,
                    'direction': direction,
                    'quantity': quantity,
                    'price': price
                }
                self.orders.append(order_info)
                
                logger.info(f"ORDER: {direction} {quantity} {symbol} @ {price:.2f} [{timestamp}]")
                
            elif event_type == EventType.FILL:
                self.events['fill'] += 1
                direction = event.get_direction()
                quantity = event.get_quantity()
                price = event.get_price()
                timestamp = event.get_timestamp()
                
                fill_info = {
                    'timestamp': timestamp,
                    'direction': direction,
                    'quantity': quantity,
                    'price': price
                }
                self.fills.append(fill_info)
                
                logger.info(f"FILL: {direction} {quantity} {symbol} @ {price:.2f} [{timestamp}]")
    
    # Create tracker instance
    tracker = VerboseTracker()
    
    # Create strategy
    strategy = MovingAverageCrossoverStrategy(
        name=f"ma_crossover_debug",
        symbols=[symbol],
        fast_window=fast_window,
        slow_window=slow_window
    )
    
    # Create event bus
    event_bus = EventBus()
    
    # Register event handlers
    event_bus.register(EventType.BAR, tracker.track_event)
    event_bus.register(EventType.SIGNAL, tracker.track_event)
    event_bus.register(EventType.ORDER, tracker.track_event)
    event_bus.register(EventType.FILL, tracker.track_event)
    
    # Set event bus on strategy
    strategy.set_event_bus(event_bus)
    
    # Reset components
    strategy.reset()
    data_handler.reset()
    
    try:
        # Run backtest
        logger.info("Starting backtest...")
        equity_curve, trades = run_backtest(
            component=strategy,
            data_handler=data_handler,
            initial_cash=initial_cash
        )
        
        # Calculate metrics
        metrics = PerformanceAnalytics.calculate_metrics(equity_curve, trades)
        
        # Log results
        logger.info("Backtest complete")
        logger.info(f"Total bars processed: {tracker.events['bar']}")
        logger.info(f"Total signals generated: {tracker.events['signal']}")
        logger.info(f"Total orders placed: {tracker.events['order']}")
        logger.info(f"Total fills executed: {tracker.events['fill']}")
        
        # Log performance metrics
        logger.info("Performance Metrics:")
        logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}")
        logger.info(f"Total Return: {metrics.get('total_return', 'N/A')}%")
        logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 'N/A')}%")
        logger.info(f"Win Rate: {metrics.get('win_rate', 'N/A')}%")
        
        # Create detailed results with all tracking information
        results = {
            'metrics': metrics,
            'trades': trades,
            'signals': tracker.signals,
            'orders': tracker.orders,
            'fills': tracker.fills,
            'event_counts': tracker.events
        }
        
        return results
        
    except Exception as e:
        logger.exception(f"Error during backtest: {e}")
        return {
            'error': str(e),
            'event_counts': tracker.events
        }

def analyze_strategy_behavior(data_handler, symbol):
    """
    Analyze MA Crossover strategy behavior with different parameter sets.
    
    Args:
        data_handler: Historical data handler
        symbol: Symbol to analyze
    """
    logger.info("=== Strategy Behavior Analysis ===")
    
    # Test sets of parameters
    parameter_sets = [
        (5, 10),    # Small windows
        (10, 20),   # Medium windows
        (20, 50),   # Large windows
        (50, 200)   # Very large windows
    ]
    
    results = []
    
    for fast, slow in parameter_sets:
        logger.info(f"Testing parameters: Fast={fast}, Slow={slow}")
        
        # Create strategy
        strategy = MovingAverageCrossoverStrategy(
            name=f"ma_crossover_{fast}_{slow}",
            symbols=[symbol],
            fast_window=fast,
            slow_window=slow
        )
        
        # Reset components
        strategy.reset()
        data_handler.reset()
        
        try:
            # Run backtest
            equity_curve, trades = run_backtest(
                component=strategy,
                data_handler=data_handler
            )
            
            # Calculate metrics
            metrics = PerformanceAnalytics.calculate_metrics(equity_curve, trades)
            
            # Add parameter info
            result = {
                'fast_window': fast,
                'slow_window': slow,
                'trades_count': len(trades),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'total_return': metrics.get('total_return', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'win_rate': metrics.get('win_rate', 0)
            }
            
            logger.info(f"Results: Trades={result['trades_count']}, "
                       f"Sharpe={result['sharpe_ratio']:.4f}, "
                       f"Return={result['total_return']:.2f}%")
            
            results.append(result)
            
        except Exception as e:
            logger.exception(f"Error testing parameters ({fast}, {slow}): {e}")
    
    # Create comparison table
    logger.info("\nParameter Comparison:")
    logger.info("| Fast | Slow | Trades | Sharpe  | Return % | Max DD % | Win % |")
    logger.info("|------|------|--------|---------|----------|----------|-------|")
    
    for r in results:
        logger.info(f"| {r['fast_window']:4d} | {r['slow_window']:4d} | {r['trades_count']:6d} | "
                   f"{r['sharpe_ratio']:7.4f} | {r['total_return']:8.2f} | "
                   f"{r['max_drawdown']:8.2f} | {r['win_rate']:5.2f} |")
    
    return results

def inspect_data_statistics(data_handler, symbol):
    """
    Analyze the dataset to determine its suitability for different MA window sizes.
    
    Args:
        data_handler: Historical data handler
        symbol: Symbol to analyze
    """
    logger.info("=== Data Statistics Analysis ===")
    
    # Get the dataframe
    df = data_handler.data_frames[symbol]
    
    # Basic statistics
    logger.info(f"Dataset size: {len(df)} bars")
    
    if isinstance(df.index, pd.DatetimeIndex):
        start_date = df.index.min()
        end_date = df.index.max()
        date_range = end_date - start_date
        
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Total duration: {date_range}")
        
        # Calculate average bar frequency
        avg_diff = df.index.to_series().diff().mean()
        logger.info(f"Average bar interval: {avg_diff}")
    
    # Price statistics
    logger.info(f"Average price: {df['close'].mean():.2f}")
    logger.info(f"Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")
    logger.info(f"Price volatility (std): {df['close'].std():.2f}")
    
    # Check for missing data
    missing_data = df.isna().sum()
    if missing_data.sum() > 0:
        logger.warning(f"Missing data detected:\n{missing_data}")
    
    # Calculate returns
    df['returns'] = df['close'].pct_change() * 100
    logger.info(f"Average daily return: {df['returns'].mean():.4f}%")
    logger.info(f"Return volatility: {df['returns'].std():.4f}%")
    
    # Calculate moving averages to check crossovers
    window_pairs = [(5, 10), (10, 20), (20, 50), (50, 200)]
    
    for fast, slow in window_pairs:
        # Ensure we have enough data for the slow window
        if len(df) <= slow:
            logger.warning(f"Dataset too small for {fast}/{slow} MA pair ({len(df)} bars < {slow})")
            continue
            
        # Calculate moving averages
        df[f'ma_{fast}'] = df['close'].rolling(window=fast).mean()
        df[f'ma_{slow}'] = df['close'].rolling(window=slow).mean()
        
        # Skip initial NaN values
        valid_df = df.dropna()
        
        # Detect crossovers
        crossovers = 0
        for i in range(1, len(valid_df)):
            # Previous bar relationship
            prev_fast = valid_df[f'ma_{fast}'].iloc[i-1]
            prev_slow = valid_df[f'ma_{slow}'].iloc[i-1]
            prev_relationship = prev_fast > prev_slow
            
            # Current bar relationship
            curr_fast = valid_df[f'ma_{fast}'].iloc[i]
            curr_slow = valid_df[f'ma_{slow}'].iloc[i]
            curr_relationship = curr_fast > curr_slow
            
            # Check for crossover
            if prev_relationship != curr_relationship:
                crossovers += 1
        
        logger.info(f"MA {fast}/{slow}: {crossovers} crossovers detected")
    
    return df

def plot_moving_averages(data_handler, symbol, fast_window, slow_window, max_bars=500):
    """
    Plot price with moving averages to visualize crossovers.
    
    Args:
        data_handler: Historical data handler
        symbol: Symbol to plot
        fast_window: Fast MA window size
        slow_window: Slow MA window size
        max_bars: Maximum bars to plot
    """
    # Get dataframe
    df = data_handler.data_frames[symbol].copy()
    
    # Calculate moving averages
    df[f'MA_Fast'] = df['close'].rolling(window=fast_window).mean()
    df[f'MA_Slow'] = df['close'].rolling(window=slow_window).mean()
    
    # Remove NaN values
    df = df.dropna()
    
    # Limit number of bars for better visualization
    if len(df) > max_bars:
        df = df.iloc[-max_bars:]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot price and MAs
    ax.plot(df.index, df['close'], label='Price', alpha=0.7)
    ax.plot(df.index, df[f'MA_Fast'], label=f'Fast MA ({fast_window})', linewidth=2)
    ax.plot(df.index, df[f'MA_Slow'], label=f'Slow MA ({slow_window})', linewidth=2)
    
    # Detect crossovers
    buy_signals = []
    sell_signals = []
    
    for i in range(1, len(df)):
        # Previous bar relationship
        prev_fast = df[f'MA_Fast'].iloc[i-1]
        prev_slow = df[f'MA_Slow'].iloc[i-1]
        prev_relationship = prev_fast > prev_slow
        
        # Current bar relationship
        curr_fast = df[f'MA_Fast'].iloc[i]
        curr_slow = df[f'MA_Slow'].iloc[i]
        curr_relationship = curr_fast > curr_slow
        
        # Check for crossover
        if not prev_relationship and curr_relationship:
            # Bullish crossover
            buy_signals.append((df.index[i], df['close'].iloc[i]))
        elif prev_relationship and not curr_relationship:
            # Bearish crossover
            sell_signals.append((df.index[i], df['close'].iloc[i]))
    
    # Plot signals
    if buy_signals:
        buy_x, buy_y = zip(*buy_signals)
        ax.scatter(buy_x, buy_y, marker='^', color='green', s=100, label='Buy Signal')
    
    if sell_signals:
        sell_x, sell_y = zip(*sell_signals)
        ax.scatter(sell_x, sell_y, marker='v', color='red', s=100, label='Sell Signal')
    
    # Set labels and title
    ax.set_title(f'{symbol}: Price with Moving Averages ({fast_window}/{slow_window})', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Format x-axis for dates
    if isinstance(df.index, pd.DatetimeIndex):
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = f'{symbol}_MA_{fast_window}_{slow_window}.png'
    plt.savefig(plot_path)
    logger.info(f"Plot saved to {plot_path}")
    
    # Log signal statistics
    logger.info(f"MA {fast_window}/{slow_window}: {len(buy_signals)} buy signals, {len(sell_signals)} sell signals")
    
    return fig

def run_diagnostics(data_path):
    """
    Run comprehensive diagnostics on the MA crossover strategy.
    
    Args:
        data_path: Path to data file
    """
    logger.info("=== STARTING MA CROSSOVER STRATEGY DIAGNOSTICS ===")
    
    # Setup data handler
    data_handler, symbol = setup_data_handler(data_path)
    
    # Inspect data statistics
    logger.info("\n--- Data Statistics Analysis ---")
    df_stats = inspect_data_statistics(data_handler, symbol)
    
    # Analyze strategy behavior with different parameters
    logger.info("\n--- Strategy Behavior Analysis ---")
    behavior_results = analyze_strategy_behavior(data_handler, symbol)
    
    # Plot representative parameter set
    logger.info("\n--- Moving Average Visualization ---")
    if len(behavior_results) > 0:
        # Find parameter set with most trades
        best_params = max(behavior_results, key=lambda x: x['trades_count'])
        fast, slow = best_params['fast_window'], best_params['slow_window']
        
        # If no trades were generated, try a small parameter set
        if best_params['trades_count'] == 0:
            fast, slow = 5, 10
            
        plot_moving_averages(data_handler, symbol, fast, slow)
    else:
        # Default parameters
        plot_moving_averages(data_handler, symbol, 10, 20)
    
    # Debug single backtest with logging
    logger.info("\n--- Detailed Single Backtest Analysis ---")
    # Use a parameter set that's likely to produce trades
    debug_results = debug_single_backtest(data_handler, symbol, 5, 10)
    
    logger.info("=== DIAGNOSTICS COMPLETE ===")
    
    # Return all results for potential further analysis
    return {
        'data_handler': data_handler,
        'symbol': symbol,
        'behavior_results': behavior_results,
        'debug_results': debug_results
    }

if __name__ == "__main__":
    # Use command line arguments if provided
    import argparse
    
    parser = argparse.ArgumentParser(description='MA Crossover Strategy Diagnostics')
    parser.add_argument('--data', type=str, required=True, help='Path to price data CSV file')
    
    args = parser.parse_args()
    
    # Run diagnostics
    run_diagnostics(args.data)
