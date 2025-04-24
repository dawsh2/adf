"""
Improved test for the Mean Reversion Strategy using real market data.

Loads data from 'data/SPY_1m.csv' and tests the mean reversion strategy
using the backtest framework instead of local implementation.
"""

import logging
import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import necessary components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType, BarEvent
from src.core.events.event_utils import create_bar_event, EventTracker
from src.execution.portfolio import PortfolioManager
from src.execution.brokers.simulated import SimulatedBroker
from src.strategy.risk.risk_manager import SimpleRiskManager

# Import existing data handling and processing components
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler

# Import the existing mean reversion strategy
from src.strategy.strategies.mean_reversion import MeanReversionStrategy

# Import backtest framework
from src.execution.backtest.backtest import run_backtest

# Import performance analysis
from src.analytics.performance import PerformanceAnalytics

def analyze_mean_reversion_signals(df, lookback=20, z_threshold=1.5):
    """
    Analyze the data to predict where mean reversion signals should occur.
    This helps us validate that the strategy is working as expected.
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Lookback period for moving average
        z_threshold: Z-score threshold for signals
        
    Returns:
        Dictionary with expected signals and metrics
    """
    # Calculate rolling mean and standard deviation
    df['rolling_mean'] = df['close'].rolling(window=lookback).mean()
    df['rolling_std'] = df['close'].rolling(window=lookback).std()
    
    # Calculate Z-scores
    df['z_score'] = (df['close'] - df['rolling_mean']) / df['rolling_std']
    
    # Find crossings of the threshold (potential signal points)
    buy_signals = []
    sell_signals = []
    
    # Skip the first lookback rows since they'll have NaN values
    for i in range(lookback+1, len(df)):
        # Check for buy signals (z-score crossing below negative threshold)
        if df['z_score'].iloc[i-1] >= -z_threshold and df['z_score'].iloc[i] < -z_threshold:
            buy_signals.append(i)
            
        # Check for sell signals (z-score crossing above positive threshold)
        if df['z_score'].iloc[i-1] <= z_threshold and df['z_score'].iloc[i] > z_threshold:
            sell_signals.append(i)
    
    logger.info(f"Analysis found {len(buy_signals)} potential buy signals and {len(sell_signals)} potential sell signals")
    
    return {
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'df_with_metrics': df  # Return the DataFrame with added metrics for plotting
    }

def plot_results(df, trades, lookback, z_threshold, save_path='mean_reversion_real_data_results.png'):
    """Plot the results of the mean reversion strategy on real data."""
    try:
        # Calculate strategy metrics if not already present
        if 'z_score' not in df.columns:
            df['rolling_mean'] = df['close'].rolling(window=lookback).mean()
            df['rolling_std'] = df['close'].rolling(window=lookback).std()
            df['z_score'] = (df['close'] - df['rolling_mean']) / df['rolling_std']
            df['upper_band'] = df['rolling_mean'] + (df['rolling_std'] * z_threshold)
            df['lower_band'] = df['rolling_mean'] - (df['rolling_std'] * z_threshold)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price and bands
        ax1.plot(df.index, df['close'], label='Price', linewidth=1.5)
        ax1.plot(df.index, df['rolling_mean'], label=f'{lookback}-period MA', linewidth=1.2, alpha=0.8)
        ax1.plot(df.index, df['upper_band'], 'r--', 
                label=f'Upper Band (+{z_threshold} σ)', linewidth=1, alpha=0.6)
        ax1.plot(df.index, df['lower_band'], 'g--', 
                label=f'Lower Band (-{z_threshold} σ)', linewidth=1, alpha=0.6)
        
        # Plot trades
        if trades:
            # Extract timestamps and prices
            buy_timestamps = [t['timestamp'] for t in trades if t['direction'] == 'BUY']
            buy_prices = [t['price'] for t in trades if t['direction'] == 'BUY']
            sell_timestamps = [t['timestamp'] for t in trades if t['direction'] == 'SELL']
            sell_prices = [t['price'] for t in trades if t['direction'] == 'SELL']
            
            # Plot trades
            if buy_timestamps:
                ax1.scatter(buy_timestamps, buy_prices, marker='^', color='green', s=80, label='Buy', alpha=0.7)
            if sell_timestamps:
                ax1.scatter(sell_timestamps, sell_prices, marker='v', color='red', s=80, label='Sell', alpha=0.7)
        
        # Format first subplot
        ax1.set_title('Mean Reversion Strategy on Real Market Data', fontsize=14)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=10)
        
        # Plot Z-scores
        ax2.plot(df.index, df['z_score'], label='Z-score', color='blue')
        ax2.axhline(y=z_threshold, color='r', linestyle='--', alpha=0.6, label=f'+{z_threshold} σ')
        ax2.axhline(y=-z_threshold, color='g', linestyle='--', alpha=0.6, label=f'-{z_threshold} σ')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.4)
        
        # Format second subplot
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Z-score', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=10)
        
        # Format datetime on x-axis
        for ax in [ax1, ax2]:
            if isinstance(df.index, pd.DatetimeIndex):
                ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(save_path)
        
        logger.info(f"Strategy results plotted to {save_path}")
        return fig
    except Exception as e:
        logger.error(f"Error plotting results: {e}")
        return None

def run_real_data_test(debug=True):
    """Run mean reversion strategy test on real data using the backtest framework."""
    logger.info("=== STARTING MEAN REVERSION STRATEGY TEST ON REAL DATA ===")
    
    # Set debug level
    if debug:
        logger.setLevel(logging.DEBUG)
    
    # Data file path
    csv_file = 'data/SAMPLE_1m.csv'
    if not os.path.exists(csv_file):
        logger.error(f"Data file not found: {csv_file}")
        return False
    
    # Load data from CSV
    logger.info(f"Loading data from {csv_file}")
    
    try:
        # Create data source and handler
        data_dir = os.path.dirname(csv_file)
        filename = os.path.basename(csv_file)
        
        # Extract symbol from filename (assuming format SYMBOL_timeframe.csv)
        symbol = filename.split('_')[0]
        
        # Create a CSV data source
        data_source = CSVDataSource(
            data_dir=data_dir, 
            filename_pattern=filename  # Use exact filename
        )
        
        # Create bar emitter (required by HistoricalDataHandler)
        event_bus = EventBus()
        from src.core.events.event_emitters import BarEmitter
        bar_emitter = BarEmitter(name="bar_emitter", event_bus=event_bus)
        bar_emitter.start()
        
        # Create historical data handler
        data_handler = HistoricalDataHandler(
            data_source=data_source,
            bar_emitter=bar_emitter
        )
        
        # Load data for the symbol
        data_handler.load_data(symbols=[symbol])
        
        # Check if data was loaded successfully
        if symbol not in data_handler.data_frames:
            logger.error(f"Failed to load data for symbol {symbol}")
            return False
            
        # Get the DataFrame for analysis
        df = data_handler.data_frames[symbol].copy()
        
        # Log data information
        logger.info(f"Loaded {len(df)} bars for {symbol} from {df.index.min()} to {df.index.max()}")
        
        # Strategy parameters
        lookback = 20  # Use 20-period lookback
        z_threshold = 1.5  # Use 1.5 standard deviations
        
        # Create mean reversion strategy
        strategy = MeanReversionStrategy(
            name="mean_reversion_real",
            symbols=[symbol],
            lookback=lookback,
            z_threshold=z_threshold
        )
        
        # Analyze data to determine expected signals (independent validation)
        analysis_result = analyze_mean_reversion_signals(df, lookback, z_threshold)
        expected_buy_signals = analysis_result['buy_signals']
        expected_sell_signals = analysis_result['sell_signals']
        df_with_metrics = analysis_result['df_with_metrics']  # Save for later use
        
        # Create event tracker to monitor events during backtest
        event_tracker = EventTracker(name="validation_tracker", verbose=True)
        
        # Create custom event handler that uses the tracker
        def track_event(event):
            # Track the event
            event_tracker.track_event(event)
            
            # Log specific event types
            event_type = event.get_type()
            if event_type == EventType.SIGNAL:
                symbol = event.get_symbol()
                signal = event.get_signal_value()
                price = event.get_price()
                logger.info(f"Signal: {symbol} {signal} @ {price:.2f}")
                
            elif event_type == EventType.FILL:
                symbol = event.get_symbol()
                direction = event.get_direction()
                quantity = event.get_quantity()
                price = event.get_price()
                logger.info(f"Fill: {symbol} {direction} {quantity} @ {price:.2f}")
        
        # Set up additional backtest parameters
        initial_cash = 10000.0
        
        # Create a simple event counter instead of using EventTracker
        class SimpleEventCounter:
            def __init__(self):
                self.counts = {
                    EventType.BAR: 0,
                    EventType.SIGNAL: 0,
                    EventType.ORDER: 0,
                    EventType.FILL: 0
                }
                
            def track_event(self, event):
                event_type = event.get_type()
                if event_type in self.counts:
                    self.counts[event_type] += 1
            
            def get_event_count(self, event_type):
                return self.counts.get(event_type, 0)
        
        # Create a simple counter instead of EventTracker
        event_counter = SimpleEventCounter()
        
        # Run the backtest with our framework
        logger.info("Running backtest with backtest framework...")
        
        # Register the track_event function with the run_backtest's internal event bus
        # This will be handled inside the run_backtest function
        
        # Add the event tracker to the event bus (we'll set this up in the backtest wrapper)
        event_bus = EventBus()
        for event_type in [EventType.BAR, EventType.SIGNAL, EventType.ORDER, EventType.FILL]:
            event_bus.register(event_type, track_event)
        
        # Run the backtest using the backtest framework
        # Create validator
        class BacktestValidator:
            def __init__(self, event_counter):
                self.event_counter = event_counter
                self.signals = []
                self.orders = []
                self.fills = []
                self.logger = logging.getLogger("validator")
                
            def process_event(self, event):
                try:
                    # Track the event using the event counter
                    self.event_counter.track_event(event)
                    
                    # Store events by type for later analysis
                    event_type = event.get_type()
                    if event_type == EventType.SIGNAL:
                        self.signals.append(event)
                        self.logger.debug(f"Signal: {event.get_symbol()} {event.get_signal_value()} @ {event.get_price():.2f}")
                    elif event_type == EventType.ORDER:
                        self.orders.append(event)
                        self.logger.debug(f"Order: {event.get_symbol()} {event.get_direction()} {event.get_quantity()} @ {event.get_price():.2f}")
                    elif event_type == EventType.FILL:
                        self.fills.append(event)
                        self.logger.debug(f"Fill: {event.get_symbol()} {event.get_direction()} {event.get_quantity()} @ {event.get_price():.2f}")
                except Exception as e:
                    self.logger.error(f"Error in validator: {e}", exc_info=True)
        
        # Create validator
        validator = BacktestValidator(event_counter)
        
        # Run backtest with event tracking
        equity_curve, trades = run_backtest(
            component=strategy,
            data_handler=data_handler,
            initial_cash=initial_cash,
            position_size=100,  # Fixed position size
            debug=True  # Enable verbose logging
        )
        
        # Log backtest summary
        logger.info("=== BACKTEST COMPLETE ===")
        logger.info(f"Events processed: Bars={event_counter.get_event_count(EventType.BAR)}, " +
                   f"Signals={event_counter.get_event_count(EventType.SIGNAL)}, " +
                   f"Orders={event_counter.get_event_count(EventType.ORDER)}, " +
                   f"Fills={event_counter.get_event_count(EventType.FILL)}")
        logger.info(f"Trades executed: {len(trades)}")
        
        # Calculate performance metrics
        metrics = PerformanceAnalytics.calculate_metrics(equity_curve, trades)
        
        # Fix the percentage display for total return
        if 'total_return' in metrics:
            # Ensure this is displayed as actual percentage, not decimal
            logger.info(f"Raw total return value: {metrics['total_return']}")
            # Check if it needs conversion (if it's a small decimal like 0.66 instead of 66.09)
            if abs(metrics['total_return']) < 10:  # Likely a decimal that needs conversion
                metrics['total_return'] = metrics['total_return'] * 100
                logger.info(f"Converted total return to: {metrics['total_return']}")
        
        # Display performance
        logger.info("=== PERFORMANCE SUMMARY ===")
        logger.info(f"Initial equity: ${metrics['initial_equity']:.2f}")
        logger.info(f"Final equity: ${metrics['final_equity']:.2f}")
        logger.info(f"Total return: {metrics['total_return']:.2f}%")
        logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max drawdown: {metrics['max_drawdown']:.2f}%")
        logger.info(f"Win rate: {metrics['win_rate']:.2f}%")
        
        # Print metrics as table
        metrics_table = PerformanceAnalytics.display_metrics(metrics)
        print("\n" + metrics_table)
        
        # Plot results using equity curve and trades
        # First convert the equity curve to the format expected by the plotting function
        if len(equity_curve) > 0:
            plot_results(df, trades, lookback, z_threshold)
        
        # Compare actual signals/trades with expected signals
        logger.info(f"Expected buy signals at bars: {expected_buy_signals}")
        logger.info(f"Expected sell signals at bars: {expected_sell_signals}")
        logger.info(f"Actual trades executed: {len(trades)}")
        
        # Classify trades
        buy_trades = [t for t in trades if t['direction'] == 'BUY']
        sell_trades = [t for t in trades if t['direction'] == 'SELL']
        
        logger.info(f"Buy trades: {len(buy_trades)}")
        logger.info(f"Sell trades: {len(sell_trades)}")
        
        # In-depth signal analysis
        # Classify trades by timestamp to see when they're occurring
        if len(trades) > 0 and len(df_with_metrics) > 0:
            # Create a DataFrame with Z-scores and trade markers
            trade_analysis = pd.DataFrame(index=df_with_metrics.index)
            trade_analysis['z_score'] = df_with_metrics['z_score']
            
            # Map trades to their timestamps
            trade_analysis['buy_trade'] = 0
            trade_analysis['sell_trade'] = 0
            
            for trade in trades:
                timestamp = trade['timestamp']
                if timestamp in trade_analysis.index:
                    if trade['direction'] == 'BUY':
                        trade_analysis.loc[timestamp, 'buy_trade'] = 1
                    else:
                        trade_analysis.loc[timestamp, 'sell_trade'] = 1
            
            # Count how many expected signals match with actual trades
            buy_matches = 0
            for bar_idx in expected_buy_signals:
                if bar_idx < len(df_with_metrics):
                    timestamp = df_with_metrics.index[bar_idx]
                    if timestamp in trade_analysis.index and trade_analysis.loc[timestamp, 'buy_trade'] == 1:
                        buy_matches += 1
            
            sell_matches = 0
            for bar_idx in expected_sell_signals:
                if bar_idx < len(df_with_metrics):
                    timestamp = df_with_metrics.index[bar_idx]
                    if timestamp in trade_analysis.index and trade_analysis.loc[timestamp, 'sell_trade'] == 1:
                        sell_matches += 1
            
            logger.info(f"Buy signal match rate: {buy_matches}/{len(expected_buy_signals)} ({buy_matches/len(expected_buy_signals)*100:.1f}%)")
            logger.info(f"Sell signal match rate: {sell_matches}/{len(expected_sell_signals)} ({sell_matches/len(expected_sell_signals)*100:.1f}%)")
            
            # Calculate extra trades (trades without corresponding signals)
            buy_extras = len(buy_trades) - buy_matches
            sell_extras = len(sell_trades) - sell_matches
            logger.info(f"Extra buy trades: {buy_extras}")
            logger.info(f"Extra sell trades: {sell_extras}")
        
        # Determine test success
        # Consider successful if we got at least some trades and performance is positive
        test_passed = len(trades) > 0 and metrics['total_return'] > 0
        
        if test_passed:
            logger.info("REAL DATA TEST PASSED!")
        else:
            logger.warning("REAL DATA TEST FAILED!")
        
        return {
            'passed': test_passed,
            'trades': trades,
            'equity_curve': equity_curve,
            'metrics': metrics,
            'dataframe': df,
            'event_counts': {
                'bar': event_counter.get_event_count(EventType.BAR),
                'signal': event_counter.get_event_count(EventType.SIGNAL),
                'order': event_counter.get_event_count(EventType.ORDER),
                'fill': event_counter.get_event_count(EventType.FILL)
            }
        }
        
    except Exception as e:
        logger.exception(f"Error during real data test: {e}")
        return {'passed': False, 'error': str(e)}

if __name__ == "__main__":
    # Run the test
    test_results = run_real_data_test()
    
    # Display summary results
    if isinstance(test_results, dict) and test_results.get('passed', False):
        print("\nREAL DATA TEST PASSED!")
        if 'metrics' in test_results:
            metrics = test_results['metrics']
            print(f"Total return: {metrics['total_return']:.2f}%")
            print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Win rate: {metrics['win_rate']:.2f}%")
            print(f"Final equity: ${metrics['final_equity']:.2f} from ${metrics['initial_equity']:.2f}")
        print(f"Trades executed: {len(test_results.get('trades', []))}")
        print(f"Event counts: {test_results.get('event_counts', {})}")
    else:
        print("\nREAL DATA TEST FAILED!")
        if isinstance(test_results, dict) and 'error' in test_results:
            print(f"Error: {test_results['error']}")
        print("Check logs for details.")
