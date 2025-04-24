"""
Simplified test script for the Mean Reversion Strategy that ensures
direct 1:1 mapping between signals and trades.
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
from src.core.events.event_types import EventType, BarEvent, SignalEvent
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler
from src.strategy.strategies.mean_reversion import MeanReversionStrategy
from src.analytics.performance import PerformanceAnalytics

# Import our simplified backtest and risk manager
# Uncomment these and adjust paths once you've placed the files
from simple import run_simplified_backtest, SimpleEventTracker
from src.strategy.risk.risk_manager import SimplePassthroughRiskManager

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
    
    # Add bands for visualization
    df['upper_band'] = df['rolling_mean'] + (df['rolling_std'] * z_threshold)
    df['lower_band'] = df['rolling_mean'] - (df['rolling_std'] * z_threshold)
    
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
        'df_with_metrics': df  # Return the DataFrame with added metrics
    }

def plot_results(df, trades, lookback, z_threshold, save_path='simplified_mean_reversion_results.png'):
    """Plot the results of the mean reversion strategy with simplified testing."""
    try:
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
        ax1.set_title('Mean Reversion Strategy (Simplified Implementation)', fontsize=14)
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

def run_simplified_mean_reversion_test(debug=True):
    """Run the mean reversion strategy test with simplified trade execution."""
    logger.info("=== STARTING SIMPLIFIED MEAN REVERSION TEST ===")
    
    # Set debug level
    if debug:
        logger.setLevel(logging.DEBUG)
    
    # Data file path
    csv_file = 'data/SAMPLE_1m.csv'
    if not os.path.exists(csv_file):
        logger.error(f"Data file not found: {csv_file}")
        return False
    
    try:
        # Create data source and handler
        data_dir = os.path.dirname(csv_file)
        filename = os.path.basename(csv_file)
        
        # Extract symbol from filename
        symbol = filename.split('_')[0]
        
        # Create data source
        data_source = CSVDataSource(
            data_dir=data_dir, 
            filename_pattern=filename
        )
        
        # Create bar emitter for data handler
        event_bus = EventBus()
        from src.core.events.event_emitters import BarEmitter
        bar_emitter = BarEmitter(name="bar_emitter", event_bus=event_bus)
        bar_emitter.start()
        
        # Create data handler
        data_handler = HistoricalDataHandler(
            data_source=data_source,
            bar_emitter=bar_emitter
        )
        
        # Load data
        data_handler.load_data(symbols=[symbol])
        
        # Check if data loaded successfully
        if symbol not in data_handler.data_frames:
            logger.error(f"Failed to load data for symbol {symbol}")
            return False
            
        # Get DataFrame for analysis
        df = data_handler.data_frames[symbol].copy()
        
        # Log data information
        logger.info(f"Loaded {len(df)} bars for {symbol} from {df.index.min()} to {df.index.max()}")
        
        # Strategy parameters
        lookback = 20
        z_threshold = 1.5
        
        # Create the strategy
        strategy = MeanReversionStrategy(
            name="mean_reversion_simplified",
            symbols=[symbol],
            lookback=lookback,
            z_threshold=z_threshold
        )
        
        # Analyze data to determine expected signals (independent validation)
        analysis_result = analyze_mean_reversion_signals(df, lookback, z_threshold)
        expected_buy_signals = analysis_result['buy_signals']
        expected_sell_signals = analysis_result['sell_signals']
        df_with_metrics = analysis_result['df_with_metrics']
        
        # Run simplified backtest
        logger.info("Running simplified backtest...")
        equity_curve, trades, event_tracker = run_simplified_backtest(
            component=strategy,
            data_handler=data_handler,
            risk_manager_class=SimplePassthroughRiskManager,
            initial_cash=10000.0,
            debug=debug
        )
        
        # Calculate performance metrics
        metrics = PerformanceAnalytics.calculate_metrics(equity_curve, trades)
        
        # Fix percentage formatting issues if needed
        if 'total_return' in metrics and abs(metrics['total_return']) < 10:
            # Likely a decimal that needs to be converted to percentage
            metrics['total_return'] = metrics['total_return'] * 100
            logger.info(f"Fixed total return: {metrics['total_return']:.2f}%")
        
        # Display performance metrics
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
        
        # Plot results
        if df_with_metrics is not None:
            plot_results(df_with_metrics, trades, lookback, z_threshold)
        
        # Compare actual trades with expected signals
        logger.info(f"Expected buy signals at bars: {expected_buy_signals}")
        logger.info(f"Expected sell signals at bars: {expected_sell_signals}")
        logger.info(f"Total expected signals: {len(expected_buy_signals) + len(expected_sell_signals)}")
        logger.info(f"Actual trades executed: {len(trades)}")
        
        # Classify trades
        buy_trades = [t for t in trades if t['direction'] == 'BUY']
        sell_trades = [t for t in trades if t['direction'] == 'SELL']
        
        logger.info(f"Buy trades: {len(buy_trades)}")
        logger.info(f"Sell trades: {len(sell_trades)}")
        
        # Check if we got the expected number of trades
        expected_total = len(expected_buy_signals) + len(expected_sell_signals)
        if len(trades) == expected_total:
            logger.info("✓ Trade count matches expected signal count!")
        else:
            logger.warning(f"✗ Trade count ({len(trades)}) doesn't match expected signal count ({expected_total})")
        
        # Check event counts
        logger.info(f"Event counts: ")
        logger.info(f"  Bars: {event_tracker.get_event_count(EventType.BAR)}")
        logger.info(f"  Signals: {event_tracker.get_event_count(EventType.SIGNAL)}")
        logger.info(f"  Orders: {event_tracker.get_event_count(EventType.ORDER)}")
        logger.info(f"  Fills: {event_tracker.get_event_count(EventType.FILL)}")
        
        # Create a detailed signal verification
        trade_verification = []
        for i, bar_idx in enumerate(expected_buy_signals):
            if i < len(df):
                timestamp = df.index[bar_idx]
                price = df['close'].iloc[bar_idx]
                
                # Check if there's a trade at this timestamp
                matching_trades = [t for t in trades if t['timestamp'] == timestamp and t['direction'] == 'BUY']
                
                trade_verification.append({
                    'bar_idx': bar_idx,
                    'timestamp': timestamp,
                    'price': price,
                    'expected_direction': 'BUY',
                    'trade_found': len(matching_trades) > 0
                })
        
        for i, bar_idx in enumerate(expected_sell_signals):
            if i < len(df):
                timestamp = df.index[bar_idx]
                price = df['close'].iloc[bar_idx]
                
                # Check if there's a trade at this timestamp
                matching_trades = [t for t in trades if t['timestamp'] == timestamp and t['direction'] == 'SELL']
                
                trade_verification.append({
                    'bar_idx': bar_idx,
                    'timestamp': timestamp, 
                    'price': price,
                    'expected_direction': 'SELL',
                    'trade_found': len(matching_trades) > 0
                })
        
        # Get verification statistics
        matches = sum(1 for v in trade_verification if v['trade_found'])
        match_rate = matches / len(trade_verification) if trade_verification else 0
        
        logger.info(f"Signal match verification: {matches}/{len(trade_verification)} ({match_rate*100:.1f}%)")
        
        # Test is considered successful if all signals generated trades
        test_passed = match_rate == 1.0
        
        if test_passed:
            logger.info("✓ SIMPLIFIED TEST PASSED! All expected signals generated trades.")
        else:
            logger.warning(f"✗ SIMPLIFIED TEST FAILED: Only {match_rate*100:.1f}% of signals generated trades.")
        
        return {
            'passed': test_passed,
            'metrics': metrics,
            'trades': trades,
            'event_counts': {
                'bar': event_tracker.get_event_count(EventType.BAR),
                'signal': event_tracker.get_event_count(EventType.SIGNAL),
                'order': event_tracker.get_event_count(EventType.ORDER),
                'fill': event_tracker.get_event_count(EventType.FILL)
            },
            'verification': trade_verification
        }
    
    except Exception as e:
        logger.exception(f"Error during simplified test: {e}")
        return {'passed': False, 'error': str(e)}

if __name__ == "__main__":
    # Run the test
    test_results = run_simplified_mean_reversion_test()
    
    # Display summary
    if isinstance(test_results, dict) and test_results.get('passed', False):
        print("\nSIMPLIFIED TEST PASSED!")
        if 'metrics' in test_results:
            metrics = test_results['metrics']
            print(f"Total return: {metrics['total_return']:.2f}%")
            print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Win rate: {metrics['win_rate']:.2f}%")
            print(f"Final equity: ${metrics['final_equity']:.2f} from ${metrics['initial_equity']:.2f}")
        print(f"Trades executed: {len(test_results.get('trades', []))}")
        print(f"Event counts: {test_results.get('event_counts', {})}")
    else:
        print("\nSIMPLIFIED TEST FAILED!")
        if isinstance(test_results, dict) and 'error' in test_results:
            print(f"Error: {test_results['error']}")
        print("Check logs for details.")
