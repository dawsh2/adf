"""
Regime Filter Validation Script

This script validates the regime filter implementation by comparing
a standard strategy with a regime-filtered version on the same data.
"""
import logging
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any

# Import core components
from src.core.events.event_bus import EventBus
from src.core.events.event_types import SignalEvent
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler

# Import the strategies
from src.strategy.strategies.mean_reversion import MeanReversionStrategy
from src.models.filters.regime.simple import SimpleRegimeFilteredStrategy  # Our new implementation

# Import execution components
from src.execution.backtest.backtest import run_backtest
from src.analytics.performance import PerformanceAnalytics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_regime_filter(data_path, output_dir=None, initial_cash=10000.0):
    """
    Run backtest comparing a standard strategy with its regime-filtered version.
    
    Args:
        data_path: Path to price data CSV file
        output_dir: Directory for saving results
        initial_cash: Initial portfolio cash
        
    Returns:
        dict: Validation results
    """
    # Set output directory
    if output_dir is None:
        output_dir = os.getcwd()
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract symbol from filename
    filename = os.path.basename(data_path)
    symbol = filename.split('_')[0]  # Assumes format SYMBOL_timeframe.csv
    
    # Set up data handler
    data_dir = os.path.dirname(data_path)
    
    # Create data source
    data_source = CSVDataSource(
        data_dir=data_dir, 
        filename_pattern=filename  # Use exact filename
    )
    
    # Create bar emitter
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
    
    # Check if data was loaded
    if symbol not in data_handler.data_frames:
        raise ValueError(f"Failed to load data for {symbol}")
    
    # Log data summary
    df = data_handler.data_frames[symbol]
    logger.info(f"Loaded {len(df)} bars for {symbol} from {df.index.min()} to {df.index.max()}")
    
    # Set strategy parameters
    lookback = 20
    z_threshold = 1.5
    ma_window = 50  # For regime detection
    
    # Create standard mean reversion strategy
    base_strategy = MeanReversionStrategy(
        name="mean_reversion",
        symbols=[symbol],
        lookback=lookback,
        z_threshold=z_threshold
    )
    
    # Create regime-filtered version of the same strategy
    regime_strategy = SimpleRegimeFilteredStrategy(
        base_strategy=MeanReversionStrategy(
            name="mean_reversion",
            symbols=[symbol],
            lookback=lookback,
            z_threshold=z_threshold
        ),
        ma_window=ma_window
    )
    
    # Run backtest for standard strategy
    logger.info(f"Running standard Mean Reversion backtest for {symbol}...")
    base_equity_curve, base_trades = run_backtest(
        component=base_strategy,
        data_handler=data_handler,
        initial_cash=initial_cash
    )
    
    # Reset data handler for the next backtest
    data_handler.reset()
    
    # Run backtest for regime-filtered strategy
    logger.info(f"Running regime-filtered Mean Reversion backtest for {symbol}...")
    regime_equity_curve, regime_trades = run_backtest(
        component=regime_strategy,
        data_handler=data_handler,
        initial_cash=initial_cash
    )
    
    # Calculate performance metrics for both strategies
    base_metrics = PerformanceAnalytics.calculate_metrics(base_equity_curve, base_trades)
    regime_metrics = PerformanceAnalytics.calculate_metrics(regime_equity_curve, regime_trades)
    
    # Log results
    logger.info("\n===== STRATEGY COMPARISON =====")
    logger.info(f"Standard Mean Reversion Strategy:")
    logger.info(f"- Final Equity: ${base_metrics['final_equity']:.2f}")
    logger.info(f"- Total Return: {base_metrics['total_return']:.2f}%")
    logger.info(f"- Max Drawdown: {base_metrics['max_drawdown']:.2f}%")
    logger.info(f"- Trade Count: {len(base_trades)}")
    
    logger.info(f"\nRegime-Filtered Mean Reversion Strategy:")
    logger.info(f"- Final Equity: ${regime_metrics['final_equity']:.2f}")
    logger.info(f"- Total Return: {regime_metrics['total_return']:.2f}%")
    logger.info(f"- Max Drawdown: {regime_metrics['max_drawdown']:.2f}%")
    logger.info(f"- Trade Count: {len(regime_trades)}")
    
    # Get regime filter stats
    regime_stats = regime_strategy.get_regime_stats()
    logger.info(f"\nRegime Filter Stats:")
    logger.info(f"- Signals Passed: {regime_stats['passed_signals']}")
    logger.info(f"- Signals Filtered: {regime_stats['filtered_signals']}")
    logger.info(f"- Filter Rate: {regime_stats['filtered_signals'] / (regime_stats['passed_signals'] + regime_stats['filtered_signals']) * 100 if (regime_stats['passed_signals'] + regime_stats['filtered_signals']) > 0 else 0:.2f}%")
    
    # Create comparison plot
    plot_comparison(base_equity_curve, regime_equity_curve, base_trades, regime_trades, 
                  symbol, output_dir)
    
    # Determine if filtering is having a positive effect
    is_improvement = regime_metrics['total_return'] > base_metrics['total_return']
    
    # Compare Sharpe ratios if available
    if 'sharpe_ratio' in base_metrics and 'sharpe_ratio' in regime_metrics:
        sharpe_improvement = regime_metrics['sharpe_ratio'] > base_metrics['sharpe_ratio']
        logger.info(f"\nSharpe Ratio: Base={base_metrics['sharpe_ratio']:.2f}, Regime={regime_metrics['sharpe_ratio']:.2f}")
    else:
        sharpe_improvement = None
    
    # Compare drawdowns
    drawdown_improvement = abs(regime_metrics['max_drawdown']) < abs(base_metrics['max_drawdown'])
    
    logger.info(f"\nValidation Result:")
    logger.info(f"- Return Improvement: {'Yes' if is_improvement else 'No'}")
    logger.info(f"- Drawdown Improvement: {'Yes' if drawdown_improvement else 'No'}")
    if sharpe_improvement is not None:
        logger.info(f"- Sharpe Ratio Improvement: {'Yes' if sharpe_improvement else 'No'}")
    
    # Return results
    return {
        'base_metrics': base_metrics,
        'regime_metrics': regime_metrics,
        'base_trades': base_trades,
        'regime_trades': regime_trades,
        'regime_stats': regime_stats,
        'is_improvement': is_improvement,
        'drawdown_improvement': drawdown_improvement,
        'sharpe_improvement': sharpe_improvement
    }

def plot_comparison(base_equity, regime_equity, base_trades, regime_trades, 
                  symbol, output_dir):
    """
    Plot comparison of equity curves for both strategies.
    
    Args:
        base_equity: Equity curve for base strategy
        regime_equity: Equity curve for regime-filtered strategy
        base_trades: Trades from base strategy
        regime_trades: Trades from regime-filtered strategy
        symbol: Symbol being traded
        output_dir: Directory for saving plot
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot equity curves
    ax1.plot(base_equity['timestamp'], base_equity['equity'], 
           label='Standard Strategy', color='blue', linewidth=1.5)
    ax1.plot(regime_equity['timestamp'], regime_equity['equity'], 
           label='Regime-Filtered Strategy', color='green', linewidth=1.5)
    
    # Configure first subplot
    ax1.set_title(f'Equity Curve Comparison: {symbol}', fontsize=14)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Calculate drawdowns for both strategies
    base_dd = calculate_drawdown(base_equity)
    regime_dd = calculate_drawdown(regime_equity)
    
    # Plot drawdowns
    ax2.fill_between(base_equity['timestamp'], 0, base_dd, color='blue', alpha=0.3, label='Base Drawdown')
    ax2.fill_between(regime_equity['timestamp'], 0, regime_dd, color='green', alpha=0.3, label='Regime Drawdown')
    
    # Configure second subplot
    ax2.set_title('Drawdown Comparison', fontsize=14)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Format x-axis dates
    import matplotlib.dates as mdates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Rotate date labels - FIXED: use get_xticklabels() instead of xticklabels()
    plt.setp(ax1.get_xticklabels(), rotation=45)
    plt.setp(ax2.get_xticklabels(), rotation=45)
    
    # Add trade statistics
    base_return = ((base_equity['equity'].iloc[-1] / base_equity['equity'].iloc[0]) - 1) * 100
    regime_return = ((regime_equity['equity'].iloc[-1] / regime_equity['equity'].iloc[0]) - 1) * 100
    
    stats_text = (
        f"Base Strategy: Return={base_return:.2f}%, "
        f"Max DD={base_dd.min():.2f}%, "
        f"Trades={len(base_trades)}\n"
        f"Regime Strategy: Return={regime_return:.2f}%, "
        f"Max DD={regime_dd.min():.2f}%, "
        f"Trades={len(regime_trades)}"
    )
    
    # Add text box with statistics
    ax1.text(0.01, 0.01, stats_text, transform=ax1.transAxes, fontsize=10,
           bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{symbol}_regime_validation.png"), dpi=150)
    logger.info(f"Saved comparison chart to {os.path.join(output_dir, f'{symbol}_regime_validation.png')}")
    
    plt.close(fig)

def calculate_drawdown(equity_curve):
    """Calculate drawdown series from equity curve."""
    # Calculate rolling maximum
    equity = equity_curve['equity'].values
    rolling_max = np.maximum.accumulate(equity)
    
    # Calculate drawdown percentage
    drawdown = (equity / rolling_max - 1) * 100
    
    return drawdown

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Regime Filter Implementation')
    parser.add_argument('--data', type=str, required=True, help='Path to price data CSV file')
    parser.add_argument('--output', type=str, default=None, help='Directory to save results')
    parser.add_argument('--cash', type=float, default=10000.0, help='Initial portfolio cash')
    
    args = parser.parse_args()
    
    results = validate_regime_filter(
        data_path=args.data,
        output_dir=args.output,
        initial_cash=args.cash
    )
    
    # Print summary
    print("\n=== REGIME FILTER VALIDATION SUMMARY ===")
    print(f"Performance Improvement: {'Yes' if results['is_improvement'] else 'No'}")
    print(f"Risk Reduction: {'Yes' if results['drawdown_improvement'] else 'No'}")
    if results['sharpe_improvement'] is not None:
        print(f"Sharpe Ratio Improvement: {'Yes' if results['sharpe_improvement'] else 'No'}")
    
    # Show filter stats
    filter_rate = results['regime_stats']['filtered_signals'] / (
        results['regime_stats']['passed_signals'] + results['regime_stats']['filtered_signals']
    ) * 100 if (results['regime_stats']['passed_signals'] + results['regime_stats']['filtered_signals']) > 0 else 0
    
    print(f"\nFilter Statistics:")
    print(f"- Total Signals: {results['regime_stats']['passed_signals'] + results['regime_stats']['filtered_signals']}")
    print(f"- Signals Passed: {results['regime_stats']['passed_signals']}")
    print(f"- Signals Filtered: {results['regime_stats']['filtered_signals']}")
    print(f"- Filter Rate: {filter_rate:.2f}%")
