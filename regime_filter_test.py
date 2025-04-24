"""
Regime Filter Backtest Script

This script runs a backtest comparing a standard MA Crossover strategy
with a regime-filtered version to demonstrate the impact of regime filtering.
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
from src.strategy.strategies.ma_crossover import MovingAverageCrossoverStrategy
from regime_filtered_ma_strategy import RegimeFilteredMAStrategy

# Import regime detection
from src.models.filters.regime.regime_detector import MarketRegime
from src.models.filters.regime.detector_factory import RegimeDetectorFactory

# Import backtest utilities
from src.execution.backtest.backtest import run_backtest
from src.analytics.performance import PerformanceAnalytics

from src.core.events.event_types import EventType, SignalEvent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_regime_filter_test(data_path, output_dir=None, initial_cash=10000.0):
    """
    Run a backtest comparing standard MA Crossover with regime-filtered version.
    
    Args:
        data_path: Path to price data CSV file
        output_dir: Directory for saving results
        initial_cash: Initial portfolio cash
        
    Returns:
        dict: Backtest results
    """
    # Enable detailed logging for debugging
    logging.getLogger('src.strategy.risk.risk_manager').setLevel(logging.DEBUG)
    logging.getLogger('src.execution.brokers.simulated').setLevel(logging.DEBUG)
    logging.getLogger('src.execution.portfolio').setLevel(logging.DEBUG)
    
    # Set output directory
    if output_dir is None:
        output_dir = os.getcwd()
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract symbol from filename
    filename = os.path.basename(data_path)
    symbol = filename.split('_')[0]  # Assumes format SYMBOL_timeframe.csv
    
    logger.info(f"Running backtest comparison for {symbol}...")
    
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
    
    # First backtest: Standard MA crossover strategy
    logger.info(f"Setting up Standard MA Crossover backtest for {symbol}...")
    
    # Create clean event bus for first backtest
    ma_event_bus = EventBus()
    
    # Define strategy parameters
    fast_window = 10
    slow_window = 30
    
    # Create standard MA Crossover strategy
    ma_strategy = MovingAverageCrossoverStrategy(
        name="ma_crossover",
        symbols=[symbol],
        fast_window=fast_window,
        slow_window=slow_window
    )
    ma_strategy.set_event_bus(ma_event_bus)
    
    # Create risk manager for standard strategy
    from src.strategy.risk.risk_manager import SimplePassthroughRiskManager
    ma_risk_manager = SimplePassthroughRiskManager(portfolio=None, event_bus=ma_event_bus)
    ma_risk_manager.set_debug(True)
    
    # Create broker for standard strategy
    from src.execution.brokers.simulated import SimulatedBroker
    ma_broker = SimulatedBroker(fill_emitter=ma_event_bus)
    ma_broker.set_event_bus(ma_event_bus)  # Set event bus using the setter method

    
    # Connect components to event bus
    ma_event_bus.register(EventType.SIGNAL, ma_risk_manager.on_signal)
    ma_event_bus.register(EventType.ORDER, ma_broker.place_order)
    
    # Connect risk manager to broker directly
    ma_risk_manager.broker = ma_broker
    
    # Run backtest for standard MA strategy
    logger.info(f"Running standard MA Crossover backtest for {symbol}...")
    ma_equity_curve, ma_trades = run_backtest(
        component=ma_strategy,
        data_handler=data_handler,
        initial_cash=initial_cash
    )
    
    if ma_trades:
        logger.info(f"Standard MA strategy executed {len(ma_trades)} trades")
        if len(ma_trades) > 0:
            logger.info(f"Sample trade: {ma_trades[0]}")
    else:
        logger.warning("Standard MA strategy did not execute any trades!")
    
    # Second backtest: Regime-filtered strategy
    logger.info(f"Setting up Regime-Filtered MA Crossover backtest for {symbol}...")
    
    # Reset data handler for the next backtest
    data_handler.reset()
    
    # Create clean event bus for second backtest
    regime_event_bus = EventBus()
    
    # Create regime detector
    regime_detector = RegimeDetectorFactory.create_detector(detector_type='enhanced')
    
    # Define allowed regimes for signals - make them more restrictive
    allowed_regimes = {
        MarketRegime.UPTREND: [SignalEvent.BUY],             # Only buy in uptrends
        MarketRegime.DOWNTREND: [SignalEvent.SELL],          # Only sell in downtrends
        MarketRegime.SIDEWAYS: [],                           # No trading in sideways
        MarketRegime.VOLATILE: [],                           # No trading in volatile markets
        MarketRegime.UNKNOWN: [SignalEvent.BUY, SignalEvent.SELL]  # Allow both if unknown
    }
    
    # Create regime-filtered strategy
    regime_filtered_strategy = RegimeFilteredMAStrategy(
        name="regime_filtered_ma",
        symbols=[symbol],
        fast_window=fast_window,
        slow_window=slow_window,
        regime_detector=regime_detector,
        allowed_regimes=allowed_regimes
    )
    regime_filtered_strategy.set_event_bus(regime_event_bus)
    
    # Create risk manager for regime-filtered strategy
    regime_risk_manager = SimplePassthroughRiskManager(portfolio=None, event_bus=regime_event_bus)
    regime_risk_manager.set_debug(True)
    
    # Create broker for regime-filtered strategy
    # Create broker for regime-filtered strategy
    regime_broker = SimulatedBroker(fill_emitter=regime_event_bus)
    regime_broker.set_event_bus(regime_event_bus)  # Set event bus using the setter method
    
    # Connect components to event bus
    regime_event_bus.register(EventType.SIGNAL, regime_risk_manager.on_signal)
    regime_event_bus.register(EventType.ORDER, regime_broker.place_order)
    
    # Connect risk manager to broker directly
    regime_risk_manager.broker = regime_broker
    
    # Run backtest for regime-filtered strategy
    logger.info(f"Running regime-filtered MA Crossover backtest for {symbol}...")
    regime_equity_curve, regime_trades = run_backtest(
        component=regime_filtered_strategy,
        data_handler=data_handler,
        initial_cash=initial_cash
    )
    
    if regime_trades:
        logger.info(f"Regime-filtered strategy executed {len(regime_trades)} trades")
        if len(regime_trades) > 0:
            logger.info(f"Sample trade: {regime_trades[0]}")
    else:
        logger.warning("Regime-filtered strategy did not execute any trades!")
    
    # Calculate performance metrics for both strategies
    ma_metrics = PerformanceAnalytics.calculate_metrics(ma_equity_curve, ma_trades)
    regime_metrics = PerformanceAnalytics.calculate_metrics(regime_equity_curve, regime_trades)
    
    # Print comparison
    logger.info("\n===== STRATEGY COMPARISON =====")
    logger.info(f"Standard MA Crossover ({fast_window}/{slow_window}):")
    logger.info(f"- Final Equity: ${ma_metrics.get('final_equity', 0.0):.2f}")
    logger.info(f"- Total Return: {ma_metrics.get('total_return', 0.0):.2f}%")
    logger.info(f"- Max Drawdown: {ma_metrics.get('max_drawdown', 0.0):.2f}%")
    logger.info(f"- Trade Count: {ma_metrics.get('trade_count', 0)}")
    logger.info(f"- Win Rate: {ma_metrics.get('win_rate', 0.0):.2f}%")
    
    logger.info(f"\nRegime-Filtered MA Crossover ({fast_window}/{slow_window}):")
    logger.info(f"- Final Equity: ${regime_metrics.get('final_equity', 0.0):.2f}")
    logger.info(f"- Total Return: {regime_metrics.get('total_return', 0.0):.2f}%")
    logger.info(f"- Max Drawdown: {regime_metrics.get('max_drawdown', 0.0):.2f}%")
    logger.info(f"- Trade Count: {regime_metrics.get('trade_count', 0)}")
    logger.info(f"- Win Rate: {regime_metrics.get('win_rate', 0.0):.2f}%")
    
    # Get regime stats
    regime_stats = regime_filtered_strategy.get_regime_stats()
    logger.info(f"\nRegime Filtering Stats:")
    logger.info(f"- Signals Passed: {regime_stats['passed_signals']}")
    logger.info(f"- Signals Filtered: {regime_stats['filtered_signals']}")
    
    # Create simplified plotting function that avoids trades
    def simplified_plot_comparison(ma_equity, regime_equity, symbol, output_dir):
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curves
        ax1.plot(ma_equity['timestamp'], ma_equity['equity'], 
               label='Standard MA', color='blue', linewidth=1.5)
        ax1.plot(regime_equity['timestamp'], regime_equity['equity'], 
               label='Regime-Filtered MA', color='green', linewidth=1.5)
        
        # Add initial cash line
        ax1.axhline(y=initial_cash, color='gray', linestyle='--', alpha=0.6, label='Initial Cash')
        
        # Configure first subplot
        ax1.set_title(f'Equity Curve Comparison: {symbol}', fontsize=14)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        # Calculate drawdowns for both strategies
        ma_dd = calculate_drawdown(ma_equity)
        regime_dd = calculate_drawdown(regime_equity)
        
        # Plot drawdowns
        ax2.fill_between(ma_equity['timestamp'], 0, ma_dd, color='blue', alpha=0.3, label='MA Drawdown')
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
        
        # Rotate x-axis labels
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{symbol}_regime_comparison.png"), dpi=150)
        plt.close()
        logger.info(f"Saved comparison chart to {os.path.join(output_dir, f'{symbol}_regime_comparison.png')}")
    
    # Use the simplified plotting function
    simplified_plot_comparison(ma_equity_curve, regime_equity_curve, symbol, output_dir)
    
    # Generate simplified report
    def simplified_generate_report(ma_metrics, regime_metrics, regime_stats, symbol, output_dir):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = [
            "# Regime-Filtered Strategy Backtest Report\n",
            f"**Symbol:** {symbol}",
            f"**Date:** {now}\n",
            
            "## Strategy Parameters",
            f"- Standard MA Crossover: Fast Window={fast_window}, Slow Window={slow_window}",
            f"- Regime-Filtered MA: Uses the same MA parameters with regime filtering\n",
            
            "## Performance Comparison\n",
            "| Metric | Standard MA | Regime-Filtered MA | Difference |",
            "|--------|------------|-------------------|------------|"
        ]
        
        # Add metric comparisons
        metrics_to_compare = [
            ('total_return', 'Total Return (%)', '{:.2f}%'),
            ('final_equity', 'Final Equity ($)', '${:.2f}'),
            ('max_drawdown', 'Max Drawdown (%)', '{:.2f}%')
        ]
        
        for key, label, format_str in metrics_to_compare:
            ma_value = ma_metrics.get(key, 0)
            regime_value = regime_metrics.get(key, 0)
            
            # Format the values
            ma_formatted = format_str.format(ma_value)
            regime_formatted = format_str.format(regime_value)
            
            # Special handling for max_drawdown (lower is better)
            if key == 'max_drawdown':
                diff = ma_value - regime_value
            else:
                diff = regime_value - ma_value
                
            diff_formatted = format_str.format(diff)
            
            report.append(f"| {label} | {ma_formatted} | {regime_formatted} | {diff_formatted} |")
        
        # Add regime filtering statistics
        report.extend([
            "\n## Regime Filtering Statistics",
            f"- Total Signals Generated: {regime_stats['passed_signals'] + regime_stats['filtered_signals']}",
            f"- Signals Passed: {regime_stats['passed_signals']}",
            f"- Signals Filtered: {regime_stats['filtered_signals']}",
            f"- Filtering Rate: {regime_stats['filtered_signals'] / (regime_stats['passed_signals'] + regime_stats['filtered_signals']) * 100 if (regime_stats['passed_signals'] + regime_stats['filtered_signals']) > 0 else 0:.2f}%\n"
        ])
        
        # Save report to file
        report_path = os.path.join(output_dir, f"{symbol}_regime_filter_report.md")
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Detailed report saved to {report_path}")
    
    # Use the simplified report generation
    simplified_generate_report(ma_metrics, regime_metrics, regime_stats, symbol, output_dir)
    
    return {
        'ma_metrics': ma_metrics,
        'regime_metrics': regime_metrics,
        'ma_trades': ma_trades,
        'regime_trades': regime_trades,
        'regime_stats': regime_stats
    }


def plot_comparison(ma_equity_curve, regime_equity_curve, ma_trades, regime_trades, 
                  symbol, output_dir):
    """
    Plot comparison of equity curves and drawdowns.
    
    Args:
        ma_equity_curve: Equity curve for MA strategy
        regime_equity_curve: Equity curve for regime-filtered strategy
        ma_trades: Trades from MA strategy
        regime_trades: Trades from regime-filtered strategy
        symbol: Symbol being traded
        output_dir: Directory for saving plot
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot equity curves
    ax1.plot(ma_equity_curve['timestamp'], ma_equity_curve['equity'], 
           label='Standard MA', color='blue', linewidth=1.5)
    ax1.plot(regime_equity_curve['timestamp'], regime_equity_curve['equity'], 
           label='Regime-Filtered MA', color='green', linewidth=1.5)
    
    # Configure first subplot
    ax1.set_title(f'Equity Curve Comparison: {symbol}', fontsize=14)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Calculate drawdowns for both strategies
    ma_dd = calculate_drawdown(ma_equity_curve)
    regime_dd = calculate_drawdown(regime_equity_curve)
    
    # Plot drawdowns
    ax2.fill_between(ma_equity_curve['timestamp'], 0, ma_dd, color='blue', alpha=0.3, label='MA Drawdown')
    ax2.fill_between(regime_equity_curve['timestamp'], 0, regime_dd, color='green', alpha=0.3, label='Regime Drawdown')
    
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
    
    # Rotate x-axis labels
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{symbol}_regime_comparison.png"), dpi=150)
    plt.close()
    logger.info(f"Saved comparison chart to {os.path.join(output_dir, f'{symbol}_regime_comparison.png')}")


def calculate_drawdown(equity_curve):
    """Calculate drawdown series from equity curve."""
    # Calculate rolling maximum
    equity = equity_curve['equity'].values
    rolling_max = np.maximum.accumulate(equity)
    
    # Calculate drawdown percentage
    drawdown = (equity / rolling_max - 1) * 100
    
    return drawdown

def generate_report(ma_metrics, regime_metrics, ma_trades, regime_trades, 
                   regime_stats, symbol, output_dir):
    """
    Generate a detailed report comparing the strategies.
    
    Args:
        ma_metrics: Performance metrics for MA strategy
        regime_metrics: Performance metrics for regime-filtered strategy
        ma_trades: Trades from MA strategy
        regime_trades: Trades from regime-filtered strategy
        regime_stats: Regime filtering statistics
        symbol: Symbol being traded
        output_dir: Directory for saving report
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = [
        "# Regime-Filtered Strategy Backtest Report\n",
        f"**Symbol:** {symbol}",
        f"**Date:** {now}\n",
        
        "## Strategy Parameters",
        f"- Standard MA Crossover: Fast Window={ma_metrics.get('fast_window', 'N/A')}, Slow Window={ma_metrics.get('slow_window', 'N/A')}",
        f"- Regime-Filtered MA: Uses the same MA parameters with regime filtering\n",
        
        "## Performance Comparison\n",
        "| Metric | Standard MA | Regime-Filtered MA | Difference | % Improvement |",
        "|--------|------------|-------------------|------------|---------------|"
    ]
    
    # Add metric comparisons
    metrics_to_compare = [
        ('total_return', 'Total Return (%)', '{:.2f}%'),
        ('final_equity', 'Final Equity ($)', '${:.2f}'),
        ('max_drawdown', 'Max Drawdown (%)', '{:.2f}%'),
        ('trade_count', 'Trade Count', '{:.0f}'),
        ('win_rate', 'Win Rate (%)', '{:.2f}%'),
        ('profit_factor', 'Profit Factor', '{:.2f}'),
        ('sharpe_ratio', 'Sharpe Ratio', '{:.2f}')
    ]
    
    for key, label, format_str in metrics_to_compare:
        if key in ma_metrics and key in regime_metrics:
            ma_value = ma_metrics[key]
            regime_value = regime_metrics[key]
            
            # Special handling for max_drawdown (lower is better)
            if key == 'max_drawdown':
                diff = ma_value - regime_value
                pct_improvement = diff / abs(ma_value) * 100 if ma_value != 0 else 0
            else:
                diff = regime_value - ma_value
                pct_improvement = diff / abs(ma_value) * 100 if ma_value != 0 else 0
            
            # Format the values
            ma_formatted = format_str.format(ma_value)
            regime_formatted = format_str.format(regime_value)
            diff_formatted = format_str.format(diff)
            pct_formatted = f"{pct_improvement:.2f}%"
            
            report.append(f"| {label} | {ma_formatted} | {regime_formatted} | {diff_formatted} | {pct_formatted} |")
    
    # Add regime filtering statistics
    report.extend([
        "\n## Regime Filtering Statistics",
        f"- Total Signals Generated: {regime_stats['passed_signals'] + regime_stats['filtered_signals']}",
        f"- Signals Passed: {regime_stats['passed_signals']}",
        f"- Signals Filtered: {regime_stats['filtered_signals']}",
        f"- Filtering Rate: {regime_stats['filtered_signals'] / (regime_stats['passed_signals'] + regime_stats['filtered_signals']) * 100 if (regime_stats['passed_signals'] + regime_stats['filtered_signals']) > 0 else 0:.2f}%\n"
    ])
    
    # Add trade analysis
    report.extend([
        "## Trade Analysis",
        "\n### Standard MA Crossover Strategy",
        f"- Total Trades: {len(ma_trades)}",
        f"- Average Trade P&L: ${np.mean([t.get('pnl', 0) for t in ma_trades]):.2f}",
        
        "\n### Regime-Filtered MA Strategy",
        f"- Total Trades: {len(regime_trades)}",
        f"- Average Trade P&L: ${np.mean([t.get('pnl', 0) for t in regime_trades]):.2f}\n"
    ])
    
    # Add conclusion based on results
    report.append("## Conclusion\n")
    
    # Compare total returns
    ma_return = ma_metrics.get('total_return', 0)
    regime_return = regime_metrics.get('total_return', 0)
    
    if regime_return > ma_return:
        report.append(f"The regime-filtered strategy outperformed the standard MA crossover by {regime_return - ma_return:.2f}% in total return. Regime filtering successfully improved performance by restricting trading to favorable market conditions.")
    elif ma_return > regime_return:
        report.append(f"The standard MA crossover strategy outperformed the regime-filtered strategy by {ma_return - regime_return:.2f}% in total return. In this case, the regime filtering may have been too restrictive and filtered out some profitable trades.")
    else:
        report.append(f"Both strategies performed equally in terms of total return. The regime filtering did not significantly impact overall performance in this test.")
    
    # Compare drawdowns
    ma_dd = ma_metrics.get('max_drawdown', 0)
    regime_dd = regime_metrics.get('max_drawdown', 0)
    
    if abs(regime_dd) < abs(ma_dd):
        report.append(f"\nHowever, the regime-filtered strategy showed improved risk management with a reduced maximum drawdown of {regime_dd:.2f}% compared to {ma_dd:.2f}% for the standard strategy.")
    elif abs(ma_dd) < abs(regime_dd):
        report.append(f"\nHowever, the standard strategy showed better risk characteristics with a lower maximum drawdown of {ma_dd:.2f}% compared to {regime_dd:.2f}% for the regime-filtered strategy.")
    
    # Final recommendations
    report.append("\n### Recommendations")
    
    if regime_return > ma_return and abs(regime_dd) <= abs(ma_dd):
        report.append("The regime-filtered strategy is clearly superior in this test, providing both higher returns and better risk management. Continue using regime filtering with this strategy.")
    elif regime_return > ma_return:
        report.append("The regime-filtered strategy provides higher returns but with increased drawdowns. Consider adjusting the regime detection parameters to better balance return and risk.")
    elif abs(regime_dd) < abs(ma_dd):
        report.append("While the regime-filtered strategy produced lower returns, it significantly reduced drawdowns. This approach may be valuable for risk-averse investors or in more volatile markets.")
    else:
        report.append("Further optimization of the regime detection parameters may be needed to improve performance. Consider testing different regime definitions or allowed signal combinations.")
    
    # Save report to file
    report_path = os.path.join(output_dir, f"{symbol}_regime_filter_report.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"Detailed report saved to {report_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Regime-Filtered Strategy Backtest')
    parser.add_argument('--data', type=str, required=True, help='Path to price data CSV file')
    parser.add_argument('--output', type=str, default=None, help='Directory to save results')
    parser.add_argument('--cash', type=float, default=10000.0, help='Initial portfolio cash')
    
    args = parser.parse_args()
    
    # Run the test and capture returned values
    results = run_regime_filter_test(
        data_path=args.data,
        output_dir=args.output,
        initial_cash=args.cash
    )
    
    # Log some additional information about the results if needed
    if results:
        ma_trades = results.get('ma_trades', [])
        regime_trades = results.get('regime_trades', [])
        
        logger.info(f"== Additional Results Information ==")
        logger.info(f"MA strategy trade count: {len(ma_trades)}")
        logger.info(f"Regime strategy trade count: {len(regime_trades)}")
        
        # Show sample trades if available
        if ma_trades and len(ma_trades) > 0:
            logger.info(f"Sample MA trade: {ma_trades[0]}")
        if regime_trades and len(regime_trades) > 0:
            logger.info(f"Sample regime trade: {regime_trades[0]}")
    
