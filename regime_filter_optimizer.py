"""
Regime-Filtered Strategy Optimizer

This script optimizes a regime-filtered MA crossover strategy, finding
the best parameters for both the moving averages and the regime detector.
"""
import logging
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Import core components
from src.core.events.event_bus import EventBus
from src.core.events.event_types import SignalEvent
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler

# Import strategies
from src.strategy.strategies.ma_crossover import MovingAverageCrossoverStrategy
from regime_filtered_ma_strategy import RegimeFilteredMAStrategy

# Import regime detection
from src.models.filters.regime.detector import MarketRegime
from src.models.filters.regime.detector_factory import RegimeDetectorFactory

# Import optimization components
from src.models.optimization import (
    GridSearchOptimizer,
    OptimizationManager,
    evaluate_backtest,
    OptimizationValidator
)

# Import analytics
from src.analytics.performance import PerformanceAnalytics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def optimize_regime_filtered_strategy(data_path, output_dir=None, initial_cash=10000.0):
    """
    Optimize a regime-filtered MA crossover strategy.
    
    Args:
        data_path: Path to price data CSV file
        output_dir: Directory for saving results
        initial_cash: Initial portfolio cash
        
    Returns:
        dict: Optimization results
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
    
    # Analyze data characteristics
    data_stats = analyze_data(df)
    
    # Define parameter space based on data characteristics
    param_space = determine_parameter_space(data_stats)
    
    # Create regime detector
    base_detector = RegimeDetectorFactory.create_detector(detector_type='enhanced')
    
    # Define base allowed regimes configuration
    base_allowed_regimes = {
        MarketRegime.UPTREND: [SignalEvent.BUY],             # Only buy in uptrends
        MarketRegime.DOWNTREND: [SignalEvent.SELL],          # Only sell in downtrends
        MarketRegime.SIDEWAYS: [SignalEvent.BUY, SignalEvent.SELL],  # Allow both in sideways
        MarketRegime.VOLATILE: [],                           # No trading in volatile markets
        MarketRegime.UNKNOWN: [SignalEvent.BUY, SignalEvent.SELL]    # Allow both if unknown
    }
    
    # Create strategy for optimization
    strategy = RegimeFilteredMAStrategy(
        name="regime_filtered_ma",
        symbols=[symbol],
        fast_window=param_space['fast_window'][0],  # Default to first value
        slow_window=param_space['slow_window'][0],  # Default to first value
        regime_detector=base_detector,
        allowed_regimes=base_allowed_regimes
    )
    
    # Define parameter constraints
    def ma_constraint(params):
        """Ensure fast window is less than slow window"""
        return params.get('fast_window', 10) < params.get('slow_window', 30)
    
    constraints = [ma_constraint]
    
    # Create optimization manager
    opt_manager = OptimizationManager(name="regime_filtered_optimizer")
    
    # Register the strategy as optimization target
    opt_manager.register_target("regime_filtered_ma", strategy)
    
    # Register optimizers (grid search)
    opt_manager.register_optimizer("grid", GridSearchOptimizer())
    
    # Create custom evaluation function that uses data_handler
    def evaluate_strategy(component, **kwargs):
        """Custom evaluation function for regime-filtered strategy."""
        return evaluate_backtest(
            component=component,
            data_handler=data_handler,
            initial_cash=initial_cash,
            metric='sharpe_ratio'
        )
    
    # Register evaluation functions
    opt_manager.register_evaluator("sharpe_ratio", evaluate_strategy)
    
    # Run optimization
    logger.info("Starting regime-filtered strategy optimization...")
    opt_results = opt_manager.optimize_component(
        target_name="regime_filtered_ma",
        optimizer_name="grid",
        evaluator_name="sharpe_ratio",
        param_space=param_space,
        constraints=constraints
    )
    
    # Extract and process results
    results = process_optimization_results(opt_results, opt_manager, strategy, symbol, output_dir)
    
    # Run a final backtest with the best parameters
    best_params = opt_results.get('best_params', {})
    logger.info(f"Running final backtest with best parameters: {best_params}")
    
    # Create a new strategy with best parameters
    best_regime_detector = RegimeDetectorFactory.create_detector(detector_type='enhanced')
    
    best_strategy = RegimeFilteredMAStrategy(
        name="regime_filtered_ma_best",
        symbols=[symbol],
        fast_window=best_params.get('fast_window', 10),
        slow_window=best_params.get('slow_window', 30),
        regime_detector=best_regime_detector,
        allowed_regimes=base_allowed_regimes  # Use base allowed regimes
    )
    
    # Reset data handler
    data_handler.reset()
    
    # Run final backtest
    equity_curve, trades = run_backtest(
        component=best_strategy,
        data_handler=data_handler,
        initial_cash=initial_cash
    )
    
    # Calculate and log final metrics
    final_metrics = PerformanceAnalytics.calculate_metrics(equity_curve, trades)
    
    logger.info("\n===== FINAL BACKTEST RESULTS =====")
    logger.info(f"Best Parameters: {best_params}")
    logger.info(f"Sharpe Ratio: {final_metrics.get('sharpe_ratio', 0):.4f}")
    logger.info(f"Total Return: {final_metrics.get('total_return', 0):.2f}%")
    logger.info(f"Max Drawdown: {final_metrics.get('max_drawdown', 0):.2f}%")
    logger.info(f"Trade Count: {final_metrics.get('trade_count', 0)}")
    logger.info(f"Win Rate: {final_metrics.get('win_rate', 0):.2f}%")
    
    # Get regime stats
    regime_stats = best_strategy.get_regime_stats()
    logger.info(f"Regime Filtering Stats:")
    logger.info(f"- Signals Passed: {regime_stats['passed_signals']}")
    logger.info(f"- Signals Filtered: {regime_stats['filtered_signals']}")
    
    # Create equity curve chart
    plot_equity_curve(equity_curve, trades, symbol, best_params, output_dir)
    
    return {
        'best_params': best_params,
        'final_metrics': final_metrics,
        'trades': trades,
        'equity_curve': equity_curve,
        'regime_stats': regime_stats
    }

def determine_parameter_space(data_stats):
    """
    Determine appropriate parameter space based on data characteristics.
    
    Args:
        data_stats: Data statistics dictionary
        
    Returns:
        dict: Parameter space for optimization
    """
    frequency = data_stats.get('frequency', 'daily')
    bars_count = data_stats.get('bars', 0)
    
    # Minimum requirements
    min_data_length = 30  # Minimum amount of data needed
    min_gap = 5  # Minimum gap between fast and slow MA
    
    if bars_count < min_data_length:
        logger.warning(f"Insufficient data ({bars_count} bars) for proper optimization. "
                      f"Minimum recommended is {min_data_length} bars.")
    
    # Set parameter space based on data frequency and size
    if frequency == 'minute':
        # For minute data, use smaller window ranges
        fast_windows = [3, 5, 8, 13, 21]
        slow_windows = [13, 21, 34, 55, 89]
    elif frequency == 'hourly':
        # For hourly data
        fast_windows = [3, 5, 8, 13, 21]
        slow_windows = [13, 21, 34, 55, 89]
    else:  # daily or weekly
        # For daily/weekly data
        fast_windows = [5, 8, 10, 12, 15, 20]
        slow_windows = [20, 30, 40, 50, 60]
    
    # Filter slow windows to ensure they are all greater than the largest fast window plus the minimum gap
    max_fast = max(fast_windows) if fast_windows else 0
    slow_windows = [sw for sw in slow_windows if sw > max_fast + min_gap]
    
    # Define regime detector parameters
    # We'll use different detector types with different sensitivities
    detector_types = ['basic', 'enhanced']
    
    # Create parameter space dictionary
    param_space = {
        'fast_window': fast_windows,
        'slow_window': slow_windows
    }
    
    # Log parameter space
    logger.info("Parameter space for optimization:")
    logger.info(f"- Fast windows: {fast_windows}")
    logger.info(f"- Slow windows: {slow_windows}")
    logger.info(f"- Total parameter combinations: {len(fast_windows) * len(slow_windows)}")
    
    return param_space

def analyze_data(df):
    """
    Analyze data characteristics to inform parameter selection.
    
    Args:
        df: DataFrame with price data
        
    Returns:
        dict: Data statistics
    """
    # Basic statistics
    stats = {
        'bars': len(df),
        'start_date': df.index.min(),
        'end_date': df.index.max()
    }
    
    if isinstance(df.index, pd.DatetimeIndex):
        stats['duration'] = df.index.max() - df.index.min()
        if len(df) > 1:
            # Calculate average time between bars
            avg_interval = (df.index.max() - df.index.min()) / (len(df) - 1)
            stats['avg_interval'] = avg_interval
            
            # Determine data frequency
            if avg_interval < pd.Timedelta(minutes=5):
                stats['frequency'] = 'minute'
            elif avg_interval < pd.Timedelta(hours=1):
                stats['frequency'] = 'hourly'
            elif avg_interval < pd.Timedelta(days=1):
                stats['frequency'] = 'daily'
            else:
                stats['frequency'] = 'weekly'
    
    # Calculate price statistics
    if 'close' in df.columns:
        stats['avg_price'] = df['close'].mean()
        stats['min_price'] = df['close'].min()
        stats['max_price'] = df['close'].max()
        stats['volatility'] = df['close'].std()
        
        # Calculate returns
        df['return'] = df['close'].pct_change()
        stats['avg_return'] = df['return'].mean()
        stats['return_std'] = df['return'].std()
    
    # Log findings
    logger.info(f"Data analysis for {len(df)} bars:")
    logger.info(f"- Date range: {stats['start_date']} to {stats['end_date']}")
    if 'duration' in stats:
        logger.info(f"- Duration: {stats['duration']}")
    if 'avg_interval' in stats:
        logger.info(f"- Average interval: {stats['avg_interval']}")
    if 'frequency' in stats:
        logger.info(f"- Detected frequency: {stats['frequency']}")
    
    return stats

def process_optimization_results(opt_results, opt_manager, strategy, symbol, output_dir):
    """
    Process optimization results and generate reports.
    
    Args:
        opt_results: Results from optimization
        opt_manager: Optimization manager
        strategy: Strategy instance
        symbol: Symbol being traded
        output_dir: Directory for output files
        
    Returns:
        dict: Processed results
    """
    # Get best parameters
    best_params = opt_results.get('best_params', {})
    best_score = opt_results.get('best_score', 0)
    
    # Get all results
    all_results = opt_results.get('all_results', [])
    
    # Log best parameters
    logger.info(f"Optimization complete. Best parameters:")
    for param, value in best_params.items():
        logger.info(f"- {param}: {value}")
    logger.info(f"Best Sharpe ratio: {best_score:.4f}")
    
    # Generate report
    report = ["# Regime-Filtered Strategy Optimization Report\n"]
    
    # Add summary
    report.append("## Summary\n")
    report.append(f"- Symbol: {symbol}")
    report.append(f"- Total parameter combinations tested: {len(all_results)}")
    report.append(f"- Best Sharpe ratio: {best_score:.4f}")
    
    report.append("")
    
    # Add top results table
    report.append("## Top Results\n")
    
    # Table header
    report.append("| Rank | Fast Window | Slow Window | Sharpe | Return (%) | Max DD (%) | Trades | Win Rate (%) |")
    report.append("|------|------------|------------|--------|------------|------------|--------|--------------|")
    
    # Sort results by score (should already be sorted, but just to be sure)
    sorted_results = sorted(
        all_results, 
        key=lambda x: x.get('score', float('-inf')), 
        reverse=True
    )
    
    # Add rows for top 20 results
    for i, result in enumerate(sorted_results[:20], 1):
        params = result.get('params', {})
        score = result.get('score', 0)
        metrics = result.get('metrics', {})
        
        report.append(
            f"| {i} | " +
            f"{params.get('fast_window', '')} | " +
            f"{params.get('slow_window', '')} | " +
            f"{score:.4f} | " +
            f"{metrics.get('total_return', 0):.2f} | " +
            f"{metrics.get('max_drawdown', 0):.2f} | " +
            f"{metrics.get('trade_count', 0)} | " +
            f"{metrics.get('win_rate', 0):.2f} |"
        )
    
    report.append("")
    
    # Add parameter sensitivity analysis if we have enough results
    if len(all_results) > 10:
        report.append("## Parameter Sensitivity Analysis\n")
        
        # Extract all unique parameter values
        fast_windows = sorted(list(set(r['params']['fast_window'] for r in all_results if 'params' in r)))
        slow_windows = sorted(list(set(r['params']['slow_window'] for r in all_results if 'params' in r)))
        
        # Create sensitivity analysis for fast windows
        report.append("### Fast Window Sensitivity\n")
        report.append("| Fast Window | Avg Sharpe | Best Sharpe | Worst Sharpe | Avg Return (%) |")
        report.append("|------------|------------|-------------|--------------|----------------|")
        
        for fast in fast_windows:
            # Get all results with this fast window
            results_with_fast = [r for r in all_results if 'params' in r and r['params'].get('fast_window') == fast]
            
            if results_with_fast:
                # Calculate statistics
                sharpes = [r.get('score', 0) for r in results_with_fast]
                returns = [r.get('metrics', {}).get('total_return', 0) for r in results_with_fast if 'metrics' in r]
                
                avg_sharpe = sum(sharpes) / len(sharpes)
                best_sharpe = max(sharpes)
                worst_sharpe = min(sharpes)
                avg_return = sum(returns) / len(returns) if returns else 0
                
                report.append(
                    f"| {fast} | " +
                    f"{avg_sharpe:.4f} | " +
                    f"{best_sharpe:.4f} | " +
                    f"{worst_sharpe:.4f} | " +
                    f"{avg_return:.2f} |"
                )
        
        # Create sensitivity analysis for slow windows
        report.append("\n### Slow Window Sensitivity\n")
        report.append("| Slow Window | Avg Sharpe | Best Sharpe | Worst Sharpe | Avg Return (%) |")
        report.append("|------------|------------|-------------|--------------|----------------|")
        
        for slow in slow_windows:
            # Get all results with this slow window
            results_with_slow = [r for r in all_results if 'params' in r and r['params'].get('slow_window') == slow]
            
            if results_with_slow:
                # Calculate statistics
                sharpes = [r.get('score', 0) for r in results_with_slow]
                returns = [r.get('metrics', {}).get('total_return', 0) for r in results_with_slow if 'metrics' in r]
                
                avg_sharpe = sum(sharpes) / len(sharpes)
                best_sharpe = max(sharpes)
                worst_sharpe = min(sharpes)
                avg_return = sum(returns) / len(returns) if returns else 0
                
                report.append(
                    f"| {slow} | " +
                    f"{avg_sharpe:.4f} | " +
                    f"{best_sharpe:.4f} | " +
                    f"{worst_sharpe:.4f} | " +
                    f"{avg_return:.2f} |"
                )
    
    # Add conclusion
    report.append("\n## Conclusion\n")
    
    # Add recommendations based on results
    top_return = None
    if sorted_results and 'metrics' in sorted_results[0]:
        top_return = sorted_results[0]['metrics'].get('total_return', 0)
    
    if top_return is not None and top_return > 0:
        # Strategy is profitable with optimal parameters
        report.append("The regime-filtered MA Crossover strategy shows positive results with the optimal parameters:")
        report.append(f"- Fast window: {best_params.get('fast_window', '')}")
        report.append(f"- Slow window: {best_params.get('slow_window', '')}")
        report.append("")
        report.append("These parameters achieved the highest Sharpe ratio during testing.")
    else:
        # Strategy did not show profitable results
        report.append("**Warning: No profitable parameter combinations were found for this strategy on this dataset.**")
        report.append("")
        report.append("Recommendations:")
        report.append("1. Try a different regime detection approach")
        report.append("2. Modify the allowed trades for each regime type")
        report.append("3. Consider a different base strategy")
        report.append("")
        report.append("The least unprofitable parameters are:")
        report.append(f"- Fast window: {best_params.get('fast_window', '')}")
        report.append(f"- Slow window: {best_params.get('slow_window', '')}")
    
    # Add timestamp
    import datetime
    now = datetime.datetime.now()
    report.append(f"\n*Report generated on {now.strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Save report to file
    report_path = os.path.join(output_dir, f"{symbol}_regime_filtered_optimization_report.md")
    with open(report_path, 'w') as f:
        f.write("\n".join(report))
    logger.info(f"Optimization report saved to {report_path}")
    
    # Create visualization
    visualize_results(opt_results, symbol, output_dir)
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': all_results,
        'report_path': report_path
    }

def visualize_results(opt_results, symbol, output_dir):
    """
    Create visualizations of optimization results.
    
    Args:
        opt_results: Optimization results
        symbol: Symbol being traded
        output_dir: Directory for saving visualizations
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    
    # Extract all results
    all_results = opt_results.get('all_results', [])
    
    if not all_results:
        logger.warning("No results available for visualization")
        return
    
    # Extract unique parameter values
    fast_windows = sorted(list(set(r['params']['fast_window'] for r in all_results if 'params' in r)))
    slow_windows = sorted(list(set(r['params']['slow_window'] for r in all_results if 'params' in r)))
    
    # Create grid for heatmap
    sharpe_grid = np.zeros((len(fast_windows), len(slow_windows)))
    return_grid = np.zeros((len(fast_windows), len(slow_windows)))
    trades_grid = np.zeros((len(fast_windows), len(slow_windows)))
    
    # Fill grids with NaN by default
    sharpe_grid.fill(np.nan)
    return_grid.fill(np.nan)
    trades_grid.fill(np.nan)
    
    # Map for looking up indices
    fast_indices = {w: i for i, w in enumerate(fast_windows)}
    slow_indices = {w: i for i, w in enumerate(slow_windows)}
    
    # Fill grids with values
    for result in all_results:
        if 'params' in result:
            fast = result['params'].get('fast_window')
            slow = result['params'].get('slow_window')
            
            # Skip if not in our maps
            if fast not in fast_indices or slow not in slow_indices:
                continue
                
            # Only include valid combinations (fast < slow)
            if fast < slow:
                i = fast_indices[fast]
                j = slow_indices[slow]
                
                # Fill Sharpe ratio grid
                sharpe_grid[i, j] = result.get('score', np.nan)
                
                # Fill return and trades grids if metrics available
                if 'metrics' in result:
                    return_grid[i, j] = result['metrics'].get('total_return', np.nan)
                    trades_grid[i, j] = result['metrics'].get('trade_count', np.nan)
    
    # Create Sharpe ratio heatmap
    fig_sharpe, ax_sharpe = plt.subplots(figsize=(12, 10))
    
    # Use a diverging colormap centered at zero
    cmap = plt.cm.RdYlGn
    norm = colors.TwoSlopeNorm(vmin=sharpe_grid.min(), vcenter=0, vmax=max(0.1, sharpe_grid.max()))
    
    # Create masked array for invalid combinations
    masked_sharpe = np.ma.masked_invalid(sharpe_grid)
    
    # Generate the heatmap
    im = ax_sharpe.imshow(masked_sharpe, cmap=cmap, norm=norm)
    
    # Add colorbar
    cbar = ax_sharpe.figure.colorbar(im, ax=ax_sharpe)
    cbar.ax.set_ylabel('Sharpe Ratio', rotation=-90, va="bottom", fontsize=12)
    
    # Set ticks and labels
    ax_sharpe.set_xticks(np.arange(len(slow_windows)))
    ax_sharpe.set_yticks(np.arange(len(fast_windows)))
    ax_sharpe.set_xticklabels(slow_windows)
    ax_sharpe.set_yticklabels(fast_windows)
    
    # Rotate tick labels
    plt.setp(ax_sharpe.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add axis labels and title
    ax_sharpe.set_xlabel('Slow Window', fontsize=14)
    ax_sharpe.set_ylabel('Fast Window', fontsize=14)
    ax_sharpe.set_title(f'{symbol}: Regime-Filtered MA Strategy Sharpe Ratio', fontsize=16)
    
    # Add grid
    ax_sharpe.set_xticks(np.arange(len(slow_windows)+1)-0.5, minor=True)
    ax_sharpe.set_yticks(np.arange(len(fast_windows)+1)-0.5, minor=True)
    ax_sharpe.grid(which="minor", color="black", linestyle='-', linewidth=1, alpha=0.2)
    
    # Add text annotations with values
    for i in range(len(fast_windows)):
        for j in range(len(slow_windows)):
            fast = fast_windows[i]
            slow = slow_windows[j]
            
            if fast < slow:  # Only for valid combinations
                value = sharpe_grid[i, j]
                if not np.isnan(value):
                    # Determine text color based on background
                    text_color = "white" if abs(value) > 2 else "black"
                    
                    # Show the value with appropriate formatting
                    text = f"{value:.2f}"
                    ax_sharpe.text(j, i, text, ha="center", va="center", 
                                 color=text_color, fontsize=8)
    
    # Save the figure
    sharpe_path = os.path.join(output_dir, f"{symbol}_regime_filtered_sharpe_heatmap.png")
    fig_sharpe.tight_layout()
    fig_sharpe.savefig(sharpe_path, dpi=150, bbox_inches='tight')
    logger.info(f"Sharpe ratio heatmap saved to {sharpe_path}")

def plot_equity_curve(equity_curve, trades, symbol, params, output_dir):
    """
    Plot equity curve with trade markers.
    
    Args:
        equity_curve: DataFrame with equity curve data
        trades: List of trades
        symbol: Symbol being traded
        params: Strategy parameters
        output_dir: Directory for saving plot
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot equity curve
    ax1.plot(equity_curve['timestamp'], equity_curve['equity'], 
           label='Portfolio Value', color='blue', linewidth=1.5)
    
    # Add initial cash line
    initial_cash = equity_curve['equity'].iloc[0] if not equity_curve.empty else 10000.0
    ax1.axhline(y=initial_cash, color='gray', linestyle='--', alpha=0.6, label='Initial Cash')
    
    # Format first subplot
    ax1.set_title(f'Regime-Filtered MA Strategy - {symbol}', fontsize=14)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add parameter info
    param_text = f"Fast MA: {params.get('fast_window', 'N/A')}, Slow MA: {params.get('slow_window', 'N/A')}"
    ax1.text(0.02, 0.05, param_text, transform=ax1.transAxes, fontsize=10, 
           bbox=dict(facecolor='white', alpha=0.8))
    
    # Add trades as markers if available
    if trades:
        # For buys
        buy_times = [t['timestamp'] for t in trades if t['direction'] == 'BUY']
        buy_equities = []
        
        for t in trades:
            if t['direction'] == 'BUY':
                ts = t['timestamp']
                idx = equity_curve['timestamp'].searchsorted(ts)
                if idx < len(equity_curve):
                    buy_equities.append(equity_curve['equity'].iloc[idx])
                else:
                    buy_equities.append(None)
                    
        buy_equities = [e for e in buy_equities if e is not None]
        buy_times = buy_times[:len(buy_equities)]
        
        if buy_times and buy_equities:
            ax1.scatter(buy_times, buy_equities, color='green', marker='^', 
                      s=50, alpha=0.7, label='Buy')
        
        # For sells
        sell_times = [t['timestamp'] for t in trades if t['direction'] == 'SELL']
        sell_equities = []
        
        for t in trades:
            if t['direction'] == 'SELL':
                ts = t['timestamp']
                idx = equity_curve['timestamp'].searchsorted(ts)
                if idx < len(equity_curve):
                    sell_equities.append(equity_curve['equity'].iloc[idx])
                else:
                    sell_equities.append(None)
                    
        sell_equities = [e for e in sell_equities if e is not None]
        sell_times = sell_times[:len(sell_equities)]
        
        if sell_times and sell_equities:
            ax1.scatter(sell_times, sell_equities, color='red', marker='v', 
                      s=50, alpha=0.7, label='Sell')
    
    ax1.legend(loc='best')
    
    # Calculate and plot drawdown
    equity_values = equity_curve['equity'].values
    rolling_max = np.maximum.accumulate(equity_values)
    drawdown = (equity_values / rolling_max - 1) * 100
    
    # Plot drawdown
    ax2.fill_between(equity_curve['timestamp'], 0, drawdown, color='red', alpha=0.3)
    
    # Format second subplot
    ax2.set_title('Drawdown', fontsize=14)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis dates
    import matplotlib.dates as mdates
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax1.xaxis.set_major_formatter(date_format)
    ax2.xaxis.set_major_formatter(date_format)
    
    # Rotate x-axis labels
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    
    # Set shared x-axis
    # plt.setp(ax1.get_xticklabels(), visible=False)
    
    # Add performance stats
    # Calculate key metrics
    initial_equity = equity_curve['equity'].iloc[0] if not equity_curve.empty else initial_cash
    final_equity = equity_curve['equity'].iloc[-1] if not equity_curve.empty else initial_cash
    total_return = ((final_equity / initial_equity) - 1) * 100
    max_dd = drawdown.min() if len(drawdown) > 0 else 0
    
    # Add text box with performance metrics
    stats_text = (f"Total Return: {total_return:.2f}%\n"
                 f"Max Drawdown: {max_dd:.2f}%\n"
                 f"Trades: {len(trades)}")
    
    # Add text box to top right of drawdown chart
    ax2.text(0.98, 0.05, stats_text, transform=ax2.transAxes, fontsize=10,
           bbox=dict(facecolor='white', alpha=0.8), ha='right')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(output_dir, f"{symbol}_regime_filtered_equity.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Equity curve chart saved to {fig_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize Regime-Filtered Strategy')
    parser.add_argument('--data', type=str, required=True, help='Path to price data CSV file')
    parser.add_argument('--output', type=str, default=None, help='Directory to save results')
    parser.add_argument('--cash', type=float, default=10000.0, help='Initial portfolio cash')
    
    args = parser.parse_args()
    
    optimize_regime_filtered_strategy(
        data_path=args.data,
        output_dir=args.output,
        initial_cash=args.cash
    )
