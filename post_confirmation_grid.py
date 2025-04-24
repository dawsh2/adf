"""
Moving Average Crossover Strategy Parameter Grid Search

This script uses the built-in optimization framework to find optimal parameters
for the moving average crossover strategy.
"""
import logging
import pandas as pd
import numpy as np
import os
import datetime
from typing import Dict, List, Any

# Import core components
from src.core.events.event_bus import EventBus
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler

# Import optimization components
from src.models.optimization import (
    GridSearchOptimizer,
    OptimizationManager,
    evaluate_backtest,
    OptimizationValidator
)

# Import strategies
from src.strategy.strategies.ma_crossover import MovingAverageCrossoverStrategy

# Import analytics
from src.analytics.performance import PerformanceAnalytics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def optimize_ma_crossover(data_path, output_dir=None, initial_cash=10000.0, test_inverse=True):
    """
    Optimize MA Crossover strategy parameters using the optimization framework.
    
    Args:
        data_path: Path to data file
        output_dir: Directory for saving results
        initial_cash: Initial portfolio cash
        test_inverse: Whether to also test inverting the signals
        
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
    
    # Analyze data characteristics to determine appropriate window sizes
    data_stats = analyze_data(df)
    
    # Define parameter ranges based on data characteristics
    param_space = determine_parameter_space(data_stats)
    
    # Add inverse parameter if testing inverse strategy
    if test_inverse:
        param_space['invert_signals'] = [False, True]
        logger.info("Testing both normal and inverted signals")
    
    # Define parameter constraints
    def ma_constraint(params):
        """Ensure fast window is less than slow window"""
        return params['fast_window'] < params['slow_window']
    
    constraints = [ma_constraint]
    
    # Create base strategy instance with signal inverter adapter
    # Create a wrapper class to handle signal inversion
    class InvertibleMAStrategy:
        """Wrapper that can invert signals from MA Crossover strategy."""
        
        def __init__(self, name, symbols, fast_window=10, slow_window=30, price_key='close', invert_signals=False):
            self.name = name
            self.symbols = symbols if isinstance(symbols, list) else [symbols]
            self.fast_window = fast_window
            self.slow_window = slow_window
            self.price_key = price_key
            self.invert_signals = invert_signals
            
            # Create the actual strategy
            self.strategy = MovingAverageCrossoverStrategy(
                name=name,
                symbols=self.symbols,
                fast_window=fast_window,
                slow_window=slow_window,
                price_key=price_key
            )
            self.event_bus = None
        
        def set_event_bus(self, event_bus):
            """Set the event bus for this component."""
            self.event_bus = event_bus
            self.strategy.set_event_bus(event_bus)
            return self
        
        def on_bar(self, event):
            """Process a bar event, possibly inverting signals."""
            signal = self.strategy.on_bar(event)
            
            if signal and self.invert_signals:
                # Invert the signal (BUY -> SELL, SELL -> BUY)
                if hasattr(signal, 'get_signal_value') and hasattr(signal, 'data'):
                    # Get current signal value
                    signal_value = signal.get_signal_value()
                    
                    # Invert it
                    from src.core.events.event_types import SignalEvent
                    if signal_value == SignalEvent.BUY:
                        signal.data['signal_value'] = SignalEvent.SELL
                    elif signal_value == SignalEvent.SELL:
                        signal.data['signal_value'] = SignalEvent.BUY
            
            return signal
        
        def reset(self):
            """Reset the strategy."""
            self.strategy.reset()
        
        def get_parameters(self):
            """Get current parameters."""
            params = self.strategy.get_parameters()
            params['invert_signals'] = self.invert_signals
            return params
        
        def set_parameters(self, params):
            """Set parameters."""
            # Extract invert_signals parameter
            if 'invert_signals' in params:
                self.invert_signals = params['invert_signals']
                
                # Create new params dict without invert_signals for the actual strategy
                strategy_params = {k: v for k, v in params.items() if k != 'invert_signals'}
                self.strategy.set_parameters(strategy_params)
            else:
                self.strategy.set_parameters(params)
            
            # Update local copies of params for easy access
            if hasattr(self.strategy, 'fast_window'):
                self.fast_window = self.strategy.fast_window
            if hasattr(self.strategy, 'slow_window'):
                self.slow_window = self.strategy.slow_window
            if hasattr(self.strategy, 'price_key'):
                self.price_key = self.strategy.price_key
    
    # Create the invertible strategy
    invertible_strategy = InvertibleMAStrategy(
        name="ma_crossover_base",
        symbols=[symbol]
    )
    
    # Create optimization manager
    opt_manager = create_optimization_manager(invertible_strategy, data_handler, initial_cash)
    
    # Check available optimizers and register grid search if needed
    logger.info(f"Available optimizers: {list(opt_manager.optimizers.keys())}")
    
    # If no grid search optimizer is registered, register one
    if not any(name.lower().startswith('grid') for name in opt_manager.optimizers.keys()):
        logger.info("Registering grid search optimizer")
        grid_optimizer = GridSearchOptimizer()
        opt_manager.register_optimizer("grid", grid_optimizer)  # Register as "grid" to match validator
        optimizer_name = "grid"
    else:
        # Use the first grid search optimizer found
        optimizer_name = next(name for name in opt_manager.optimizers.keys() 
                            if name.lower().startswith('grid'))
    
    logger.info(f"Using optimizer: {optimizer_name}")
    
    # Run optimization
    logger.info("Starting MA Crossover optimization...")
    opt_results = opt_manager.optimize_component(
        target_name="ma_crossover_base",  # Use the actual registered name
        optimizer_name=optimizer_name,
        evaluator_name="sharpe_ratio",
        param_space=param_space,
        constraints=constraints  # Add constraints
    )
    
    # Extract and process results
    results = process_optimization_results(opt_results, opt_manager, symbol, output_dir)
    
    # Check if we got valid results
    best_params = opt_results.get('best_params', {})
    best_score = opt_results.get('best_score', 0)
    
    # Validate parameters
    if not best_params or best_score <= 0:
        logger.warning("No optimal parameters found or all performed poorly.")
        
        # Try to find best parameters from all results
        all_results = opt_results.get('all_results', [])
        if all_results:
            # Find best valid parameters (where fast < slow)
            valid_results = [r for r in all_results 
                            if r.get('params', {}).get('fast_window', 0) < 
                               r.get('params', {}).get('slow_window', float('inf'))]
            
            if valid_results:
                # Sort by score
                valid_results.sort(key=lambda x: x.get('score', float('-inf')), reverse=True)
                best_params = valid_results[0].get('params', {})
                best_score = valid_results[0].get('score', 0)
                logger.info(f"Found alternative best parameters: {best_params} with score {best_score}")
    
    # Check if we have valid parameters before validation
    if best_params and 'fast_window' in best_params and 'slow_window' in best_params:
        if best_params['fast_window'] < best_params['slow_window']:
            logger.info(f"Validating best parameter set: {best_params}")
            
            # Create strategy with best params for validation
            invert_signals = best_params.get('invert_signals', False)
            best_strategy = InvertibleMAStrategy(
                name="ma_crossover_best",
                symbols=[symbol],
                **best_params
            )
            
            # Register this strategy for validation
            opt_manager.register_target("ma_crossover_best", best_strategy)
            
            # Ensure grid optimizer is registered for validation with the name "grid"
            if "grid" not in opt_manager.optimizers:
                grid_optimizer = GridSearchOptimizer()
                opt_manager.register_optimizer("grid", grid_optimizer)

            # Run validation to confirm results consistency
            validator = OptimizationValidator(opt_manager, data_handler)
            validation_results = validator.validate_component(
                component_name="ma_crossover_best",
                param_space={k: [v] for k, v in best_params.items()},
                evaluator_name="sharpe_ratio",
                constraints=constraints  # Add constraints
            )
            
            logger.info(f"Validation passed: {validation_results.get('validation_passed', False)}")
            
            # Log if the best strategy uses inverted signals
            if 'invert_signals' in best_params:
                if best_params['invert_signals']:
                    logger.info("Best strategy uses INVERTED signals (reverses traditional MA crossover rules)")
                else:
                    logger.info("Best strategy uses traditional MA crossover rules")
        else:
            logger.warning(f"Best parameters are invalid: fast_window ({best_params['fast_window']}) "
                         f"must be less than slow_window ({best_params['slow_window']})")
    else:
        logger.warning("No valid parameters found for validation")
    
    return results

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

    data_stats['frequency'] = 'minute'
        
    # Set parameter space based on data frequency and size
    if frequency == 'minute':
        # For minute data, use smaller window ranges
        fast_windows = [2, 3, 5, 8, 13, 21]
        
        # Limit slow window based on data size
        max_slow = min(100, bars_count // 5)  # Use at most 1/5 of available bars
        
        # Generate slow windows using Fibonacci sequence up to max_slow
        slow_windows = []
        fib_seq = [5, 8, 13, 21, 34, 55, 89]
        for fib in fib_seq:
            if fib <= max_slow:
                slow_windows.append(fib)
            else:
                break
                
        # Add a few larger values if data permits
        if max_slow > 89:
            slow_windows.extend(list(range(100, max_slow + 1, 20)))
            
    elif frequency == 'hourly':
        # For hourly data
        fast_windows = [3, 5, 8, 13, 21]
        
        # Limit slow window based on data size
        max_slow = min(72, bars_count // 4)
        slow_windows = list(range(12, max_slow + 1, 12))
        
    elif frequency == 'daily':
        # For daily data
        fast_windows = [5, 10, 15, 20, 25, 30, 40, 50]
        
        # Limit slow window based on data size
        max_slow = min(200, bars_count // 3)
        slow_windows = list(range(20, max_slow + 1, 20))
        
    else:  # weekly or longer
        # For weekly or longer data
        fast_windows = [4, 6, 8, 10, 12]
        
        # Limit slow window based on data size
        max_slow = min(52, bars_count // 2)
        slow_windows = list(range(10, max_slow + 1, 6))
    
    # Filter slow windows to ensure they are all greater than the largest fast window plus the minimum gap
    max_fast = max(fast_windows) if fast_windows else 0
    slow_windows = [sw for sw in slow_windows if sw > max_fast + min_gap]
    
    # If no valid slow windows remain, create some
    if not slow_windows:
        start = max_fast + min_gap + 1
        end = min(start + 100, bars_count // 2)
        step = min_gap
        slow_windows = list(range(start, end, step))
    
    # Create parameter space dictionary
    param_space = {
        'fast_window': fast_windows,
        'slow_window': slow_windows,
        'price_key': ['close']  # Can add 'open', 'high', 'low' if needed
    }
    
    # Log parameter space
    logger.info("Parameter space for optimization:")
    logger.info(f"- Fast windows: {fast_windows}")
    logger.info(f"- Slow windows: {slow_windows}")
    logger.info(f"- Total parameter combinations: {len(fast_windows) * len(slow_windows) * len(param_space['price_key'])}")
    logger.info(f"- All combinations satisfy fast_window < slow_window with min gap of {min_gap}")
    
    return param_space

# The following function helps generate a more detailed report for inverting signals
def generate_optimization_report(opt_results, symbol):
    """
    Generate a comprehensive report of optimization results.
    
    Args:
        opt_results: Results from optimization
        symbol: Symbol being traded
        
    Returns:
        str: Markdown report
    """
    # Extract results
    best_params = opt_results.get('best_params', {})
    best_score = opt_results.get('best_score', 0)
    all_results = opt_results.get('all_results', [])
    
    # Check if we tested inverting signals
    tested_inverse = any('invert_signals' in r.get('params', {}) for r in all_results if 'params' in r)
    
    # Sort results by score (should already be sorted, but just to be sure)
    sorted_results = sorted(
        all_results, 
        key=lambda x: x.get('score', float('-inf')), 
        reverse=True
    )
    
    # Create report
    report = ["# Moving Average Crossover Strategy Optimization Report\n"]
    
    # Add summary
    report.append("## Summary\n")
    report.append(f"- Symbol: {symbol}")
    report.append(f"- Total parameter combinations tested: {len(all_results)}")
    report.append(f"- Best Sharpe ratio: {best_score:.4f}")
    
    # Add information about signal inversion if tested
    if tested_inverse and 'invert_signals' in best_params:
        if best_params['invert_signals']:
            report.append(f"- Best strategy uses **INVERTED signals** (Buy when fast MA crosses below slow MA)")
        else:
            report.append(f"- Best strategy uses **traditional signals** (Buy when fast MA crosses above slow MA)")
    
    # Extract additional metrics from best result if available
    if sorted_results and 'metrics' in sorted_results[0]:
        metrics = sorted_results[0]['metrics']
        if 'total_return' in metrics:
            report.append(f"- Best total return: {metrics['total_return']:.4f}%")
        if 'max_drawdown' in metrics:
            report.append(f"- Max drawdown: {metrics['max_drawdown']:.4f}%")
        if 'total_pnl' in metrics:
            report.append(f"- Total P&L: ${metrics['total_pnl']:.2f}")
    
    report.append("")
    
    # Add top results table
    report.append("## Top Results\n")
    
    # Table header with invert_signals if tested
    if tested_inverse:
        report.append("| Rank | Fast Window | Slow Window | Price | Invert Signals | Sharpe | Return (%) | P&L ($) | Max DD (%) | Trades | Win Rate (%) |")
        report.append("|------|------------|------------|-------|---------------|--------|------------|---------|------------|--------|--------------|")
    else:
        report.append("| Rank | Fast Window | Slow Window | Price | Sharpe | Return (%) | P&L ($) | Max DD (%) | Trades | Win Rate (%) |")
        report.append("|------|------------|------------|-------|--------|------------|---------|------------|--------|--------------|")
    
    # Add rows for top 20 results
    for i, result in enumerate(sorted_results[:20], 1):
        params = result.get('params', {})
        score = result.get('score', 0)
        metrics = result.get('metrics', {})
        
        if tested_inverse:
            invert = "Yes" if params.get('invert_signals', False) else "No"
            report.append(
                f"| {i} | " +
                f"{params.get('fast_window', '')} | " +
                f"{params.get('slow_window', '')} | " +
                f"{params.get('price_key', '')} | " +
                f"{invert} | " +
                f"{score:.4f} | " +
                f"{metrics.get('total_return', 0):.4f} | " +
                f"{metrics.get('total_pnl', 0):.2f} | " +
                f"{metrics.get('max_drawdown', 0):.4f} | " +
                f"{metrics.get('trade_count', 0)} | " +
                f"{metrics.get('win_rate', 0):.4f} |"
            )
        else:
            report.append(
                f"| {i} | " +
                f"{params.get('fast_window', '')} | " +
                f"{params.get('slow_window', '')} | " +
                f"{params.get('price_key', '')} | " +
                f"{score:.4f} | " +
                f"{metrics.get('total_return', 0):.4f} | " +
                f"{metrics.get('total_pnl', 0):.2f} | " +
                f"{metrics.get('max_drawdown', 0):.4f} | " +
                f"{metrics.get('trade_count', 0)} | " +
                f"{metrics.get('win_rate', 0):.4f} |"
            )
    
    report.append("")
    
    # Add detailed analysis of best result
    report.append("## Best Parameter Set Analysis\n")
    report.append(f"- Fast window: {best_params.get('fast_window', '')}")
    report.append(f"- Slow window: {best_params.get('slow_window', '')}")
    report.append(f"- Price data: {best_params.get('price_key', '')}")
    if tested_inverse and 'invert_signals' in best_params:
        report.append(f"- Invert signals: {'Yes' if best_params.get('invert_signals') else 'No'}")
    report.append("")
    
    # If we tested inverting signals, add comparison section
    if tested_inverse:
        report.append("## Normal vs. Inverted Signal Performance\n")
        
        # Group results by inversion status
        normal_results = [r for r in all_results if 'params' in r and not r['params'].get('invert_signals', False)]
        inverted_results = [r for r in all_results if 'params' in r and r['params'].get('invert_signals', False)]
        
        # Calculate average performance
        if normal_results:
            normal_avg_score = sum(r.get('score', 0) for r in normal_results) / len(normal_results)
            normal_best = max(normal_results, key=lambda x: x.get('score', float('-inf')))
            normal_best_score = normal_best.get('score', 0)
        else:
            normal_avg_score = 0
            normal_best_score = 0
            
        if inverted_results:
            inverted_avg_score = sum(r.get('score', 0) for r in inverted_results) / len(inverted_results)
            inverted_best = max(inverted_results, key=lambda x: x.get('score', float('-inf')))
            inverted_best_score = inverted_best.get('score', 0)
        else:
            inverted_avg_score = 0
            inverted_best_score = 0
        
        # Create comparison table
        report.append("| Signal Type | Combinations Tested | Avg Sharpe Ratio | Best Sharpe Ratio |")
        report.append("|------------|---------------------|------------------|-------------------|")
        report.append(f"| Normal | {len(normal_results)} | {normal_avg_score:.4f} | {normal_best_score:.4f} |")
        report.append(f"| Inverted | {len(inverted_results)} | {inverted_avg_score:.4f} | {inverted_best_score:.4f} |")
        
        report.append("\n### Interpretation\n")
        if normal_best_score > inverted_best_score:
            report.append("The traditional MA crossover signals (buy when fast MA crosses above slow MA) performed better for this dataset.")
        elif inverted_best_score > normal_best_score:
            report.append("The inverted MA crossover signals (buy when fast MA crosses below slow MA) performed better for this dataset.")
        else:
            report.append("Both traditional and inverted signals showed similar performance.")
        
        # Add explanation of what inverted signals mean
        report.append("\n**Signal Interpretation:**")
        report.append("- **Traditional Signals**: Buy when the fast MA crosses above the slow MA (trend-following)")
        report.append("- **Inverted Signals**: Buy when the fast MA crosses below the slow MA (mean-reversion)")
        
        report.append("")
    
    # Add metrics if available
    if sorted_results and 'metrics' in sorted_results[0]:
        metrics = sorted_results[0]['metrics']
        report.append("### Performance Metrics")
        report.append(f"- Sharpe ratio: {best_score:.4f}")
        if 'total_return' in metrics:
            report.append(f"- Total return: {metrics['total_return']:.4f}%")
        if 'total_pnl' in metrics:
            report.append(f"- Total P&L: ${metrics['total_pnl']:.2f}")
        if 'max_drawdown' in metrics:
            report.append(f"- Max drawdown: {metrics['max_drawdown']:.4f}%")
        if 'sortino_ratio' in metrics:
            report.append(f"- Sortino ratio: {metrics['sortino_ratio']:.4f}")
        if 'calmar_ratio' in metrics:
            report.append(f"- Calmar ratio: {metrics['calmar_ratio']:.4f}")
        report.append("")
        
        # Add trading statistics
        report.append("### Trading Statistics")
        if 'trade_count' in metrics:
            report.append(f"- Total trades: {metrics['trade_count']}")
        if 'win_rate' in metrics:
            report.append(f"- Win rate: {metrics['win_rate']:.4f}%")
        if 'profit_factor' in metrics:
            report.append(f"- Profit factor: {metrics['profit_factor']:.4f}")
        if 'avg_win' in metrics:
            report.append(f"- Average win: ${metrics['avg_win']:.2f}")
        if 'avg_loss' in metrics:
            report.append(f"- Average loss: ${metrics['avg_loss']:.2f}")
    
    # Add parameter sensitivity analysis if we have enough results
    if len(all_results) > 10:
        report.append("\n## Parameter Sensitivity Analysis\n")
        
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
                    f"{avg_return:.4f} |"
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
                    f"{avg_return:.4f} |"
                )
        
        # If we tested inversion, add analysis for that
        if tested_inverse:
            report.append("\n### Signal Inversion Sensitivity\n")
            report.append("| Invert Signals | Avg Sharpe | Best Sharpe | Worst Sharpe | Avg Return (%) |")
            report.append("|---------------|------------|-------------|--------------|----------------|")
            
            for invert in [False, True]:
                # Get all results with this inversion setting
                results_with_invert = [r for r in all_results if 'params' in r and r['params'].get('invert_signals', False) == invert]
                
                if results_with_invert:
                    # Calculate statistics
                    sharpes = [r.get('score', 0) for r in results_with_invert]
                    returns = [r.get('metrics', {}).get('total_return', 0) for r in results_with_invert if 'metrics' in r]
                    
                    avg_sharpe = sum(sharpes) / len(sharpes)
                    best_sharpe = max(sharpes)
                    worst_sharpe = min(sharpes)
                    avg_return = sum(returns) / len(returns) if returns else 0
                    
                    report.append(
                        f"| {'Yes' if invert else 'No'} | " +
                        f"{avg_sharpe:.4f} | " +
                        f"{best_sharpe:.4f} | " +
                        f"{worst_sharpe:.4f} | " +
                        f"{avg_return:.4f} |"
                    )
    
    # Add conclusion
    report.append("\n## Conclusion\n")
    
    # Add recommendations based on results
    top_return = None
    if sorted_results and 'metrics' in sorted_results[0]:
        top_return = sorted_results[0]['metrics'].get('total_return', 0)
    
    if top_return is not None and top_return > 0:
        # Strategy is profitable with optimal parameters
        report.append("The Moving Average Crossover strategy shows positive results with the optimal parameters:")
        report.append(f"- Fast window: {best_params.get('fast_window', '')}")
        report.append(f"- Slow window: {best_params.get('slow_window', '')}")
        report.append(f"- Price data: {best_params.get('price_key', '')}")
        if tested_inverse and 'invert_signals' in best_params:
            report.append(f"- Invert signals: {'Yes' if best_params.get('invert_signals') else 'No'}")
        report.append("")
        report.append("These parameters achieved the highest Sharpe ratio during testing.")
    else:
        # Strategy did not show profitable results
        report.append("**Warning: No profitable parameter combinations were found for this strategy on this dataset.**")
        report.append("")
        report.append("Recommendations:")
        report.append("1. Try a different strategy that may be better suited to this dataset")
        report.append("2. Use a larger dataset with more price history")
        report.append("3. Consider testing on a different timeframe or market condition")
        report.append("")
        report.append("The least unprofitable parameters are:")
        report.append(f"- Fast window: {best_params.get('fast_window', '')}")
        report.append(f"- Slow window: {best_params.get('slow_window', '')}")
        report.append(f"- Price data: {best_params.get('price_key', '')}")
        if tested_inverse and 'invert_signals' in best_params:
            report.append(f"- Invert signals: {'Yes' if best_params.get('invert_signals') else 'No'}")
    
    # Add timestamp
    now = datetime.datetime.now()
    report.append(f"\n*Report generated on {now.strftime('%Y-%m-%d %H:%M:%S')}*")
    
    return "\n".join(report)


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
            stats['avg_interval'] = (df.index.max() - df.index.min()) / (len(df) - 1)
            
            # Determine data frequency
            if stats['avg_interval'] < pd.Timedelta(minutes=5):
                stats['frequency'] = 'minute'
            elif stats['avg_interval'] < pd.Timedelta(hours=1):
                stats['frequency'] = 'hourly'
            elif stats['avg_interval'] < pd.Timedelta(days=1):
                stats['frequency'] = 'daily'
            else:
                stats['frequency'] = 'weekly_or_longer'
    
    # Calculate price statistics
    if 'close' in df.columns:
        stats['avg_price'] = df['close'].mean()
        stats['min_price'] = df['close'].min()
        stats['max_price'] = df['close'].max()
        stats['volatility'] = df['close'].std()
        
        # Calculate returns
        df['return'] = df['close'].pct_change() * 100
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
    if 'volatility' in stats:
        logger.info(f"- Price volatility (std): {stats['volatility']:.4f}")
    
    # Check if data is sufficient for MA strategy
    recommended_min_bars = 200  # For 200-period MA
    if stats['bars'] < recommended_min_bars:
        logger.warning(f"Dataset has only {stats['bars']} bars. "
                     f"Recommended minimum is {recommended_min_bars} bars for MA strategies.")
        logger.warning("Consider using smaller MA windows or getting more data.")
    
    return stats




def create_optimization_manager(strategy, data_handler, initial_cash):
    """
    Create and configure the optimization manager.
    
    Args:
        strategy: Strategy to optimize
        data_handler: Data handler with market data2
        initial_cash: Initial portfolio cash
        
    Returns:
        OptimizationManager: Configured optimization manager
    """
    # Create optimization manager
    opt_manager = OptimizationManager(name="ma_crossover_optimizer")
    
    # Register the strategy as an optimization target
    opt_manager.register_target("ma_crossover_base", strategy)
    
    # Register optimizers (grid search is already registered by default)
    # Add our own grid search optimizer to be safe
    opt_manager.register_optimizer("grid", GridSearchOptimizer())
    
    # Create custom evaluation function that uses data_handler
    def evaluate_ma_crossover(component, **kwargs):
        return evaluate_backtest(
            component=component,
            data_handler=data_handler,
            initial_cash=initial_cash,
            metric='sharpe_ratio'
        )
    
    # Register evaluation functions
    opt_manager.register_evaluator("sharpe_ratio", evaluate_ma_crossover)
    
    # Add evaluators for other metrics
    def evaluate_total_return(component, **kwargs):
        return evaluate_backtest(
            component=component,
            data_handler=data_handler,
            initial_cash=initial_cash,
            metric='total_return'
        )
        
    def evaluate_max_drawdown(component, **kwargs):
        # Negate the drawdown because optimization maximizes the metric
        return -evaluate_backtest(
            component=component,
            data_handler=data_handler,
            initial_cash=initial_cash,
            metric='max_drawdown'
        )
        
    def evaluate_win_rate(component, **kwargs):
        return evaluate_backtest(
            component=component,
            data_handler=data_handler,
            initial_cash=initial_cash,
            metric='win_rate'
        )
    
    # Register additional evaluators
    opt_manager.register_evaluator("total_return", evaluate_total_return)
    opt_manager.register_evaluator("max_drawdown", evaluate_max_drawdown)
    opt_manager.register_evaluator("win_rate", evaluate_win_rate)
    
    return opt_manager

def process_optimization_results(opt_results, opt_manager, symbol, output_dir):
    """
    Process optimization results and generate reports.
    
    Args:
        opt_results: Results from optimization
        opt_manager: Optimization manager
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
    
    # Find the best result object with its detailed metrics
    best_result_obj = None
    for result in all_results:
        if result.get('params') == best_params:
            best_result_obj = result
            break
    
    # Generate report
    report = ["# Moving Average Crossover Strategy Optimization Report\n"]
    
    # Add summary
    report.append("## Summary\n")
    report.append(f"- Symbol: {symbol}")
    report.append(f"- Total parameter combinations tested: {len(all_results)}")
    report.append(f"- Best Sharpe ratio: {best_score:.4f}")
    
    # Add detailed performance metrics if available
    if best_result_obj and 'metrics' in best_result_obj:
        metrics = best_result_obj['metrics']
        report.append(f"- Total Return: {metrics.get('total_return', 0):.6f}%")
        report.append(f"- Total P&L: ${metrics.get('total_pnl', 0):.2f}")
        report.append(f"- Max Drawdown: {metrics.get('max_drawdown', 0):.4f}%")
        report.append(f"- Total Trades: {metrics.get('trade_count', 0)}")
        report.append(f"- Win Rate: {metrics.get('win_rate', 0):.2f}%")
        report.append(f"- Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    
    # Add detailed trade statistics if available
    if best_result_obj and 'trade_statistics' in best_result_obj:
        stats = best_result_obj['trade_statistics']
        report.append("\n## Trade Statistics\n")
        report.append(f"- Total Trades: {stats.get('total_trades', 0)}")
        report.append(f"- Winning Trades: {stats.get('win_count', 0)}")
        report.append(f"- Losing Trades: {stats.get('loss_count', 0)}")
        report.append(f"- Win Rate: {stats.get('win_rate', 0):.2f}%")
        report.append(f"- Average Win: ${stats.get('avg_win', 0):.2f}")
        report.append(f"- Average Loss: ${stats.get('avg_loss', 0):.2f}")
        report.append(f"- Largest Win: ${stats.get('max_win', 0):.2f}")
        report.append(f"- Largest Loss: ${stats.get('max_loss', 0):.2f}")
        report.append(f"- Average Trade P&L: ${stats.get('average_pnl', 0):.4f}")
    
    report.append("")
    
    # Add top results table
    report.append("## Top Results\n")
    
    # Table header
    report.append("| Rank | Fast Window | Slow Window | Price | Sharpe | Return (%) | P&L ($) | Max DD (%) | Trades | Win Rate (%) |")
    report.append("|------|------------|------------|-------|--------|------------|---------|------------|--------|--------------|")
    
    # Sort results by score (should already be sorted, but just to be sure)
    sorted_results = sorted(
        all_results, 
        key=lambda x: x.get('score', float('-inf')), 
        reverse=True
    )
    
    # Add rows for all results
    for i, result in enumerate(sorted_results, 1):
        params = result.get('params', {})
        score = result.get('score', 0)
        metrics = result.get('metrics', {})
        
        # Format with high precision to show small values
        report.append(
            f"| {i} | " +
            f"{params.get('fast_window', '')} | " +
            f"{params.get('slow_window', '')} | " +
            f"{params.get('price_key', '')} | " +
            f"{score:.4f} | " +
            f"{metrics.get('total_return', 0):.6f} | " +
            f"{metrics.get('total_pnl', 0):.2f} | " +
            f"{metrics.get('max_drawdown', 0):.6f} | " +
            f"{metrics.get('trade_count', 0)} | " +
            f"{metrics.get('win_rate', 0):.4f} |"
        )
    
    report.append("")
    
    # Add detailed analysis of best result
    report.append("## Best Parameter Set Analysis\n")
    report.append(f"- Fast window: {best_params.get('fast_window', '')}")
    report.append(f"- Slow window: {best_params.get('slow_window', '')}")
    report.append(f"- Price data: {best_params.get('price_key', '')}")
    
    # Add additional details if available
    if best_result_obj and 'metrics' in best_result_obj:
        metrics = best_result_obj['metrics']
        report.append("\n### Performance Metrics")
        report.append(f"- Sharpe ratio: {best_score:.4f}")
        report.append(f"- Total return: {metrics.get('total_return', 0):.6f}%")
        report.append(f"- Total P&L: ${metrics.get('total_pnl', 0):.2f}")
        report.append(f"- Max drawdown: {metrics.get('max_drawdown', 0):.6f}%")
        
        if 'sortino_ratio' in metrics:
            report.append(f"- Sortino ratio: {metrics.get('sortino_ratio', 0):.4f}")
        if 'calmar_ratio' in metrics:
            report.append(f"- Calmar ratio: {metrics.get('calmar_ratio', 0):.4f}")
    
    report.append("")
    
    # Add parameter sensitivity analysis if we have enough results
    if len(all_results) > 1:
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
                    f"{avg_return:.6f} |"
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
                    f"{avg_return:.6f} |"
                )
    
    # Add conclusion
    report.append("\n## Conclusion\n")
    
    # Add recommendations based on results
    top_return = None
    if sorted_results and 'metrics' in sorted_results[0]:
        top_return = sorted_results[0]['metrics'].get('total_return', 0)
    
    if top_return is not None and top_return > 0:
        # Strategy is profitable with optimal parameters
        report.append("The Moving Average Crossover strategy shows positive results with the optimal parameters:")
        report.append(f"- Fast window: {best_params.get('fast_window', '')}")
        report.append(f"- Slow window: {best_params.get('slow_window', '')}")
        report.append(f"- Price data: {best_params.get('price_key', '')}")
        report.append("")
        report.append("These parameters achieved the highest Sharpe ratio during testing.")
    else:
        # Strategy did not show profitable results
        report.append("**Warning: No profitable parameter combinations were found for this strategy on this dataset.**")
        report.append("")
        report.append("Recommendations:")
        report.append("1. Try a different strategy that may be better suited to this dataset")
        report.append("2. Use a larger dataset with more price history")
        report.append("3. Consider testing on a different timeframe or market condition")
        report.append("")
        report.append("The least unprofitable parameters are:")
        report.append(f"- Fast window: {best_params.get('fast_window', '')}")
        report.append(f"- Slow window: {best_params.get('slow_window', '')}")
        report.append(f"- Price data: {best_params.get('price_key', '')}")
    
    # Add timestamp
    import datetime
    now = datetime.datetime.now()
    report.append(f"\n*Report generated on {now.strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Save report to file
    report_path = os.path.join(output_dir, f"ma_crossover_{symbol}_optimization_report.md")
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
    ax_sharpe.set_title(f'{symbol}: MA Crossover Sharpe Ratio by Window Size', fontsize=16)
    
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
    sharpe_path = os.path.join(output_dir, f"{symbol}_ma_sharpe_heatmap.png")
    fig_sharpe.tight_layout()
    fig_sharpe.savefig(sharpe_path, dpi=150, bbox_inches='tight')
    logger.info(f"Sharpe ratio heatmap saved to {sharpe_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize MA Crossover Strategy Parameters')
    parser.add_argument('--data', type=str, required=True, help='Path to price data CSV file')
    parser.add_argument('--output', type=str, default=None, help='Directory to save results')
    parser.add_argument('--cash', type=float, default=10000.0, help='Initial portfolio cash')
    parser.add_argument('--inverse', action='store_true', help='Also test inverting signals')
    
    args = parser.parse_args()
    
    # Call the optimization function with the inverse parameter
    optimize_ma_crossover(
        data_path=args.data, 
        output_dir=args.output, 
        initial_cash=args.cash,
        test_inverse=args.inverse  # Pass the inverse flag
    )

