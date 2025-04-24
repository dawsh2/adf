"""
Regime Configuration Optimizer

This script optimizes the regime configuration (which signals are allowed in which regimes)
for a regime-filtered MA crossover strategy.
"""
import logging
import pandas as pd
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple

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

# Import backtest utilities
from src.execution.backtest.backtest import run_backtest
from src.analytics.performance import PerformanceAnalytics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def optimize_regime_configuration(data_path, output_dir=None, initial_cash=10000.0, 
                                strategy_params=None):
    """
    Optimize the regime configuration for a regime-filtered strategy.
    
    Args:
        data_path: Path to price data CSV file
        output_dir: Directory for saving results
        initial_cash: Initial portfolio cash
        strategy_params: Pre-determined strategy parameters (fast_window, slow_window)
        
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
    
    # Use provided strategy parameters or defaults
    if strategy_params is None:
        strategy_params = {'fast_window': 10, 'slow_window': 30}
    
    # Create regime detector
    regime_detector = RegimeDetectorFactory.create_detector(detector_type='enhanced')
    
    # Define base configuration
    base_config = {
        # Default configurations allow trading in all regimes
        MarketRegime.UPTREND: [SignalEvent.BUY, SignalEvent.SELL],        
        MarketRegime.DOWNTREND: [SignalEvent.BUY, SignalEvent.SELL],      
        MarketRegime.SIDEWAYS: [SignalEvent.BUY, SignalEvent.SELL],       
        MarketRegime.VOLATILE: [SignalEvent.BUY, SignalEvent.SELL],       
        MarketRegime.UNKNOWN: [SignalEvent.BUY, SignalEvent.SELL]         
    }
    
    logger.info("Generating regime configurations to test...")
    
    # Generate all possible configurations
    configurations = generate_configurations()
    logger.info(f"Generated {len(configurations)} configurations to test")
    
    # Run backtest for each configuration
    results = []
    
    for i, config in enumerate(configurations, 1):
        logger.info(f"Testing configuration {i}/{len(configurations)}")
        
        # Create strategy with this configuration
        strategy = RegimeFilteredMAStrategy(
            name=f"regime_config_{i}",
            symbols=[symbol],
            fast_window=strategy_params['fast_window'],
            slow_window=strategy_params['slow_window'],
            regime_detector=regime_detector,
            allowed_regimes=config
        )
        
        # Reset data handler
        data_handler.reset()
        
        # Run backtest
        equity_curve, trades = run_backtest(
            component=strategy,
            data_handler=data_handler,
            initial_cash=initial_cash
        )
        
        # Calculate metrics
        metrics = PerformanceAnalytics.calculate_metrics(equity_curve, trades)
        
        # Get regime stats
        regime_stats = strategy.get_regime_stats()
        
        # Create a summary of this configuration
        config_summary = {
            'uptrend': 'Buy' if SignalEvent.BUY in config[MarketRegime.UPTREND] else '-',
            'uptrend_sell': 'Sell' if SignalEvent.SELL in config[MarketRegime.UPTREND] else '-',
            'downtrend': 'Buy' if SignalEvent.BUY in config[MarketRegime.DOWNTREND] else '-',
            'downtrend_sell': 'Sell' if SignalEvent.SELL in config[MarketRegime.DOWNTREND] else '-',
            'sideways': 'Buy' if SignalEvent.BUY in config[MarketRegime.SIDEWAYS] else '-',
            'sideways_sell': 'Sell' if SignalEvent.SELL in config[MarketRegime.SIDEWAYS] else '-',
            'volatile': 'Buy' if SignalEvent.BUY in config[MarketRegime.VOLATILE] else '-',
            'volatile_sell': 'Sell' if SignalEvent.SELL in config[MarketRegime.VOLATILE] else '-'
        }
        
        # Store result
        results.append({
            'config_id': i,
            'config': config,
            'config_summary': config_summary,
            'metrics': metrics,
            'regime_stats': regime_stats,
            'filtered_signals': regime_stats['filtered_signals'],
            'passed_signals': regime_stats['passed_signals'],
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'total_return': metrics.get('total_return', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'trade_count': metrics.get('trade_count', 0),
            'win_rate': metrics.get('win_rate', 0)
        })
        
        logger.info(f"Configuration {i}: Return={metrics.get('total_return', 0):.2f}%, "
                   f"Sharpe={metrics.get('sharpe_ratio', 0):.2f}, "
                   f"MaxDD={metrics.get('max_drawdown', 0):.2f}%, "
                   f"Trades={metrics.get('trade_count', 0)}")
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['sharpe_ratio'])
    
    logger.info("\n===== BEST REGIME CONFIGURATION =====")
    logger.info(f"Configuration #{best_result['config_id']}")
    logger.info(f"Sharpe Ratio: {best_result['sharpe_ratio']:.4f}")
    logger.info(f"Total Return: {best_result['total_return']:.2f}%")
    logger.info(f"Max Drawdown: {best_result['max_drawdown']:.2f}%")
    logger.info(f"Trades: {best_result['trade_count']}")
    logger.info(f"Win Rate: {best_result['win_rate']:.2f}%")
    
    logger.info("\nRegime Configuration:")
    for regime, signals in best_result['config'].items():
        logger.info(f"- {regime.value}: {[s for s in signals]}")
    
    # Run a final backtest with the best configuration
    logger.info("\nRunning final backtest with best configuration...")
    
    # Create strategy with best configuration
    best_detector = RegimeDetectorFactory.create_detector(detector_type='enhanced')
    
    best_strategy = RegimeFilteredMAStrategy(
        name="best_regime_config",
        symbols=[symbol],
        fast_window=strategy_params['fast_window'],
        slow_window=strategy_params['slow_window'],
        regime_detector=best_detector,
        allowed_regimes=best_result['config']
    )
    
    # Reset data handler
    data_handler.reset()
    
    # Run backtest
    equity_curve, trades = run_backtest(
        component=best_strategy,
        data_handler=data_handler,
        initial_cash=initial_cash
    )
    
    # Generate report
    generate_report(results, best_result, symbol, output_dir)
    
    # Plot equity curve
    plot_equity_curve(equity_curve, trades, symbol, strategy_params, best_result['config'], output_dir)
    
    return {
        'best_config': best_result['config'],
        'best_metrics': best_result['metrics'],
        'all_results': results,
        'equity_curve': equity_curve,
        'trades': trades
    }

def generate_configurations():
    """
    Generate different regime configurations to test.
    
    Returns:
        list: List of regime configurations
    """
    # Set all possible signal combinations for each regime
    signal_options = [
        [],  # No signals allowed
        [SignalEvent.BUY],  # Only buy signals
        [SignalEvent.SELL],  # Only sell signals
        [SignalEvent.BUY, SignalEvent.SELL]  # Both signals allowed
    ]
    
    # Create meaningful configurations instead of testing all combinations
    # (testing all would be 4^5 = 1024 configurations)
    
    configs = []
    
    # Configuration 1: Default - traditional trend following
    configs.append({
        MarketRegime.UPTREND: [SignalEvent.BUY],             # Only buy in uptrends
        MarketRegime.DOWNTREND: [SignalEvent.SELL],          # Only sell in downtrends
        MarketRegime.SIDEWAYS: [SignalEvent.BUY, SignalEvent.SELL],  # Allow both in sideways
        MarketRegime.VOLATILE: [],                           # No trading in volatile markets
        MarketRegime.UNKNOWN: [SignalEvent.BUY, SignalEvent.SELL]    # Allow both if unknown
    })
    
    # Configuration 2: Contrarian - buy dips, sell rallies
    configs.append({
        MarketRegime.UPTREND: [SignalEvent.SELL],            # Sell in uptrends (take profits)
        MarketRegime.DOWNTREND: [SignalEvent.BUY],           # Buy in downtrends (buy dips)
        MarketRegime.SIDEWAYS: [SignalEvent.BUY, SignalEvent.SELL],  # Allow both in sideways
        MarketRegime.VOLATILE: [],                           # No trading in volatile markets
        MarketRegime.UNKNOWN: [SignalEvent.BUY, SignalEvent.SELL]    # Allow both if unknown
    })
    
    # Configuration 3: Conservative - only trade in favorable conditions
    configs.append({
        MarketRegime.UPTREND: [SignalEvent.BUY],             # Only buy in uptrends
        MarketRegime.DOWNTREND: [SignalEvent.SELL],          # Only sell in downtrends
        MarketRegime.SIDEWAYS: [],                           # No trading in sideways
        MarketRegime.VOLATILE: [],                           # No trading in volatile markets
        MarketRegime.UNKNOWN: []                             # No trading if unknown
    })
    
    # Configuration 4: Aggressive - trade everything except volatile
    configs.append({
        MarketRegime.UPTREND: [SignalEvent.BUY, SignalEvent.SELL],  # All signals in uptrends
        MarketRegime.DOWNTREND: [SignalEvent.BUY, SignalEvent.SELL], # All signals in downtrends
        MarketRegime.SIDEWAYS: [SignalEvent.BUY, SignalEvent.SELL],  # All signals in sideways
        MarketRegime.VOLATILE: [],                                  # No trading in volatile markets
        MarketRegime.UNKNOWN: [SignalEvent.BUY, SignalEvent.SELL]   # All signals if unknown
    })
    
    # Configuration 5: Only trade in sideways markets
    configs.append({
        MarketRegime.UPTREND: [],                            # No trading in uptrends
        MarketRegime.DOWNTREND: [],                          # No trading in downtrends
        MarketRegime.SIDEWAYS: [SignalEvent.BUY, SignalEvent.SELL],  # All signals in sideways
        MarketRegime.VOLATILE: [],                           # No trading in volatile markets
        MarketRegime.UNKNOWN: []                             # No trading if unknown
    })
    
    # Configuration 6: Uptrend only
    configs.append({
        MarketRegime.UPTREND: [SignalEvent.BUY, SignalEvent.SELL],   # All signals in uptrends
        MarketRegime.DOWNTREND: [],                                  # No trading in downtrends
        MarketRegime.SIDEWAYS: [],                                   # No trading in sideways
        MarketRegime.VOLATILE: [],                                   # No trading in volatile markets
        MarketRegime.UNKNOWN: []                                     # No trading if unknown
    })
    
    # Configuration 7: Downtrend only
    configs.append({
        MarketRegime.UPTREND: [],                                    # No trading in uptrends
        MarketRegime.DOWNTREND: [SignalEvent.BUY, SignalEvent.SELL], # All signals in downtrends
        MarketRegime.SIDEWAYS: [],                                   # No trading in sideways
        MarketRegime.VOLATILE: [],                                   # No trading in volatile markets
        MarketRegime.UNKNOWN: []                                     # No trading if unknown
    })
    
    # Configuration 8: Buy only
    configs.append({
        MarketRegime.UPTREND: [SignalEvent.BUY],                     # Buy in uptrends
        MarketRegime.DOWNTREND: [SignalEvent.BUY],                   # Buy in downtrends
        MarketRegime.SIDEWAYS: [SignalEvent.BUY],                    # Buy in sideways
        MarketRegime.VOLATILE: [],                                   # No trading in volatile markets
        MarketRegime.UNKNOWN: [SignalEvent.BUY]                      # Buy if unknown
    })
    
    # Configuration 9: Sell only
    configs.append({
        MarketRegime.UPTREND: [SignalEvent.SELL],                    # Sell in uptrends
        MarketRegime.DOWNTREND: [SignalEvent.SELL],                  # Sell in downtrends
        MarketRegime.SIDEWAYS: [SignalEvent.SELL],                   # Sell in sideways
        MarketRegime.VOLATILE: [],                                   # No trading in volatile markets
        MarketRegime.UNKNOWN: [SignalEvent.SELL]                     # Sell if unknown
    })
    
    # Add more targeted combinations
    
    # Trade everything including volatile
    configs.append({
        MarketRegime.UPTREND: [SignalEvent.BUY, SignalEvent.SELL],    # All signals in uptrends
        MarketRegime.DOWNTREND: [SignalEvent.BUY, SignalEvent.SELL],  # All signals in downtrends
        MarketRegime.SIDEWAYS: [SignalEvent.BUY, SignalEvent.SELL],   # All signals in sideways
        MarketRegime.VOLATILE: [SignalEvent.BUY, SignalEvent.SELL],   # All signals in volatile
        MarketRegime.UNKNOWN: [SignalEvent.BUY, SignalEvent.SELL]     # All signals if unknown
    })
    
    # Only buy in uptrends, only sell in downtrends
    configs.append({
        MarketRegime.UPTREND: [SignalEvent.BUY],                      # Buy in uptrends
        MarketRegime.DOWNTREND: [SignalEvent.SELL],                   # Sell in downtrends
        MarketRegime.SIDEWAYS: [],                                    # No trading in sideways
        MarketRegime.VOLATILE: [],                                    # No trading in volatile
        MarketRegime.UNKNOWN: []                                      # No trading if unknown
    })
    
    # Buy in uptrends and sideways, sell in downtrends and sideways
    configs.append({
        MarketRegime.UPTREND: [SignalEvent.BUY],                      # Buy in uptrends
        MarketRegime.DOWNTREND: [SignalEvent.SELL],                   # Sell in downtrends
        MarketRegime.SIDEWAYS: [SignalEvent.BUY, SignalEvent.SELL],   # Both in sideways
        MarketRegime.VOLATILE: [],                                    # No trading in volatile
        MarketRegime.UNKNOWN: []                                      # No trading if unknown
    })
    
    # Buy in downtrends (contrarian), sell in uptrends (contrarian)
    configs.append({
        MarketRegime.UPTREND: [SignalEvent.SELL],                     # Sell in uptrends
        MarketRegime.DOWNTREND: [SignalEvent.BUY],                    # Buy in downtrends
        MarketRegime.SIDEWAYS: [],                                    # No trading in sideways
        MarketRegime.VOLATILE: [],                                    # No trading in volatile
        MarketRegime.UNKNOWN: []                                      # No trading if unknown
    })
    
    # Mixed strategy - trend following in some regimes, contrarian in others
    configs.append({
        MarketRegime.UPTREND: [SignalEvent.BUY],                      # Buy in uptrends (trend following)
        MarketRegime.DOWNTREND: [SignalEvent.BUY],                    # Buy in downtrends (contrarian)
        MarketRegime.SIDEWAYS: [SignalEvent.BUY, SignalEvent.SELL],   # Both in sideways
        MarketRegime.VOLATILE: [],                                    # No trading in volatile
        MarketRegime.UNKNOWN: [SignalEvent.BUY]                       # Buy if unknown
    })
    
    configs.append({
        MarketRegime.UPTREND: [SignalEvent.SELL],                     # Sell in uptrends (contrarian)
        MarketRegime.DOWNTREND: [SignalEvent.SELL],                   # Sell in downtrends (trend following)
        MarketRegime.SIDEWAYS: [SignalEvent.BUY, SignalEvent.SELL],   # Both in sideways
        MarketRegime.VOLATILE: [],                                    # No trading in volatile
        MarketRegime.UNKNOWN: [SignalEvent.SELL]                      # Sell if unknown
    })
    
    return configs

def generate_report(results, best_result, symbol, output_dir):
    """
    Generate a detailed report of the regime configuration optimization.
    
    Args:
        results: List of all configuration results
        best_result: Best configuration result
        symbol: Symbol being traded
        output_dir: Directory for saving report
    """
    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Sort results by Sharpe ratio
    sorted_results = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)
    
    report = [
        "# Regime Configuration Optimization Report\n",
        f"**Symbol:** {symbol}",
        f"**Date:** {now}\n",
        
        "## Best Configuration\n",
        "| Regime | Buy Signals | Sell Signals |",
        "|--------|------------|-------------|"
    ]
    
    # Add best configuration details
    for regime, signals in best_result['config'].items():
        buy = "✓" if SignalEvent.BUY in signals else "-"
        sell = "✓" if SignalEvent.SELL in signals else "-"
        report.append(f"| {regime.value} | {buy} | {sell} |")
    
    # Add performance details
    report.append("\n## Performance Metrics\n")
    report.append(f"- Sharpe Ratio: {best_result['sharpe_ratio']:.4f}")
    report.append(f"- Total Return: {best_result['total_return']:.2f}%")
    report.append(f"- Max Drawdown: {best_result['max_drawdown']:.2f}%")
    report.append(f"- Number of Trades: {best_result['trade_count']}")
    report.append(f"- Win Rate: {best_result['win_rate']:.2f}%")
    report.append(f"- Signals Passed: {best_result['passed_signals']}")
    report.append(f"- Signals Filtered: {best_result['filtered_signals']}")
    
    # Add details of all configurations
    report.append("\n## All Configurations\n")
    report.append("| Rank | Uptrend Buy | Uptrend Sell | Downtrend Buy | Downtrend Sell | Sideways Buy | Sideways Sell | Volatile Buy | Volatile Sell | Sharpe | Return | Max DD | Trades |")
    report.append("|------|------------|-------------|--------------|---------------|-------------|--------------|-------------|--------------|--------|--------|--------|--------|")
    
    for i, result in enumerate(sorted_results, 1):
        config = result['config']
        summary = result['config_summary']
        
        # Create table row
        uptrend_buy = "✓" if SignalEvent.BUY in config[MarketRegime.UPTREND] else "-"
        uptrend_sell = "✓" if SignalEvent.SELL in config[MarketRegime.UPTREND] else "-"
        downtrend_buy = "✓" if SignalEvent.BUY in config[MarketRegime.DOWNTREND] else "-"
        downtrend_sell = "✓" if SignalEvent.SELL in config[MarketRegime.DOWNTREND] else "-"
        sideways_buy = "✓" if SignalEvent.BUY in config[MarketRegime.SIDEWAYS] else "-"
        sideways_sell = "✓" if SignalEvent.SELL in config[MarketRegime.SIDEWAYS] else "-"
        volatile_buy = "✓" if SignalEvent.BUY in config[MarketRegime.VOLATILE] else "-"
        volatile_sell = "✓" if SignalEvent.SELL in config[MarketRegime.VOLATILE] else "-"
        
        report.append(
            f"| {i} | {uptrend_buy} | {uptrend_sell} | {downtrend_buy} | {downtrend_sell} | " +
            f"{sideways_buy} | {sideways_sell} | {volatile_buy} | {volatile_sell} | " +
            f"{result['sharpe_ratio']:.4f} | {result['total_return']:.2f}% | {result['max_drawdown']:.2f}% | {result['trade_count']} |"
        )
    
    # Add analysis and observations
    report.append("\n## Analysis and Observations\n")
    
    # Compare different strategies
    trend_following = next((r for r in results if r['config_id'] == 1), None)
    contrarian = next((r for r in results if r['config_id'] == 2), None)
    
    if trend_following and contrarian:
        report.append("### Trend Following vs. Contrarian\n")
        tf_return = trend_following['total_return']
        con_return = contrarian['total_return']
        
        if tf_return > con_return:
            report.append(f"The trend following approach (Configuration #1) outperformed the contrarian approach (Configuration #2) by {tf_return - con_return:.2f}% in total return. This suggests that following the trend is more effective for this symbol during the tested period.")
        elif con_return > tf_return:
            report.append(f"The contrarian approach (Configuration #2) outperformed the trend following approach (Configuration #1) by {con_return - tf_return:.2f}% in total return. This suggests that contrarian trading might be more effective for this symbol during the tested period.")
        else:
            report.append(f"The trend following and contrarian approaches performed equally in terms of total return ({tf_return:.2f}%). The choice between them might depend on other factors like drawdown or risk-adjusted return.")
    
    # Analyze best performing regimes
    report.append("\n### Best Performing Regimes\n")
    
    # Extract results for uptrend-only, downtrend-only, and sideways-only
    uptrend_only = next((r for r in results if r['config_id'] == 6), None)
    downtrend_only = next((r for r in results if r['config_id'] == 7), None)
    sideways_only = next((r for r in results if r['config_id'] == 5), None)
    
    regime_performances = []
    if uptrend_only:
        regime_performances.append(("Uptrend", uptrend_only['total_return']))
    if downtrend_only:
        regime_performances.append(("Downtrend", downtrend_only['total_return']))
    if sideways_only:
        regime_performances.append(("Sideways", sideways_only['total_return']))
    
    # Sort by performance
    regime_performances.sort(key=lambda x: x[1], reverse=True)
    
    if regime_performances:
        # Report on best performing regime
        best_regime, best_return = regime_performances[0]
        report.append(f"Trading only in {best_regime} regimes yielded the highest return among single-regime strategies at {best_return:.2f}%.")
        
        # Compare to combined approach
        if best_result['total_return'] > best_return:
            report.append(f"\nHowever, the optimal combined regime approach (Configuration #{best_result['config_id']}) outperformed the best single-regime approach by {best_result['total_return'] - best_return:.2f}%.")
        else:
            report.append(f"\nInterestingly, the {best_regime}-only approach outperformed the optimal combined regime approach by {best_return - best_result['total_return']:.2f}%.")
    
    # Conclusion
    report.append("\n## Conclusion\n")
    
    # Generate recommendations based on results
    if best_result['sharpe_ratio'] > 1.0 and best_result['total_return'] > 0:
        report.append(f"The optimal regime configuration (Configuration #{best_result['config_id']}) shows promising results with a Sharpe ratio of {best_result['sharpe_ratio']:.2f} and total return of {best_result['total_return']:.2f}%.")
        report.append("\nKey features of this configuration:")
        
        # Highlight key features of the best configuration
        features = []
        if SignalEvent.BUY in best_result['config'][MarketRegime.UPTREND]:
            features.append("Buy signals in uptrends")
        if SignalEvent.SELL in best_result['config'][MarketRegime.UPTREND]:
            features.append("Sell signals in uptrends")
        if SignalEvent.BUY in best_result['config'][MarketRegime.DOWNTREND]:
            features.append("Buy signals in downtrends")
        if SignalEvent.SELL in best_result['config'][MarketRegime.DOWNTREND]:
            features.append("Sell signals in downtrends")
        if SignalEvent.BUY in best_result['config'][MarketRegime.SIDEWAYS]:
            features.append("Buy signals in sideways markets")
        if SignalEvent.SELL in best_result['config'][MarketRegime.SIDEWAYS]:
            features.append("Sell signals in sideways markets")
        if SignalEvent.BUY in best_result['config'][MarketRegime.VOLATILE]:
            features.append("Buy signals in volatile markets")
        if SignalEvent.SELL in best_result['config'][MarketRegime.VOLATILE]:
            features.append("Sell signals in volatile markets")
        
        for feature in features:
            report.append(f"- Allows {feature}")
            
        # Check if it's closer to trend following or contrarian
        is_trend_following = (SignalEvent.BUY in best_result['config'][MarketRegime.UPTREND] and 
                             SignalEvent.SELL in best_result['config'][MarketRegime.DOWNTREND])
        is_contrarian = (SignalEvent.SELL in best_result['config'][MarketRegime.UPTREND] and 
                        SignalEvent.BUY in best_result['config'][MarketRegime.DOWNTREND])
        
        if is_trend_following and not is_contrarian:
            report.append("\nThis configuration follows a primarily trend-following approach.")
        elif is_contrarian and not is_trend_following:
            report.append("\nThis configuration follows a primarily contrarian approach.")
        elif is_trend_following and is_contrarian:
            report.append("\nThis configuration uses a mixed approach, incorporating both trend-following and contrarian elements.")
        else:
            report.append("\nThis configuration uses a specialized approach that is neither purely trend-following nor contrarian.")
    else:
        report.append(f"None of the tested regime configurations yielded satisfactory results. The best configuration (#{best_result['config_id']}) achieved a Sharpe ratio of {best_result['sharpe_ratio']:.2f} and total return of {best_result['total_return']:.2f}%.")
        report.append("\nRecommendations:")
        report.append("1. Consider optimizing the base strategy parameters before applying regime filtering")
        report.append("2. Test with different regime detection methods or parameters")
        report.append("3. Evaluate if the strategy is appropriate for this market or timeframe")
    
    # Save report to file
    report_path = os.path.join(output_dir, f"{symbol}_regime_config_report.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"Optimization report saved to {report_path}")

def plot_equity_curve(equity_curve, trades, symbol, strategy_params, regime_config, output_dir):
    """
    Plot equity curve with trade markers.
    
    Args:
        equity_curve: DataFrame with equity curve data
        trades: List of trades
        symbol: Symbol being traded
        strategy_params: Strategy parameters
        regime_config: Regime configuration
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
    ax1.set_title(f'Regime-Filtered Strategy - {symbol}', fontsize=14)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add parameter info
    param_text = f"MA Parameters: Fast={strategy_params.get('fast_window', 'N/A')}, Slow={strategy_params.get('slow_window', 'N/A')}"
    ax1.text(0.02, 0.95, param_text, transform=ax1.transAxes, fontsize=10, 
           bbox=dict(facecolor='white', alpha=0.8))
    
    # Add regime config info
    regime_text = "Regime Config:\n"
    for regime, signals in regime_config.items():
        signals_str = []
        if SignalEvent.BUY in signals:
            signals_str.append("Buy")
        if SignalEvent.SELL in signals:
            signals_str.append("Sell")
        regime_text += f"{regime.value}: {', '.join(signals_str) if signals_str else 'None'}\n"
    
    ax1.text(0.02, 0.05, regime_text, transform=ax1.transAxes, fontsize=8, 
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
    fig_path = os.path.join(output_dir, f"{symbol}_best_regime_config.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Equity curve chart saved to {fig_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize Regime Configuration')
    parser.add_argument('--data', type=str, required=True, help='Path to price data CSV file')
    parser.add_argument('--output', type=str, default=None, help='Directory to save results')
    parser.add_argument('--cash', type=float, default=10000.0, help='Initial portfolio cash')
    parser.add_argument('--fast', type=int, default=10, help='Fast MA window')
    parser.add_argument('--slow', type=int, default=30, help='Slow MA window')
    
    args = parser.parse_args()
    
    strategy_params = {
        'fast_window': args.fast,
        'slow_window': args.slow
    }
    
    optimize_regime_configuration(
        data_path=args.data,
        output_dir=args.output,
        initial_cash=args.cash,
        strategy_params=strategy_params
    )
