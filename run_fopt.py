#!/usr/bin/env python
"""
Regime-Based Optimization Implementation

This demonstrates how to implement regime detection and regime-specific optimization
for trading strategies using the existing component framework.
"""
import os
import sys
import numpy as np
import pandas as pd
import datetime
import logging
import itertools
import pytz  # Added pytz for timezone handling
import random 
import traceback
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict


# Data components
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler

# Event system components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType, Event, SignalEvent, OrderEvent, FillEvent, BarEvent
from src.core.events.event_emitters import BarEmitter
from src.core.events.event_utils import create_fill_event, create_order_event, EventTracker 

# Strategy and execution components
from src.execution.brokers.simulated import SimulatedBroker
from src.execution.execution_base import ExecutionEngine
from src.execution.portfolio import PortfolioManager
from src.strategy.risk.risk_manager import SimpleRiskManager

# For grid search
from src.models.optimization.grid_search import GridSearchOptimizer

# Performance reporting
from src.analytics import PerformanceCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Create event system components - MOVE THIS UP before using it
event_bus = EventBus(use_weak_refs=False)
event_manager = EventManager(event_bus)

tracker = EventTracker(verbose=True)

for event_type in EventType:
    event_bus.register(event_type, tracker.track_event)


def generate_optimization_report(optimization_results, data_handler, symbol, regime_detector):
    """
    Generate a comprehensive report of optimization results and regime analysis.
    
    Args:
        optimization_results: Results from regime optimization
        data_handler: Data handler with loaded data
        symbol: Symbol being analyzed
        regime_detector: Regime detector instance
    """
    regime_params = optimization_results.get('regime_parameters', {})
    baseline_params = optimization_results.get('baseline_parameters', {})
    baseline_score = optimization_results.get('baseline_score', 0)
    
    print("\n" + "="*80)
    print("             COMPREHENSIVE OPTIMIZATION REPORT")
    print("="*80)
    
    # Section 1: Regime Distribution
    print("\n1. REGIME DISTRIBUTION ANALYSIS")
    print("-"*50)
    
    # Get regime periods and calculate stats
    regime_periods = regime_detector.get_regime_periods(symbol)
    
    # Count bars in each regime
    regime_bars = {}
    total_bars = 0
    
    for regime, periods in regime_periods.items():
        bar_count = 0
        for start, end in periods:
            # Simple estimation - could be more accurate with actual bar counting
            days = (end - start).total_seconds() / (24 * 3600)
            bar_count += int(days * 390)  # Rough estimate for minute bars
        regime_bars[regime] = bar_count
        total_bars += bar_count
    
    # Print regime distribution
    print(f"Total trading periods analyzed: {sum(len(periods) for periods in regime_periods.values())}")
    print(f"Estimated total bars: {total_bars}")
    print("\nRegime Distribution:")
    
    for regime, count in sorted(regime_bars.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_bars * 100) if total_bars > 0 else 0
        bar_length = int(pct / 2)  # Scale to make it fit in console
        bar = "█" * bar_length
        print(f"  {regime.value:<10}: {count:>6} bars ({pct:>6.2f}%) {bar}")
    
    # Section 2: Optimization Results by Regime
    print("\n2. OPTIMIZATION RESULTS BY REGIME")
    print("-"*50)
    
    print(f"Baseline Parameters: {baseline_params}")
    print(f"Baseline Performance Score: {baseline_score:.4f}")
    print("\nRegime-Specific Parameters:")
    
    # Table header
    print("\n{:<12} {:<20} {:<15} {:<15} {:<15}".format(
        "Regime", "Parameters", "Score", "vs Baseline", "Bars"
    ))
    print("-"*80)
    
    # Display each regime's results
    for regime in sorted(regime_bars.keys(), key=lambda r: regime_bars.get(r, 0), reverse=True):
        params = regime_params.get(regime, baseline_params)
        is_baseline = params == baseline_params
        
        # Get performance metrics if available
        score = optimization_results.get(f'{regime.value}_score', baseline_score if is_baseline else "N/A")
        if isinstance(score, (int, float)):
            vs_baseline = f"{((score / baseline_score) - 1) * 100:.2f}%" if baseline_score != 0 else "N/A"
        else:
            vs_baseline = "N/A"
        
        params_str = f"fast={params.get('fast_window')}, slow={params.get('slow_window')}"
        print("{:<12} {:<20} {:<15} {:<15} {:<15}".format(
            regime.value, 
            params_str, 
            f"{score:.4f}" if isinstance(score, (int, float)) else score,
            vs_baseline,
            f"{regime_bars.get(regime, 0):,}"
        ))
    
    # Section 3: Parameter Performance Heatmap
    print("\n3. PARAMETER PERFORMANCE HEATMAP")
    print("-"*50)
    
    # Extract parameter grid from results
    all_params_results = optimization_results.get('all_param_results', [])
    if not all_params_results:
        print("Parameter performance data not available in optimization results.")
    else:
        # Find unique fast and slow window values
        fast_windows = sorted(set(p['params']['fast_window'] for p in all_params_results))
        slow_windows = sorted(set(p['params']['slow_window'] for p in all_params_results))
        
        # Create score matrix
        score_matrix = {}
        for result in all_params_results:
            params = result['params']
            if 'fast_window' in params and 'slow_window' in params:
                if params['fast_window'] < params['slow_window']:  # Only valid combinations
                    key = (params['fast_window'], params['slow_window'])
                    score_matrix[key] = result.get('score', 0)
        
        # Print heatmap header
        print("\nPerformance by Parameter Combination (Score Value):")
        print(f"{'fast\\slow':<10}", end="")
        for slow in slow_windows:
            print(f"{slow:>8}", end="")
        print()
        
        # Print heatmap rows
        for fast in fast_windows:
            print(f"{fast:<10}", end="")
            for slow in slow_windows:
                if fast < slow:  # Valid combination
                    score = score_matrix.get((fast, slow), "")
                    if isinstance(score, (int, float)):
                        print(f"{score:>8.4f}", end="")
                    else:
                        print(f"{'':<8}", end="")
                else:
                    print(f"{'X':>8}", end="")  # Invalid combination
            print()
    
    # Section 4: Key Findings & Recommendations
    print("\n4. KEY FINDINGS & RECOMMENDATIONS")
    print("-"*50)
    
    # Find best overall parameters
    best_regime = None
    best_score = baseline_score
    
    for regime, params in regime_params.items():
        score = optimization_results.get(f'{regime.value}_score', 0)
        if isinstance(score, (int, float)) and score > best_score:
            best_score = score
            best_regime = regime
    
    # Generate insights
    print("Key Insights:")
    
    if best_regime and best_score > baseline_score:
        print(f"• Best performance achieved in {best_regime.value} regime: {best_score:.4f} " +
              f"({((best_score/baseline_score)-1)*100:.2f}% above baseline)")
    else:
        print("• Baseline parameters performed best overall - regime-specific optimization did not improve performance")
    
    # Most frequent regime
    most_frequent = max(regime_bars.items(), key=lambda x: x[1])[0] if regime_bars else None
    if most_frequent:
        print(f"• Most frequent regime: {most_frequent.value} ({regime_bars.get(most_frequent, 0)} bars, " +
              f"{regime_bars.get(most_frequent, 0)/total_bars*100:.2f}% of data)")
    
    # Analyze parameter trends
    fast_trend = {}
    slow_trend = {}
    
    for regime, params in regime_params.items():
        if params != baseline_params:  # Only consider optimized params
            fast = params.get('fast_window')
            slow = params.get('slow_window')
            if fast:
                fast_trend[regime] = fast
            if slow:
                slow_trend[regime] = slow
    
    if fast_trend:
        print("\nParameter Trends by Regime:")
        for regime, fast in fast_trend.items():
            slow = slow_trend.get(regime)
            print(f"• {regime.value}: Prefers {'faster' if fast < baseline_params.get('fast_window', 0) else 'slower'} " +
                  f"fast MA ({fast} vs {baseline_params.get('fast_window')}) and " +
                  f"{'wider' if slow > baseline_params.get('slow_window', 0) else 'narrower'} " +
                  f"slow MA ({slow} vs {baseline_params.get('slow_window')})")
    
    # General recommendations
    print("\nRecommendations:")
    
    if most_frequent and most_frequent.value == "sideways" and regime_bars.get(most_frequent, 0)/total_bars > 0.8:
        print("• Data appears predominantly sideways - consider using mean-reversion strategies instead of trend-following")
    
    if best_regime and best_score > baseline_score * 1.2:  # 20% improvement
        print(f"• Consider using regime-specific parameters for {best_regime.value} periods")
    else:
        print("• Stick with baseline parameters as regime-specific optimization shows limited benefit")
    
    if sum(1 for _, bars in regime_bars.items() if bars > 100) < 3:  # Less than 3 regimes with sufficient data
        print("• Insufficient data in some regimes for reliable optimization - collect more data or adjust regime detection")
    
    print("\n" + "="*80)

# Define a helper function to make timestamp comparisons consistent
def make_timestamps_compatible(ts1, ts2):
    """
    Make timestamps compatible for comparison by ensuring both are either
    timezone-aware or timezone-naive.
    
    Args:
        ts1: First timestamp
        ts2: Second timestamp
        
    Returns:
        tuple: (ts1_compatible, ts2_compatible)
    """
    # Check if timestamps have timezone info
    ts1_has_tz = hasattr(ts1, 'tzinfo') and ts1.tzinfo is not None
    ts2_has_tz = hasattr(ts2, 'tzinfo') and ts2.tzinfo is not None
    
    # If both have same timezone status, return as is
    if ts1_has_tz == ts2_has_tz:
        return ts1, ts2
    
    # If only one has timezone, make both naive
    if ts1_has_tz and not ts2_has_tz:
        return ts1.replace(tzinfo=None), ts2
    
    if not ts1_has_tz and ts2_has_tz:
        return ts1, ts2.replace(tzinfo=None)
    
    # Shouldn't reach here, but return original as fallback
    return ts1, ts2

# Adding a function to calculate portfolio stats since it's referenced 
# but not defined in the original code

def calculate_portfolio_stats(portfolio, equity_curve):
    """
    Calculate portfolio performance statistics.
    
    Args:
        portfolio: Portfolio manager instance
        equity_curve: List of equity values by timestamp
        
    Returns:
        dict: Performance statistics
    """
    # Create the performance calculator
    calculator = PerformanceCalculator()
    
    # Calculate performance metrics
    metrics = calculator.calculate(equity_curve, portfolio.fill_history)
    
    # Ensure the results match the expected keys in run_backtest
    result = {
        'initial_equity': metrics['initial_equity'],
        'final_equity': metrics['final_equity'],
        'total_return': metrics['total_return'],
        'annual_return': metrics['annual_return'],
        'max_drawdown': metrics['max_drawdown'],
        'sharpe_ratio': metrics['sharpe_ratio'],
        'trades': metrics['num_trades']  # Map num_trades to trades for compatibility
    }
    
    return result



# Adding a placeholder for run_backtest as it's referenced but not defined
def run_backtest(data_dir, symbols, start_date=None, end_date=None, 
               timeframe='1m', fast_window=5, slow_window=15, fixed_position_size=100):
    """
    Run a backtest with the given parameters.
    
    Args:
        data_dir: Directory containing data files
        symbols: Symbol or list of symbols to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        timeframe: Data timeframe
        fast_window: Fast moving average window
        slow_window: Slow moving average window
        fixed_position_size: Fixed position size for each trade
        
    Returns:
        tuple: (results, tracker, portfolio, equity_curve)
    """
    # Create event system
    event_bus = EventBus(use_weak_refs=False)
    event_manager = EventManager(event_bus)
    
    # Create tracker
    tracker = EventTracker()
    
    # Register tracker
    for event_type in EventType:
        event_bus.register(event_type, tracker.track_event)
    
    # Create data source
    data_source = CSVDataSource(
        data_dir=data_dir,
        filename_pattern='{symbol}_{timeframe}.csv',
        date_column='timestamp',
        date_format=None,
        column_map={
            'open': ['Open'],
            'high': ['High'],
            'low': ['Low'],
            'close': ['Close'],
            'volume': ['Volume']
        }
    )
    
    # Create emitter
    bar_emitter = BarEmitter("backtest_bar_emitter", event_bus)
    bar_emitter.start()
    
    # Create data handler
    data_handler = HistoricalDataHandler(data_source, bar_emitter)
    
    # Create strategy
    strategy = MovingAverageCrossoverStrategy(
        name="ma_crossover",
        symbols=symbols,
        fast_window=fast_window,
        slow_window=slow_window
    )
    strategy.set_event_bus(event_bus)
    
    # Create portfolio
    portfolio = PortfolioManager(initial_cash=100000.0, event_bus=event_bus)
    
    # Create risk manager
    risk_manager = SimpleRiskManager(
        portfolio=portfolio,
        event_bus=event_bus,
        fixed_size=fixed_position_size
    )
    
    # Create execution components
    broker = SimulatedBroker(fill_emitter=event_bus)
    execution_engine = ExecutionEngine(broker_interface=broker, event_bus=event_bus)
    
    # Register components
    event_manager.register_component('strategy', strategy, [EventType.BAR])
    event_manager.register_component('risk', risk_manager, [EventType.SIGNAL])
    event_manager.register_component('execution', execution_engine, [EventType.ORDER])
    event_manager.register_component('portfolio', portfolio, [EventType.FILL])
    
    # Load data
    for symbol in symbols:
        data_handler.load_data(symbol, start_date=start_date, end_date=end_date, timeframe=timeframe)
    
    # Process data
    equity_curve = []
    
    for symbol in symbols:
        bar_count = 0
        
        # Process bars
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
            
            # Update broker market data
            broker.update_market_data(symbol, {"price": bar.get_close()})
            
            # Record equity
            equity_curve.append({
                'timestamp': bar.get_timestamp(),
                'equity': portfolio.get_equity({symbol: bar.get_close()})
            })
            
            bar_count += 1
            
            if bar_count % 100 == 0:
                logger.info(f"Processed {bar_count} bars for {symbol}")
        
        logger.info(f"Completed processing {bar_count} bars for {symbol}")
    
    # Calculate performance stats
    stats = calculate_portfolio_stats(portfolio, equity_curve)
    
    # Compile results
    results = {
        'initial_equity': stats['initial_equity'],
        'final_equity': stats['final_equity'],
        'return': stats['total_return'],
        'return_pct': stats['total_return'] * 100,
        'annual_return': stats['annual_return'] * 100,
        'max_drawdown': stats['max_drawdown'] * 100,
        'sharpe_ratio': stats['sharpe_ratio'],
        'trade_count': stats['trades'],
        'signal_count': len(tracker.events[EventType.SIGNAL]),
        'order_count': len(tracker.events[EventType.ORDER])
    }
    
    return results, tracker, portfolio, equity_curve

#################################################
# Moving Average Crossover Strategy
#################################################

class MovingAverageCrossoverStrategy:
    """
    Moving average crossover strategy that generates buy/sell signals
    when fast MA crosses above/below slow MA.
    """
    
    def __init__(self, name, symbols, fast_window=5, slow_window=15, market_filter=None):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            symbols: List of symbols to trade
            fast_window: Fast moving average window
            slow_window: Slow moving average window
            market_filter: Optional market condition filter
        """
        self.name = name
        self.event_bus = None
        
        # Settings
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.market_filter = market_filter
        
        # Store price history
        self.price_history = {symbol: [] for symbol in self.symbols}
        
        # State
        self.last_ma_values = {symbol: {'fast': None, 'slow': None} for symbol in self.symbols}
        self.signals = []
        
        logger.info(f"Initialized MA Crossover strategy: fast={fast_window}, slow={slow_window}")
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
        return self
        
    def set_parameters(self, params):
        """Set strategy parameters."""
        if 'fast_window' in params:
            self.fast_window = params['fast_window']
        if 'slow_window' in params:
            self.slow_window = params['slow_window']
        
        logger.info(f"Updated MA parameters: fast={self.fast_window}, slow={self.slow_window}")
        
        # Reset state as parameter change invalidates previous calculations
        self.reset()
        
        return self
    
    def get_parameters(self):
        """Get current strategy parameters."""
        return {
            'fast_window': self.fast_window,
            'slow_window': self.slow_window
        }
    
    def validate_parameters(self, params):
        """Validate strategy parameters."""
        if 'fast_window' in params and 'slow_window' in params:
            return params['fast_window'] < params['slow_window']
        return True
    
    def on_bar(self, event):
        """Process a bar event."""
        if not isinstance(event, BarEvent):
            return None
            
        symbol = event.get_symbol()
        if symbol not in self.symbols:
            return None
        
        # Apply market filter if available
        if self.market_filter:
            # Update filter with new data
            active = self.market_filter.update(event)
            
            # Skip if filtered out
            if not active:
                reason = self.market_filter.get_reason(symbol)
                logger.debug(f"Signal generation filtered out for {symbol}: {reason}")
                return None
            
        # Add price to history
        close_price = event.get_close()
        self.price_history[symbol].append(close_price)
        
        logger.debug(f"Adding price {close_price} for {symbol}, history size: {len(self.price_history[symbol])}")
        
        # Keep history manageable
        if len(self.price_history[symbol]) > self.slow_window + 10:
            self.price_history[symbol] = self.price_history[symbol][-(self.slow_window + 10):]
        
        # Check if we have enough data
        if len(self.price_history[symbol]) < self.slow_window:
            return None
            
        # Calculate MAs
        fast_ma = sum(self.price_history[symbol][-self.fast_window:]) / self.fast_window
        slow_ma = sum(self.price_history[symbol][-self.slow_window:]) / self.slow_window
        
        logger.debug(f"MAs for {symbol}: fast={fast_ma:.2f}, slow={slow_ma:.2f}")
        
        # Get previous MA values
        prev_fast = self.last_ma_values[symbol]['fast']
        prev_slow = self.last_ma_values[symbol]['slow']
        
        # Update MA values
        self.last_ma_values[symbol]['fast'] = fast_ma
        self.last_ma_values[symbol]['slow'] = slow_ma
        
        # Skip if no previous values
        if prev_fast is None or prev_slow is None:
            logger.debug(f"No previous MA values for {symbol}, skipping signal generation")
            return None
            
        # Check for crossover
        signal = None
        
        # Buy signal: fast MA crosses above slow MA
        if fast_ma > slow_ma and prev_fast <= prev_slow:
            logger.info(f"BUY SIGNAL: {symbol} fast MA crossed above slow MA")
            signal = SignalEvent(
                signal_value=SignalEvent.BUY,
                price=close_price,
                symbol=symbol,
                rule_id=self.name,
                confidence=1.0,
                metadata={
                    'fast_ma': fast_ma,
                    'slow_ma': slow_ma
                },
                timestamp=event.get_timestamp()
            )
            
        # Sell signal: fast MA crosses below slow MA
        elif fast_ma < slow_ma and prev_fast >= prev_slow:
            logger.info(f"SELL SIGNAL: {symbol} fast MA crossed below slow MA")
            signal = SignalEvent(
                signal_value=SignalEvent.SELL,
                price=close_price,
                symbol=symbol,
                rule_id=self.name,
                confidence=1.0,
                metadata={
                    'fast_ma': fast_ma,
                    'slow_ma': slow_ma
                },
                timestamp=event.get_timestamp()
            )
        
        # Emit signal if generated
        if signal:
            self.signals.append(signal)
            
            if self.event_bus:
                logger.debug(f"Emitting signal: {symbol} {'BUY' if signal.get_signal_value() == SignalEvent.BUY else 'SELL'}")
                self.event_bus.emit(signal)
        
        return signal
    
    def reset(self):
        """Reset the strategy state."""
        self.price_history = {symbol: [] for symbol in self.symbols}
        self.last_ma_values = {symbol: {'fast': None, 'slow': None} for symbol in self.symbols}
        self.signals = []

#################################################
# Walk-Forward Optimizer
#################################################

class WalkForwardOptimizer:
    """
    Walk-forward optimization for strategy parameters.
    Uses time-based train/test splits to prevent look-ahead bias.
    """
    
    def __init__(self, train_size=0.6, test_size=0.4, windows=3):
        """
        Initialize the walk-forward optimizer.
        
        Args:
            train_size: Proportion of data for training (in each window)
            test_size: Proportion of data for testing (in each window)
            windows: Number of train/test windows to use
        """
        self.train_size = train_size
        self.test_size = test_size
        self.windows = windows
        self.results = []


    def optimize(self, param_grid, data_handler, evaluation_func, start_date=None, end_date=None):
        """
        Perform walk-forward optimization.

        Args:
            param_grid: Dictionary mapping parameter names to possible values
            data_handler: Data handler with loaded data
            evaluation_func: Function to evaluate parameter combinations
            start_date: Start date for optimization
            end_date: End date for optimization

        Returns:
            dict: Optimization results with best parameters and performance
        """
        # Get data date range (handles None values)
        symbol = data_handler.get_symbols()[0]  # Use first symbol

        # Reset data handler and collect bars
        data_handler.reset()
        all_bars = []
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
            all_bars.append(bar)

        # Reset data handler again
        data_handler.reset()

        if not all_bars:
            logger.warning("No data available for optimization")
            return None

        # Sort bars by timestamp
        all_bars.sort(key=lambda bar: bar.get_timestamp())

        # Get date range from actual bar data
        first_bar_time = all_bars[0].get_timestamp()
        last_bar_time = all_bars[-1].get_timestamp()

        # Override with provided dates if specified
        actual_start = first_bar_time if start_date is None else pd.to_datetime(start_date)
        actual_end = last_bar_time if end_date is None else pd.to_datetime(end_date)

        # Ensure both dates have the same timezone info
        if first_bar_time.tzinfo is not None:
            # Data has timezone info, make sure dates do too
            if actual_start.tzinfo is None:
                actual_start = actual_start.replace(tzinfo=first_bar_time.tzinfo)
            if actual_end.tzinfo is None:
                actual_end = actual_end.replace(tzinfo=first_bar_time.tzinfo)
        else:
            # Data is timezone naive, make dates naive too
            if actual_start.tzinfo is not None:
                actual_start = actual_start.replace(tzinfo=None)
            if actual_end.tzinfo is not None:
                actual_end = actual_end.replace(tzinfo=None)

        logger.info(f"Walk-forward optimization from {actual_start} to {actual_end}")

        # Create time-based windows
        window_results = []
        try:
            # Calculate date range in seconds
            date_range = (actual_end - actual_start).total_seconds()
            window_size = date_range / self.windows

            for i in range(self.windows):
                # Calculate window dates
                window_start = actual_start + datetime.timedelta(seconds=i * window_size)
                window_end = actual_start + datetime.timedelta(seconds=(i + 1) * window_size)

                # Calculate train/test split within window
                split_point = window_start + datetime.timedelta(seconds=window_size * self.train_size)

                train_start = window_start
                train_end = split_point
                test_start = split_point
                test_end = window_end

                logger.info(f"\nWindow {i+1}/{self.windows}:")
                logger.info(f"  Train: {train_start} to {train_end}")
                logger.info(f"  Test:  {test_start} to {test_end}")

                # Grid search on training data
                best_params, best_score = self._grid_search(
                    param_grid, 
                    data_handler,
                    evaluation_func,
                    train_start, 
                    train_end
                )

                # Test best parameters on test data
                test_score = evaluation_func(
                    best_params, 
                    data_handler, 
                    test_start, 
                    test_end
                )

                logger.info(f"  Best parameters: {best_params}")
                logger.info(f"  Train score: {best_score:.4f}, Test score: {test_score:.4f}")

                # Store window results
                window_results.append({
                    'window': i + 1,
                    'params': best_params,
                    'train_score': best_score,
                    'test_score': test_score,
                    'train_period': (train_start, train_end),
                    'test_period': (test_start, test_end)
                })

        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            import traceback
            traceback.print_exc()
            # Return best parameters found so far if any
            if window_results:
                # Use the best parameters from the windows we managed to complete
                best_window = max(window_results, key=lambda x: x['test_score'])
                return {
                    'best_params': best_window['params'],
                    'best_avg_score': best_window['test_score'],
                    'window_results': window_results
                }
            return None

        # Aggregate results across windows
        param_scores = defaultdict(list)

        # Track test performance for each parameter set
        for result in window_results:
            params_key = self._params_to_key(result['params'])
            param_scores[params_key].append(result['test_score'])

        # Calculate average performance
        avg_scores = {}
        for params_key, scores in param_scores.items():
            avg_scores[params_key] = {
                'params': self._key_to_params(params_key),
                'avg_score': np.mean(scores),
                'min_score': min(scores),
                'max_score': max(scores),
                'std_score': np.std(scores),
                'scores': scores
            }

        # Find best overall parameters (based on average test score)
        if not avg_scores:
            logger.error("No valid parameter combinations found")
            return None

        best_key = max(avg_scores.keys(), key=lambda k: avg_scores[k]['avg_score'])
        best_overall = avg_scores[best_key]

        # Format results
        aggregated_results = {
            'best_params': best_overall['params'],
            'best_avg_score': best_overall['avg_score'],
            'best_min_score': best_overall['min_score'],
            'best_max_score': best_overall['max_score'],
            'best_std_score': best_overall['std_score'],
            'window_results': window_results,
            'all_params': [avg_scores[k] for k in avg_scores]
        }

        # Sort all parameter results by average score (descending)
        aggregated_results['all_params'].sort(key=lambda x: x['avg_score'], reverse=True)

        return aggregated_results        
 

        
 
    def _grid_search(self, param_grid, data_handler, evaluation_func, start_date, end_date):
        """
        Perform grid search on a specific time period.
        
        Args:
            param_grid: Dictionary of parameters to test
            data_handler: Data handler with loaded data
            evaluation_func: Function to evaluate parameters
            start_date: Start date for training
            end_date: End date for training
            
        Returns:
            tuple: (best_params, best_score)
        """
        # Generate all parameter combinations
        param_names = sorted(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        combinations = list(itertools.product(*param_values))
        
        logger.info(f"Grid search: evaluating {len(combinations)} parameter combinations")
        
        # Track best parameters
        best_params = None
        best_score = float('-inf')
        
        # Evaluate each combination
        for i, values in enumerate(combinations):
            # Create parameter dict
            params = {name: value for name, value in zip(param_names, values)}
            
            # Skip invalid parameter combinations (e.g., fast_window >= slow_window)
            if 'fast_window' in params and 'slow_window' in params:
                if params['fast_window'] >= params['slow_window']:
                    continue
            
            # Evaluate on training data
            score = evaluation_func(params, data_handler, start_date, end_date)
            
            # Update best if improved
            if score > best_score:
                best_score = score
                best_params = params
                
            # Progress update
            if (i + 1) % 5 == 0 or (i + 1) == len(combinations):
                logger.info(f"Evaluated {i + 1}/{len(combinations)} combinations")
        
        return best_params, best_score
        
    def _params_to_key(self, params):
        """Convert parameters dict to string key for tracking."""
        return str(sorted(params.items()))
        
    def _key_to_params(self, key):
        """Convert string key back to parameters dict."""
        # Parse string representation of sorted items back to dict
        # This is a simplification - in practice, proper parsing would be needed
        items = eval(key)
        return dict(items)

#################################################
# 1. Regime Detection
#################################################

def run_walk_forward_optimization(data_dir, symbols, start_date, end_date, timeframe):
    """
    Run walk-forward optimization on the moving average strategy.
    
    Args:
        data_dir: Directory containing data files
        symbols: Symbol or list of symbols to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        timeframe: Data timeframe
        
    Returns:
        dict: Optimization results
    """
    print("\n=== Running Walk-Forward Optimization ===")
    
    # Convert symbols to list if needed
    if isinstance(symbols, str):
        symbols = [symbols]
    
    # Define parameter grid
    param_grid = {
        'fast_window': [5, 10, 15, 20],
        'slow_window': [20, 30, 40, 50]
    }
    
    # Create data source with timezone handling
    data_source = CSVDataSource(
        data_dir=data_dir,
        filename_pattern='{symbol}_{timeframe}.csv',
        date_column='timestamp',
        date_format=None,
        column_map={
            'open': ['Open'],
            'high': ['High'],
            'low': ['Low'],
            'close': ['Close'],
            'volume': ['Volume']
        }
    )
    
    # Create data handler (without event bus, just for data access)
    data_handler = HistoricalDataHandler(data_source, None)
    
    # Load data
    for symbol in symbols:
        data_handler.load_data(symbol, start_date=start_date, end_date=end_date, timeframe=timeframe)
    
    # Debug output to show what data is available
    for symbol in symbols:
        print(f"Symbol {symbol} data:")
        data_handler.reset()
        bar_count = 0
        start_date_seen = None
        end_date_seen = None
        
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
            bar_count += 1
            
            if bar_count == 1:
                start_date_seen = bar.get_timestamp()
            end_date_seen = bar.get_timestamp()
        
        print(f"  Bars: {bar_count}")
        print(f"  Date range: {start_date_seen} to {end_date_seen}")
        
        # Reset data handler
        data_handler.reset()
    
    # Create walk-forward optimizer
    optimizer = WalkForwardOptimizer(train_size=0.6, test_size=0.4, windows=3)
    
    # Define evaluation function that handles date range properly
    def evaluate_params(params, data_handler, period_start, period_end):
        """
        Run backtest with given parameters and return score.

        Args:
            params: Strategy parameters
            data_handler: Data handler with loaded data
            period_start: Start date for evaluation
            period_end: End date for evaluation

        Returns:
            float: Performance score (Sharpe ratio)
        """
        fast_window = params['fast_window']
        slow_window = params['slow_window']

        print(f"\nTesting parameters: Fast MA: {fast_window}, Slow MA: {slow_window}")
        print(f"Period: {period_start} to {period_end}")

        try:
            # Run backtest directly with the date parameters
            # The backtest function should handle timezone consistency internally
            results, _, _, equity_curve = run_backtest(
                data_dir=data_dir,
                symbols=symbols,
                start_date=period_start,
                end_date=period_end,
                timeframe=timeframe,
                fast_window=fast_window,
                slow_window=slow_window,
                fixed_position_size=100
            )

            # Calculate score (could be Sharpe ratio, returns, etc.)
            if not results:
                return float('-inf')

            # Use Sharpe ratio as score (could use other metrics)
            score = results.get('sharpe_ratio', 0)

            print(f"Parameters: Fast MA: {fast_window}, Slow MA: {slow_window}, Score: {score:.4f}")

            return score

        except Exception as e:
            print(f"Error evaluating parameters: {e}")
            return float('-inf')


    



class RegimeSpecificOptimizer:
    """
    Optimizer that tunes strategy parameters specifically for each market regime.
    """
    
    def __init__(self, regime_detector, grid_optimizer=None):
        """
        Initialize the regime-specific optimizer.
        
        Args:
            regime_detector: Regime detector instance
            grid_optimizer: Optional grid search optimizer instance
        """
        self.regime_detector = regime_detector
        self.grid_optimizer = grid_optimizer or GridSearchOptimizer()
        self.results = {}


    def optimize(self, param_grid, data_handler, evaluation_func, 
                start_date=None, end_date=None, min_regime_bars=30,
                optimize_metric='sharpe_ratio', min_improvement=0.1):
        """
        Perform regime-specific optimization.

        Args:
            param_grid: Dictionary of parameter names to possible values
            data_handler: Data handler with loaded data
            evaluation_func: Function to evaluate parameter combinations
            start_date: Start date for optimization
            end_date: End date for optimization
            min_regime_bars: Minimum bars required for a regime to be optimized
            optimize_metric: Metric to optimize ('sharpe_ratio', 'return', 'max_drawdown')
            min_improvement: Minimum improvement required to use regime-specific parameters

        Returns:
            dict: Best parameters for each regime
        """
        symbol = data_handler.get_symbols()[0]  # Use first symbol
        optimization_scores = {}
        all_param_results = []

        # 1. Run regime detection on historical data
        logger.info(f"Performing regime detection on historical data for {symbol}")
        self._detect_regimes(data_handler, start_date, end_date)

        # Print a summary of detected regimes
        self.regime_detector.print_regime_summary(symbol)

        # 2. Segment data by regime
        regime_periods = self.regime_detector.get_regime_periods(symbol, start_date, end_date)

        # Print periods for each regime
        logger.info("\nRegime Periods:")
        for regime, periods in regime_periods.items():
            total_days = sum((end - start).total_seconds() / (24 * 3600) for start, end in periods)
            logger.info(f"  {regime.value}: {len(periods)} periods, {total_days:.1f} days")

        # 3. First, optimize parameters on the entire dataset (baseline)
        logger.info("\n--- Optimizing Baseline Parameters (All Regimes) ---")
        baseline_params, baseline_score = self._grid_search(
            param_grid, 
            data_handler,
            evaluation_func,
            start_date, 
            end_date,
            optimize_metric
        )

        logger.info(f"Baseline parameters: {baseline_params}, Score: {baseline_score:.4f}")

        # 4. For each regime, optimize parameters
        regime_params = {
            MarketRegime.UNKNOWN: baseline_params  # Default to baseline for unknown
        }

        for regime in list(MarketRegime):
            if regime == MarketRegime.UNKNOWN:
                continue  # Already set to baseline

            if regime not in regime_periods or not regime_periods[regime]:
                logger.info(f"No data for {regime.value} regime, using baseline parameters")
                regime_params[regime] = baseline_params
                continue

            # Count total bars in this regime
            try:
                periods = regime_periods[regime]
                bar_count = self._count_bars_in_periods(data_handler, symbol, periods)

                if bar_count < min_regime_bars:
                    logger.info(f"Skipping optimization for {regime.value} - insufficient data ({bar_count} bars)")
                    regime_params[regime] = baseline_params
                    continue

                logger.info(f"\n--- Optimizing for {regime.value} regime ({bar_count} bars) ---")

                # Optimize for this regime
                regime_best_params, regime_score = self._optimize_for_regime(
                    param_grid, data_handler, evaluation_func, regime, periods, optimize_metric
                )

                optimization_scores[regime] = regime_score

                self.results = {
                    'regime_parameters': regime_params,
                    'baseline_parameters': baseline_params,
                    'baseline_score': baseline_score,
                    'regime_periods': {r.value: periods for r, periods in regime_periods.items() if r in regime_periods}
                }

                # Add optimization scores to results
                for regime, score in optimization_scores.items():
                    self.results[f'{regime.value}_score'] = score

                # Store all parameter combinations tested
                self.results['all_param_results'] = all_param_results

                # Check if regime-specific parameters are better than baseline
                improvement = (regime_score - baseline_score) / abs(baseline_score) if baseline_score != 0 else float('inf')

                if improvement >= min_improvement:
                    logger.info(f"Best parameters for {regime.value}: {regime_best_params}, "
                               f"Score: {regime_score:.4f} (Improvement: {improvement:.2%})")
                    regime_params[regime] = regime_best_params
                else:
                    logger.info(f"Parameters for {regime.value} not better than baseline "
                               f"(Score: {regime_score:.4f}, Improvement: {improvement:.2%})")
                    regime_params[regime] = baseline_params
            except Exception as e:
                logger.error(f"Error optimizing for {regime.value}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                regime_params[regime] = baseline_params

        # Store and return results
        self.results = {
            'regime_parameters': regime_params,
            'baseline_parameters': baseline_params,
            'baseline_score': baseline_score,
            'regime_periods': {r.value: periods for r, periods in regime_periods.items() if r in regime_periods}
        }

        for regime, params in regime_params.items():
            if regime in optimization_scores:
                self.results[f'{regime.value}_score'] = optimization_scores[regime]

        # Store all parameter combinations tested
        self.results['all_param_results'] = all_param_results  # This needs to be collected during optimization
        
        return regime_params
    



    def _detect_regimes(self, data_handler, start_date, end_date):
        """Run regime detection on historical data."""
        symbol = data_handler.get_symbols()[0]

        # Reset data handler to start of data
        data_handler.reset()

        # Reset detector
        self.regime_detector.reset()

        # Convert string dates to pandas Timestamp objects if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # Ensure consistent timezone handling
        # If start_date has timezone but bar timestamp doesn't, make start_date naive
        # If bar timestamp has timezone but start_date doesn't, make start_date aware

        # Process each bar to detect regimes
        bar_count = 0
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break

            # Get bar timestamp
            bar_timestamp = bar.get_timestamp()

            # Skip bars outside date range, handling timezone differences
            if start_date:
                # Make timestamp comparison timezone-consistent
                if bar_timestamp.tzinfo is not None and start_date.tzinfo is None:
                    # Bar has timezone but start_date doesn't, so convert start_date
                    start_date = start_date.tz_localize(bar_timestamp.tzinfo)
                elif bar_timestamp.tzinfo is None and start_date.tzinfo is not None:
                    # Start_date has timezone but bar doesn't, so convert bar timestamp
                    bar_ts_aware = bar_timestamp.tz_localize(start_date.tzinfo)
                    if bar_ts_aware < start_date:
                        continue
                elif bar_timestamp < start_date:
                    continue

            if end_date:
                # Make timestamp comparison timezone-consistent
                if bar_timestamp.tzinfo is not None and end_date.tzinfo is None:
                    # Bar has timezone but end_date doesn't, so convert end_date
                    end_date = end_date.tz_localize(bar_timestamp.tzinfo)
                elif bar_timestamp.tzinfo is None and end_date.tzinfo is not None:
                    # End_date has timezone but bar doesn't, so convert bar timestamp
                    bar_ts_aware = bar_timestamp.tz_localize(end_date.tzinfo)
                    if bar_ts_aware > end_date:
                        break
                elif bar_timestamp > end_date:
                    break

            # Detect regime
            self.regime_detector.update(bar)
            bar_count += 1

            if bar_count % 100 == 0:
                logger.debug(f"Processed {bar_count} bars for regime detection")

        logger.info(f"Processed {bar_count} bars for regime detection")

        # Reset data handler for optimization
        data_handler.reset()


    # Update the _count_bars_in_periods method
    def _count_bars_in_periods(self, data_handler, symbol, periods):
        """Count total bars across multiple time periods."""
        # Reset data handler
        data_handler.reset()

        total_bars = 0

        # Process each bar
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break

            # Get bar timestamp
            timestamp = bar.get_timestamp()

            for start, end in periods:
                # Make timestamps compatible for comparison
                comp_start, comp_ts = make_timestamps_compatible(start, timestamp)
                comp_end, _ = make_timestamps_compatible(end, timestamp)

                # Now compare with compatible timestamps
                if comp_start <= comp_ts <= comp_end:
                    total_bars += 1
                    break

        # Reset data handler
        data_handler.reset()

        return total_bars

    def _optimize_for_regime(self, param_grid, data_handler, evaluation_func, regime, periods, optimize_metric):
        """Optimize parameters for a specific regime."""
        # Define regime-specific evaluation wrapper
        def regime_evaluation(params):
            # Evaluate only on bars in this regime
            result = self._evaluate_in_periods(
                params, data_handler, evaluation_func, periods, optimize_metric
            )
            return result

        # Run grid search with regime-specific evaluation
        try:
            param_combinations = []
            for name, values in param_grid.items():
                param_combinations.append(values)

            logger.info(f"Grid search: evaluating {len(list(itertools.product(*param_combinations)))} parameter combinations")

            # Run grid search
            result = self.grid_optimizer.optimize(param_grid, regime_evaluation)
            return result['best_params'], result['best_score']
        except Exception as e:
            logger.error(f"Error in regime optimization: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    

    def _evaluate_in_periods(self, params, data_handler, evaluation_func, periods, optimize_metric='sharpe_ratio'):
        """
        Evaluate parameters only on bars within specific time periods.

        This function extracts the bars within the specified periods and creates
        a mini-backtest environment to evaluate the parameters.
        """
        # Reset data handler
        data_handler.reset()

        symbol = data_handler.get_symbols()[0]
        bars_in_periods = []

        # Get the time range from periods
        min_start = None
        max_end = None
        for start, end in periods:
            start_naive = start.replace(tzinfo=None) if hasattr(start, 'tzinfo') and start.tzinfo else start
            end_naive = end.replace(tzinfo=None) if hasattr(end, 'tzinfo') and end.tzinfo else end

            if min_start is None or start_naive < min_start:
                min_start = start_naive
            if max_end is None or end_naive > max_end:
                max_end = end_naive

        # If no periods, return low score
        if min_start is None or max_end is None:
            return float('-inf')

        # Now we have the full date range to test
        try:
            # Call evaluation_func with the date range parameters it expects
            score = evaluation_func(params, data_handler, min_start, max_end)

            # If result is a dict, extract the metric
            if isinstance(score, dict):
                score = score.get(optimize_metric, float('-inf'))
                # Handle some metrics where lower is better
                if optimize_metric == 'max_drawdown':
                    score = -score  # Negate drawdown so lower values are better
        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return float('-inf')

        return score
        


        
    def _grid_search(self, param_grid, data_handler, evaluation_func, start_date, end_date, optimize_metric):
        """
        Perform grid search on a specific time period.
        
        Args:
            param_grid: Dictionary of parameters to test
            data_handler: Data handler with loaded data
            evaluation_func: Function to evaluate parameters
            start_date: Start date for training
            end_date: End date for training
            optimize_metric: Metric to optimize
            
        Returns:
            tuple: (best_params, best_score)
        """
        # Define evaluation wrapper for the metric
        def metric_evaluation(params):
            result = evaluation_func(params, data_handler, start_date, end_date)
            if isinstance(result, dict):
                # If result is a dict, extract the metric
                score = result.get(optimize_metric, float('-inf'))
                # Handle some metrics where lower is better
                if optimize_metric == 'max_drawdown':
                    score = -score  # Negate drawdown so lower values are better
            else:
                # Assume result is already the score
                score = result
            return score
        
        # Generate all parameter combinations for logging
        param_combinations = []
        for name, values in param_grid.items():
            param_combinations.append(values)
        
        logger.info(f"Grid search: evaluating {len(list(itertools.product(*param_combinations)))} parameter combinations")
        
        # Run grid search
        try:
            result = self.grid_optimizer.optimize(param_grid, metric_evaluation)
            return result['best_params'], result['best_score']
        except Exception as e:
            logger.error(f"Error in grid search: {e}")
            raise        

#################################################
# 2. Regime-Based Strategy
#################################################

class RegimeAwareStrategy:
    """Extends strategy with regime-based parameter switching."""
    
    def __init__(self, base_strategy, regime_detector):
        """
        Initialize the regime-aware strategy wrapper.
        
        Args:
            base_strategy: Base strategy to enhance with regime awareness
            regime_detector: Regime detector instance
        """
        self.strategy = base_strategy
        self.regime_detector = regime_detector
        self.event_bus = None
        self.name = f"regime_aware_{base_strategy.name}"
        self.symbols = base_strategy.symbols
        
        # Parameter sets for different regimes
        self.regime_parameters = {
            MarketRegime.UPTREND: {},
            MarketRegime.DOWNTREND: {},
            MarketRegime.SIDEWAYS: {},
            MarketRegime.VOLATILE: {},
            MarketRegime.UNKNOWN: {}  # Default parameters
        }
        
        # Current active parameters
        self.active_regime = {symbol: MarketRegime.UNKNOWN for symbol in self.symbols}
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
        self.strategy.set_event_bus(event_bus)
        return self
    
    def set_regime_parameters(self, regime, parameters):
        """
        Set parameters for a specific regime.
        
        Args:
            regime: MarketRegime to set parameters for
            parameters: Parameter dict for this regime
        """
        self.regime_parameters[regime] = parameters
        
        # If this is for the current regime of any symbol, apply immediately
        for symbol in self.symbols:
            if self.active_regime[symbol] == regime:
                logger.info(f"Applying new {regime.value} parameters to {symbol}: {parameters}")
                self.strategy.set_parameters(parameters)

    def on_bar(self, event):
        """
        Process a bar event with regime-specific parameters.

        Args:
            event: Bar event to process

        Returns:
            Signal event or None
        """
        symbol = event.get_symbol()

        if symbol not in self.symbols:
            return None

        # Detect current regime
        current_regime = self.regime_detector.update(event)

        # Check if regime changed
        if current_regime != self.active_regime[symbol]:
            # Log the regime change
            print(f"\n*** REGIME CHANGE at {event.get_timestamp()}: {symbol} {self.active_regime[symbol]} -> {current_regime}")
            print(f"    Switching parameters: {self.strategy.get_parameters()} -> {self.regime_parameters.get(current_regime, {})}")

            # Switch parameters
            self._switch_parameters(symbol, current_regime)

        # Log active parameters occasionally
        if random.random() < 0.01:  # Log roughly 1% of the time
            print(f"Active parameters for {symbol}: {self.strategy.get_parameters()}, Regime: {current_regime}")

        # Process with current parameters
        return self.strategy.on_bar(event)
                

    
    def _switch_parameters(self, symbol, regime):
        """
        Switch strategy parameters based on regime.
        
        Args:
            symbol: Symbol that experienced regime change
            regime: New regime for the symbol
        """
        logger.info(f"Regime change for {symbol}: {self.active_regime[symbol]} -> {regime}")
        
        # Update active regime
        self.active_regime[symbol] = regime
        
        # Apply regime-specific parameters
        parameters = self.regime_parameters.get(regime)
        
        if parameters:
            logger.info(f"Switching to {regime.value} parameters for {symbol}: {parameters}")
            self.strategy.set_parameters(parameters)
        else:
            logger.warning(f"No parameters defined for {regime.value} regime")
    
    def reset(self):
        """Reset the strategy and detector."""
        self.strategy.reset()
        self.regime_detector.reset()
        self.active_regime = {symbol: MarketRegime.UNKNOWN for symbol in self.symbols}


#################################################
# 3. Regime-Based Optimizer
#################################################

class RegimeOptimizer:
    """
    Optimizes strategy parameters separately for each market regime.
    """
    
    def __init__(self, regime_detector, base_optimizer):
        """
        Initialize the regime optimizer.
        
        Args:
            regime_detector: Regime detector instance
            base_optimizer: Base optimizer to use for each regime
        """
        self.regime_detector = regime_detector
        self.base_optimizer = base_optimizer
        self.results = {}
    
    def optimize(self, param_grid, data_handler, evaluation_func, 
                start_date=None, end_date=None, min_regime_bars=50):
        """
        Perform regime-specific optimization.
        
        Args:
            param_grid: Dictionary of parameter names to possible values
            data_handler: Data handler with loaded data
            evaluation_func: Function to evaluate parameter combinations
            start_date: Start date for optimization
            end_date: End date for optimization
            min_regime_bars: Minimum bars required for a regime to be optimized
            
        Returns:
            dict: Best parameters for each regime
        """
        symbol = data_handler.get_symbols()[0]  # Use first symbol
        
        # 1. Perform regime detection on historical data
        logger.info(f"Performing regime detection on historical data for {symbol}")
        self._detect_regimes(data_handler, start_date, end_date)
        
        # 2. Segment data by regime
        regime_periods = self.regime_detector.get_regime_periods(symbol, start_date, end_date)
        
        # 3. For each regime, optimize parameters
        regime_params = {}
        
        for regime, periods in regime_periods.items():
            if not periods:
                continue
                
            # Count total bars in this regime
            bar_count = self._count_bars_in_periods(data_handler, symbol, periods)
            
            if bar_count < min_regime_bars:
                logger.info(f"Skipping optimization for {regime.value} - insufficient data ({bar_count} bars)")
                continue
                
            logger.info(f"\n--- Optimizing for {regime.value} regime ({bar_count} bars) ---")
            
            # Optimize for this regime
            best_params, best_score = self._optimize_for_regime(
                param_grid, data_handler, evaluation_func, regime, periods
            )
            
            if best_params:
                logger.info(f"Best parameters for {regime.value}: {best_params}, Score: {best_score:.4f}")
                regime_params[regime] = best_params
            else:
                logger.warning(f"Optimization failed for {regime.value}")
        
        # 4. Set default parameters if any regime is missing
        all_regimes = list(MarketRegime)
        for regime in all_regimes:
            if regime not in regime_params:
                # Use parameters from most similar regime or average of others
                regime_params[regime] = self._get_default_params(regime, regime_params)
                
        # Store and return results
        self.results = {
            'regime_parameters': regime_params,
            'regime_periods': {r.value: periods for r, periods in regime_periods.items()}
        }
        
        return regime_params
    
    def _detect_regimes(self, data_handler, start_date, end_date):
        """Run regime detection on historical data."""
        symbol = data_handler.get_symbols()[0]
        
        # Reset data handler to start of data
        data_handler.reset()
        
        # Create temporary detector for historical analysis
        detector = self.regime_detector
        detector.reset()
        
        # Process each bar to detect regimes
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
                
            # Skip bars outside date range
            if start_date and bar.get_timestamp() < start_date:
                continue
            if end_date and bar.get_timestamp() > end_date:
                break
                
            # Detect regime
            detector.update(bar)
        
        # Reset data handler for optimization
        data_handler.reset()
    
    def _count_bars_in_periods(self, data_handler, symbol, periods):
        """Count total bars across multiple time periods."""
        # Reset data handler
        data_handler.reset()
        
        total_bars = 0
        
        # Process each bar
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
                
            # Check if bar falls in any of the periods
            timestamp = bar.get_timestamp()
            for start, end in periods:
                if start <= timestamp <= end:
                    total_bars += 1
                    break
        
        # Reset data handler
        data_handler.reset()
        
        return total_bars


    def _optimize_for_regime(self, param_grid, data_handler, evaluation_func, regime, periods, optimize_metric):
        """Optimize parameters for a specific regime."""
        # Define regime-specific evaluation wrapper
        def regime_evaluation(params):
            # Evaluate only on bars in this regime
            result = self._evaluate_in_periods(
                params, data_handler, evaluation_func, periods, optimize_metric
            )
            return result

        # Run grid search with regime-specific evaluation
        try:
            param_combinations = []
            for name, values in param_grid.items():
                param_combinations.append(values)

            logger.info(f"Grid search: evaluating {len(list(itertools.product(*param_combinations)))} parameter combinations")

            # Run grid search
            result = self.grid_optimizer.optimize(param_grid, regime_evaluation)
            return result['best_params'], result['best_score']
        except Exception as e:
            logger.error(f"Error in regime optimization: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _evaluate_in_periods(self, params, data_handler, evaluation_func, periods, optimize_metric='sharpe_ratio'):
        """
        Evaluate parameters only on bars within specific time periods.

        This function extracts the bars within the specified periods and creates
        a mini-backtest environment to evaluate the parameters.
        """
        # Reset data handler
        data_handler.reset()

        symbol = data_handler.get_symbols()[0]
        bars_in_periods = []

        # Get the time range from periods
        min_start = None
        max_end = None
        for start, end in periods:
            start_naive = start.replace(tzinfo=None) if hasattr(start, 'tzinfo') and start.tzinfo else start
            end_naive = end.replace(tzinfo=None) if hasattr(end, 'tzinfo') and end.tzinfo else end

            if min_start is None or start_naive < min_start:
                min_start = start_naive
            if max_end is None or end_naive > max_end:
                max_end = end_naive

        # If no periods, return low score
        if min_start is None or max_end is None:
            return float('-inf')

        # Now we have the full date range to test
        try:
            # Call evaluation_func with the date range parameters it expects
            score = evaluation_func(params, data_handler, min_start, max_end)

            # If result is a dict, extract the metric
            if isinstance(score, dict):
                score = score.get(optimize_metric, float('-inf'))
                # Handle some metrics where lower is better
                if optimize_metric == 'max_drawdown':
                    score = -score  # Negate drawdown so lower values are better
        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return float('-inf')

        return score        
    

    def _get_default_params(self, regime, existing_params):
        """Create default parameters for a regime with no optimization data."""
        # If we have parameters for other regimes, use similar or average
        if not existing_params:
            return {}
            
        # Map of regime similarities
        similarities = {
            MarketRegime.UPTREND: [MarketRegime.VOLATILE, MarketRegime.SIDEWAYS, MarketRegime.DOWNTREND],
            MarketRegime.DOWNTREND: [MarketRegime.VOLATILE, MarketRegime.SIDEWAYS, MarketRegime.UPTREND],
            MarketRegime.SIDEWAYS: [MarketRegime.UPTREND, MarketRegime.DOWNTREND, MarketRegime.VOLATILE],
            MarketRegime.VOLATILE: [MarketRegime.UPTREND, MarketRegime.DOWNTREND, MarketRegime.SIDEWAYS],
            MarketRegime.UNKNOWN: [MarketRegime.SIDEWAYS, MarketRegime.UPTREND, MarketRegime.DOWNTREND, MarketRegime.VOLATILE]
        }
        
        # Try to find parameters from most similar regime
        for similar_regime in similarities.get(regime, []):
            if similar_regime in existing_params:
                logger.info(f"Using {similar_regime.value} parameters for {regime.value}")
                return existing_params[similar_regime]
        
        # If no similar regime has parameters, use average of all available parameters
        logger.info(f"Using average parameters for {regime.value}")
        avg_params = {}
        
        # Get all parameter keys
        all_keys = set()
        for params in existing_params.values():
            all_keys.update(params.keys())
            
        # Calculate average for each parameter
        for key in all_keys:
            values = [params[key] for params in existing_params.values() if key in params]
            if values:
                avg_params[key] = sum(values) / len(values)
                
        return avg_params


#################################################
# 4. Example Usage
#################################################


def run_regime_optimization(data_dir, symbols, start_date, end_date, timeframe):
    """
    Run regime-specific optimization on the moving average strategy.
    
    Args:
        data_dir: Directory containing data files
        symbols: Symbol or list of symbols to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        timeframe: Data timeframe
        
    Returns:
        dict: Optimization results
    """
    print("\n=== Running Regime-Based Optimization ===")
    
    # Convert symbols to list if needed
    if isinstance(symbols, str):
        symbols = [symbols]
    
    # Convert string dates to pandas Timestamp objects
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Define parameter grid
    param_grid = {
        'fast_window': [3, 5, 8, 10, 15, 20],
        'slow_window': [15, 20, 30, 40, 50, 60]
    }
    
    # Create data source with timezone handling
    data_source = CSVDataSource(
        data_dir=data_dir,
        filename_pattern='{symbol}_{timeframe}.csv',
        date_column='timestamp',
        date_format=None,
        column_map={
            'open': ['Open'],
            'high': ['High'],
            'low': ['Low'],
            'close': ['Close'],
            'volume': ['Volume']
        }
    )
    
    # Create data handler (without event bus, just for data access)
    data_handler = HistoricalDataHandler(data_source, None)
    
    # Load data
    for symbol in symbols:
        data_handler.load_data(symbol, start_date=start_date, end_date=end_date, timeframe=timeframe)
    
    # Debug output to show what data is available
    for symbol in symbols:
        print(f"Symbol {symbol} data:")
        data_handler.reset()
        bar_count = 0
        start_date_seen = None
        end_date_seen = None
        
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
            bar_count += 1
            
            if bar_count == 1:
                start_date_seen = bar.get_timestamp()
            end_date_seen = bar.get_timestamp()
        
        print(f"  Bars: {bar_count}")
        print(f"  Date range: {start_date_seen} to {end_date_seen}")
        
        # Reset data handler
        data_handler.reset()
    
    # Create enhanced regime detector with debug enabled
    # Create enhanced regime detector with much more sensitive thresholds
    regime_detector = EnhancedRegimeDetector(
        lookback_window=10,           # Shorter window for faster regime changes
        trend_lookback=20,            # Shorter trend window
        volatility_lookback=5,        # Very short volatility window
        trend_threshold=0.002,        # Much lower threshold (0.2% change is enough for a trend)
        volatility_threshold=0.005,   # Lower volatility threshold
        sideways_threshold=0.001,     # Much tighter sideways definition
        debug=True
    )


    print("\nDetailed Regime Analysis:")
    regime_detector.print_regime_summary(symbol)
    
    # Create grid search optimizer
    grid_optimizer = GridSearchOptimizer()
    
    # Create regime-specific optimizer
    regime_optimizer = RegimeSpecificOptimizer(regime_detector, grid_optimizer)
    
    # Define evaluation function for risk-adjusted returns
    def evaluate_strategy(params, data_handler, start_date, end_date):
        """
        Run backtest with given parameters and return performance metrics.

        Args:
            params: Strategy parameters
            data_handler: Data handler with loaded data
            start_date: Start date for evaluation
            end_date: End date for evaluation

        Returns:
            dict: Performance metrics dictionary
        """
        fast_window = params['fast_window']
        slow_window = params['slow_window']

        # Validate parameters - skip invalid combinations
        if fast_window >= slow_window:
            return {
                'sharpe_ratio': float('-inf'),
                'return_pct': float('-inf'),
                'max_drawdown': float('inf'),
                'trade_count': 0
            }

        print(f"\nTesting parameters: Fast MA: {fast_window}, Slow MA: {slow_window}")
        print(f"Period: {start_date} to {end_date}")

        try:
            # Run backtest
            results, _, _, _ = run_backtest(
                data_dir=data_dir,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                fast_window=fast_window,
                slow_window=slow_window,
                fixed_position_size=100
            )

            if results is None:
                print("Backtest failed")
                return {
                    'sharpe_ratio': float('-inf'),
                    'return_pct': float('-inf'),
                    'max_drawdown': float('inf'),
                    'trade_count': 0
                }

            # Return performance metrics
            metrics = {
                'sharpe_ratio': results.get('sharpe_ratio', float('-inf')),
                'return_pct': results.get('return_pct', float('-inf')),
                'max_drawdown': results.get('max_drawdown', float('inf')),
                'trade_count': results.get('trade_count', 0)
            }

            print(f"Parameters: Fast MA: {fast_window}, Slow MA: {slow_window}, "
                  f"Sharpe: {metrics['sharpe_ratio']:.2f}, Return: {metrics['return_pct']:.2f}%, "
                  f"Drawdown: {metrics['max_drawdown']:.2f}%")

            return metrics

        except Exception as e:
            print(f"Error evaluating parameters: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return {
                'sharpe_ratio': float('-inf'),
                'return_pct': float('-inf'),
                'max_drawdown': float('inf'),
                'trade_count': 0
            }

    
    # Run regime-specific optimization
    regime_params = regime_optimizer.optimize(
        param_grid=param_grid,
        data_handler=data_handler,
        evaluation_func=evaluate_strategy,
        start_date=start_date,
        end_date=end_date,
        min_regime_bars=100,  # Need at least 100 bars to optimize for a regime
        optimize_metric='sharpe_ratio',  # Optimize for risk-adjusted returns
        min_improvement=0.1  # Require at least 10% improvement over baseline
    )
    
    # Print optimization results
    print("\n=== Regime-Based Optimization Results ===")
    print(f"Baseline Parameters: {regime_optimizer.results['baseline_parameters']}")
    print(f"Baseline Score: {regime_optimizer.results['baseline_score']:.4f}")
    
    print("\nRegime-Specific Parameters:")
    for regime, params in regime_params.items():
        is_baseline = params == regime_optimizer.results['baseline_parameters']
        print(f"{regime.value}: {params}" + (" (using baseline)" if is_baseline else ""))

    generate_optimization_report(regime_optimizer.results, data_handler, symbols[0], regime_detector)

    # Add this right before returning the results
    print_comprehensive_report(regime_optimizer.results, regime_detector, symbols[0])
    return regime_optimizer.results        


def print_comprehensive_report(results, regime_detector, symbol):
    """
    Print a comprehensive report of optimization results.
    
    Args:
        results: Optimization results dictionary
        regime_detector: Regime detector instance
        symbol: Symbol being analyzed
    """
    regime_params = results.get('regime_parameters', {})
    baseline_params = results.get('baseline_parameters', {})
    baseline_score = results.get('baseline_score', 0)
    
    print("\n" + "="*80)
    print("             COMPREHENSIVE OPTIMIZATION REPORT")
    print("="*80)
    
    # 1. Regime Distribution
    print("\n1. REGIME DISTRIBUTION")
    print("-"*50)
    regime_detector.print_regime_summary(symbol)
    
    # 2. Parameter Results by Regime
    print("\n2. OPTIMIZATION RESULTS BY REGIME")
    print("-"*50)
    
    print(f"Baseline Parameters: {baseline_params}")
    print(f"Baseline Score: {baseline_score:.4f}")
    print("\nRegime-Specific Parameters:")
    
    # Table header
    print("\n{:<12} {:<25} {:<15}".format(
        "Regime", "Parameters", "Using Baseline?"
    ))
    print("-"*60)
    
    # Display each regime's results
    for regime_type in list(MarketRegime):
        regime = regime_type  # Use enum directly
        if regime in regime_params:
            params = regime_params[regime]
            is_baseline = params == baseline_params
            
            params_str = f"fast={params.get('fast_window')}, slow={params.get('slow_window')}"
            print("{:<12} {:<25} {:<15}".format(
                regime.value, 
                params_str,
                "Yes" if is_baseline else "No"
            ))
    
    # 3. Analysis of Results
    print("\n3. ANALYSIS & RECOMMENDATIONS")
    print("-"*50)
    
    # Count regimes using custom parameters
    custom_count = sum(1 for regime, params in regime_params.items() 
                      if params != baseline_params)
    
    print(f"• Number of regimes using custom parameters: {custom_count} out of {len(regime_params)}")


    # Check regime distribution
    dominant_regime = None
    dominant_pct = 0

    for regime, regime_data in regime_detector.regime_history.items():
        # If regime_data is a list, get its length or sum
        if isinstance(regime_data, list):
            count = len(regime_data)
        else:
            # If it's already a number, use it directly
            count = regime_data

        if count > dominant_pct:
            dominant_pct = count
            dominant_regime = regime

    # Check if dominant_regime exists and is high enough
    if dominant_regime and dominant_pct > 80:
        # Need to check the type of dominant_regime
        if hasattr(dominant_regime, 'value'):
            # It's an enum, use the value attribute
            regime_name = dominant_regime.value
        else:
            # It's already a string
            regime_name = dominant_regime

        print(f"• Market predominantly in {regime_name} regime ({dominant_pct:.1f}% of data)")
        print("  Consider using single-regime optimization instead of regime-switching")            


    # Parameter trends
    if custom_count > 0:
        print("\n• Parameter trends by regime:")
        for regime, params in regime_params.items():
            if params != baseline_params:
                fast_diff = params.get('fast_window', 0) - baseline_params.get('fast_window', 0)
                slow_diff = params.get('slow_window', 0) - baseline_params.get('slow_window', 0)
                
                trend = []
                if fast_diff != 0:
                    trend.append(f"{'shorter' if fast_diff < 0 else 'longer'} fast MA ({fast_diff:+d})")
                if slow_diff != 0:
                    trend.append(f"{'shorter' if slow_diff < 0 else 'longer'} slow MA ({slow_diff:+d})")
                
                if trend:
                    print(f"  - {regime.value} regime prefers {' and '.join(trend)}")
    
    # General recommendations
    print("\nRecommendations:")
    
    if custom_count == 0:
        print("• Stick with baseline parameters as no regime shows significant improvement")
    elif custom_count == 1:
        improved_regime = next((r.value for r, p in regime_params.items() 
                              if p != baseline_params), None)
        print(f"• Consider using custom parameters for {improved_regime} regime only")
    else:
        print("• Full regime-switching approach recommended with provided parameters")
    
    print("\n" + "="*80)

def run_regime_backtest(data_dir, symbols, start_date, end_date, timeframe, regime_params):
    """
    Run a backtest with regime-specific parameter switching.
    
    Args:
        data_dir: Directory containing data files
        symbols: Symbol or list of symbols to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        timeframe: Data timeframe
        regime_params: Dictionary of regime-specific parameters
        
    Returns:
        tuple: (results, event_tracker, portfolio)
    """
    print("\n=== Running Regime-Based Backtest ===")
    
    # Convert symbols to list if needed
    if isinstance(symbols, str):
        symbols = [symbols]
    
    # Expand tilde in path if present
    if data_dir.startswith('~'):
        data_dir = os.path.expanduser(data_dir)
    
    # --- Setup Event System ---
    
    # Create event system
    event_bus = EventBus(use_weak_refs=False)
    event_manager = EventManager(event_bus)
    
    # Create event tracker
    tracker = EventTracker()
    
    # Register event tracking
    for event_type in EventType:
        event_bus.register(event_type, tracker.track_event)
    
    # --- Setup Data Components ---
    
    # Create data source
    data_source = CSVDataSource(
        data_dir=data_dir,
        filename_pattern='{symbol}_{timeframe}.csv',
        date_column='timestamp',
        date_format=None,
        column_map={
            'open': ['Open'],
            'high': ['High'],
            'low': ['Low'],
            'close': ['Close'],
            'volume': ['Volume']
        }
    )
    
    # Create bar emitter
    bar_emitter = BarEmitter("backtest_bar_emitter", event_bus)
    bar_emitter.start()
    
    # Create data handler
    data_handler = HistoricalDataHandler(data_source, bar_emitter)
    
    # --- Setup Portfolio ---
    
    # Create portfolio with initial capital
    initial_capital = 100000.0
    portfolio = PortfolioManager(initial_cash=initial_capital, event_bus=event_bus)
    
    # --- Setup Risk Manager ---
    
    # Create simplified risk manager
    
    risk_manager = SimpleRiskManager(
        portfolio=portfolio,
        event_bus=event_bus,
        fixed_size=10
    )
    
    # --- Setup Execution Components ---
    
    # Create broker
    broker = SimulatedBroker(fill_emitter=event_bus)
    
    # Create execution engine
    execution_engine = ExecutionEngine(broker_interface=broker, event_bus=event_bus)
    
    # --- Setup Strategy ---
    
    # Create base strategy (with default parameters)
    base_strategy = MovingAverageCrossoverStrategy(
        name="ma_crossover",
        symbols=symbols,
        fast_window=10,  # Default parameters
        slow_window=30   # Will be overridden by regime-specific params
    )
    
    # Create regime detector
    regime_detector = EnhancedRegimeDetector(
        lookback_window=15,        # Even shorter window for responsiveness
        trend_lookback=30,         # Shorter trend window
        volatility_lookback=10,    # Shorter volatility window
        trend_threshold=0.005,     # Much lower threshold (0.5% change)
        volatility_threshold=0.008, # Lower volatility threshold
        sideways_threshold=0.003,  # Much stricter sideways definition
        debug=True
    )



    # Create regime-aware strategy wrapper
    strategy = RegimeAwareStrategy(base_strategy, regime_detector)
    
    # Set regime-specific parameters
    for regime, params in regime_params.items():
        strategy.set_regime_parameters(regime, params)
    
    strategy.set_event_bus(event_bus)
    
    # Initial market price setup
    for symbol in symbols:
        broker.update_market_data(symbol, {"price": 100.0})
    
    # --- Register Components with Event Manager ---
    
    # Register all components
    event_manager.register_component('strategy', strategy, [EventType.BAR])
    event_manager.register_component('risk', risk_manager, [EventType.SIGNAL])
    event_manager.register_component('execution', execution_engine, [EventType.ORDER])
    event_manager.register_component('portfolio', portfolio, [EventType.FILL])
    
    # --- Run Backtest ---
    
    # Load data
    for symbol in symbols:
        data_handler.load_data(symbol, start_date=start_date, end_date=end_date, timeframe=timeframe)
    
    # Process bars for each symbol
    equity_curve = []
    regime_transitions = []
    
    # Go through data chronologically
    for symbol in symbols:
        bar_count = 0
        current_regime = None
        
        # Process bars
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
            
            # Update broker's market data with current price
            broker.update_market_data(symbol, {"price": bar.get_close()})
            
            # Record equity
            equity_curve.append({
                'timestamp': bar.get_timestamp(),
                'equity': portfolio.get_equity({symbol: bar.get_close()})
            })
            
            # Track regime transitions
            new_regime = regime_detector.get_current_regime(symbol)
            if new_regime != current_regime:
                regime_transitions.append({
                    'timestamp': bar.get_timestamp(),
                    'symbol': symbol,
                    'from_regime': current_regime.value if current_regime else 'None',
                    'to_regime': new_regime.value
                })
                current_regime = new_regime
            
            bar_count += 1
            
            if bar_count % 100 == 0:
                logger.info(f"Processed {bar_count} bars for {symbol}, Portfolio equity: ${portfolio.get_equity():,.2f}")
        
        logger.info(f"Completed processing {bar_count} bars for {symbol}")
    
    # --- Calculate Results ---
    
    # Calculate performance metrics
    stats = calculate_portfolio_stats(portfolio, equity_curve)
    
    # Compile results
    results = {
        'initial_equity': stats.get('initial_equity', initial_capital),
        'final_equity': stats.get('final_equity', portfolio.get_equity()),
        'return': stats.get('total_return', 0),
        'return_pct': stats.get('total_return', 0) * 100,
        'annual_return': stats.get('annual_return', 0) * 100,
        'max_drawdown': stats.get('max_drawdown', 0) * 100,
        'sharpe_ratio': stats.get('sharpe_ratio', 0),
        'trade_count': stats.get('trades', 0),
        'signal_count': len(tracker.events[EventType.SIGNAL]),
        'order_count': len(tracker.events[EventType.ORDER]),
        'regime_transitions': regime_transitions
    }
    
    # --- Print Summary ---
    
    print("\n=== Regime-Based Backtest Summary ===")
    print(f"Initial Equity: ${results['initial_equity']:,.2f}")
    print(f"Final Equity: ${results['final_equity']:,.2f}")
    print(f"Return: {results['return_pct']:.2f}%")
    print(f"Annual Return: {results['annual_return']:.2f}%")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Signals Generated: {results['signal_count']}")
    print(f"Orders Placed: {results['order_count']}")
    print(f"Trades Executed: {results['trade_count']}")
    
    # Print regime statistics
    regime_counts = {}
    for transition in regime_transitions:
        regime = transition['to_regime']
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
    
    print("\n=== Regime Statistics ===")
    for regime, count in regime_counts.items():
        print(f"{regime.capitalize()}: {count} occurrences")
    
    print(f"Total Regime Transitions: {len(regime_transitions)}")
    
    return results, tracker, portfolio

def compare_regime_vs_standard(data_dir, symbols, start_date, end_date, timeframe):
    """
    Compare regime-based optimization with standard optimization.
    
    Args:
        data_dir: Directory containing data files
        symbols: Symbol or list of symbols to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        timeframe: Data timeframe
    """
    print("\n=== Comparing Regime-Based vs. Standard Optimization ===")
    
    # Ensure symbols is a list
    if isinstance(symbols, str):
        symbols = [symbols]
    
    # Expand tilde in path if present
    if data_dir.startswith('~'):
        data_dir = os.path.expanduser(data_dir)
    
    try:
        # Debug: Print some information about the data
        print("\n--- Data Information ---")
        for symbol in symbols:
            file_path = os.path.join(data_dir, f"{symbol}_{timeframe}.csv")
            if not os.path.exists(file_path):
                print(f"WARNING: Data file not found: {file_path}")
            else:
                print(f"Data file found: {file_path}")
        
        # 1. Skip optimization and use fixed parameters for standard strategy
        print("\n--- Using Fixed Parameters for Standard Strategy ---")
        standard_params = {'fast_window': 5, 'slow_window': 20}
        print(f"Fixed parameters: {standard_params}")
        
        # 2. Run backtest with standard parameters
        print("\n--- Running Standard Backtest ---")
        # Store all the results in a tuple and unpack what we need
        standard_results = run_backtest(
            data_dir=data_dir,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            fast_window=standard_params['fast_window'],
            slow_window=standard_params['slow_window'],
            fixed_position_size=100
        )
        
        # Unpack appropriately based on the number of returned values
        if len(standard_results) == 4:
            standard_backtest, standard_tracker, standard_portfolio, standard_equity = standard_results
        elif len(standard_results) == 3:
            standard_backtest, standard_tracker, standard_portfolio = standard_results
            standard_equity = None
        else:
            print(f"Unexpected number of values from run_backtest: {len(standard_results)}")
            standard_backtest, standard_tracker, standard_portfolio, standard_equity = None, EventTracker(), None, None
        
        if standard_backtest is None:
            print("Standard backtest failed. Using default results.")
            standard_backtest = {
                'return_pct': 0.0,
                'annual_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'trade_count': 0,
                'signal_count': 0
            }
            standard_tracker = EventTracker()
        
        # Print detailed results from standard backtest
        print("\n--- Standard Backtest Details ---")
        print(f"Signals generated: {standard_tracker.event_counts.get(EventType.SIGNAL, 0)}")
        print(f"Orders placed: {standard_tracker.event_counts.get(EventType.ORDER, 0)}")
        print(f"Fills executed: {standard_tracker.event_counts.get(EventType.FILL, 0)}")
        
        # 3. Create simple regime parameters instead of optimizing
        print("\n--- Using Fixed Parameters for Regime-Based Strategy ---")
        # Create default regime parameters - using more aggressive parameters
        regime_params = {
            MarketRegime.UPTREND: {'fast_window': 3, 'slow_window': 15},
            MarketRegime.DOWNTREND: {'fast_window': 8, 'slow_window': 25},
            MarketRegime.SIDEWAYS: {'fast_window': 10, 'slow_window': 30},
            MarketRegime.VOLATILE: {'fast_window': 5, 'slow_window': 25},
            MarketRegime.UNKNOWN: {'fast_window': 5, 'slow_window': 20}
        }
        print(f"Fixed regime parameters: {regime_params}")
        
        # 4. Run backtest with regime-based parameters
        print("\n--- Running Regime-Based Backtest ---")
        # Store all the results in a tuple and unpack what we need
        regime_results = run_regime_backtest(
            data_dir=data_dir,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            regime_params=regime_params
        )
        
        # Unpack appropriately based on the number of returned values
        if len(regime_results) == 3:
            regime_backtest, regime_tracker, regime_portfolio = regime_results
            regime_equity = None
        elif len(regime_results) == 4:
            regime_backtest, regime_tracker, regime_portfolio, regime_equity = regime_results
        else:
            print(f"Unexpected number of values from run_regime_backtest: {len(regime_results)}")
            regime_backtest, regime_tracker, regime_portfolio, regime_equity = None, EventTracker(), None, None
        
        if regime_backtest is None:
            print("Regime-based backtest failed. Using default results.")
            regime_backtest = {
                'return_pct': 0.0,
                'annual_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'trade_count': 0,
                'signal_count': 0
            }
            regime_tracker = EventTracker()
        
        # Print detailed results from regime backtest
        print("\n--- Regime-Based Backtest Details ---")
        print(f"Signals generated: {regime_tracker.event_counts.get(EventType.SIGNAL, 0)}")
        print(f"Orders placed: {regime_tracker.event_counts.get(EventType.ORDER, 0)}")
        print(f"Fills executed: {regime_tracker.event_counts.get(EventType.FILL, 0)}")
        
        # 5. Compare results
        print("\n=== Performance Comparison ===")
        print(f"                     Standard      Regime-Based")
        print(f"Return:             {standard_backtest.get('return_pct', 0.0):>8.2f}%     {regime_backtest.get('return_pct', 0.0):>8.2f}%")
        print(f"Annual Return:      {standard_backtest.get('annual_return', 0.0):>8.2f}%     {regime_backtest.get('annual_return', 0.0):>8.2f}%")
        print(f"Max Drawdown:       {standard_backtest.get('max_drawdown', 0.0):>8.2f}%     {regime_backtest.get('max_drawdown', 0.0):>8.2f}%")
        print(f"Sharpe Ratio:       {standard_backtest.get('sharpe_ratio', 0.0):>8.2f}      {regime_backtest.get('sharpe_ratio', 0.0):>8.2f}")
        print(f"Trade Count:        {standard_backtest.get('trade_count', 0):>8d}      {regime_backtest.get('trade_count', 0):>8d}")
        print(f"Signal Count:       {standard_backtest.get('signal_count', 0):>8d}      {regime_backtest.get('signal_count', 0):>8d}")
        
        # Calculate improvement if possible
        std_sharpe = standard_backtest.get('sharpe_ratio', 0.0)
        regime_sharpe = regime_backtest.get('sharpe_ratio', 0.0)
        
        if std_sharpe > 0 and regime_sharpe > 0:
            sharpe_improvement = (regime_sharpe / std_sharpe - 1) * 100
            print(f"\nSharpe Ratio Improvement: {sharpe_improvement:.2f}%")
        
        # Return comparison results
        return {
            'standard': {
                'params': standard_params,
                'results': standard_backtest
            },
            'regime_based': {
                'params': regime_params,
                'results': regime_backtest
            }
        }
        
    except Exception as e:
        print(f"Error in comparison process: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Use absolute path with tilde expansion
    DATA_DIR = "~/adf/data"  # Path to your data directory
    DATA_DIR = os.path.expanduser(DATA_DIR)  # Expand tilde to full path
    
    print(f"Using data directory: {DATA_DIR}")
    
    # Define parameters for backtest
    SYMBOL = "SPY"
    START_DATE = "2024-03-26" 
    END_DATE = "2024-04-10"
    TIMEFRAME = "1m"
    
    # Define parameter grid
    param_grid = {
        'fast_window': [3, 5, 8, 10, 15, 20],
        'slow_window': [15, 20, 30, 40, 50, 60]
    }
    
    # Run the optimization
    print("\n=== Running Regime Optimization ===")
    optimization_results = run_regime_optimization(
        data_dir=DATA_DIR,
        symbols=[SYMBOL],
        start_date=START_DATE,
        end_date=END_DATE,
        timeframe=TIMEFRAME
    )
