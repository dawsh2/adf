#!/usr/bin/env python
"""
Regime-Based Optimization Example

This script demonstrates how to use the regime detection and optimization components
to tune strategy parameters for different market regimes.
"""
import os
import sys
import logging
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import data components
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler

# Import event system components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType, BarEvent
from src.core.events.event_emitters import BarEmitter
from src.core.events.event_utils import EventTracker

# Import regime detection and optimization
from src.models.filters.regime.regime_detector import MarketRegime, EnhancedRegimeDetector
from src.models.filters.regime.detector_factory import RegimeDetectorFactory
from src.models.optimization.regime_optimizer import RegimeSpecificOptimizer
from src.models.optimization.grid_search import GridSearchOptimizer

# Import strategy components
from src.models.filters.regime.regime_strategy import RegimeAwareStrategy

# Import moving average strategy for this example
# This assumes you have a MovingAverageCrossoverStrategy implementation
# If you don't, you'll need to create or import a different strategy
from src.strategy.strategies.ma_crossover import MovingAverageCrossoverStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sample_data(data_dir, symbol, timeframe):
    """
    Load sample market data for demonstration.
    
    Args:
        data_dir: Directory containing data files
        symbol: Symbol to load
        timeframe: Data timeframe
        
    Returns:
        tuple: (data_source, data_handler) for the loaded data
    """
    # Create data source
    data_source = CSVDataSource(
        data_dir=data_dir,
        filename_pattern='{symbol}_{timeframe}.csv',
        date_column='timestamp',
        date_format=None,  # Auto-detect format
        column_map={
            'open': ['Open', 'open'],
            'high': ['High', 'high'],
            'low': ['Low', 'low'],
            'close': ['Close', 'close'],
            'volume': ['Volume', 'volume']
        }
    )
    
    # Create data handler
    data_handler = HistoricalDataHandler(data_source, None)
    
    # Load data
    data_handler.load_data(symbol, timeframe=timeframe)
    
    return data_source, data_handler


def detect_market_regimes(data_handler, symbol, detector_preset='advanced_sensitive'):
    """
    Run regime detection on the loaded data.
    
    Args:
        data_handler: Data handler with loaded data
        symbol: Symbol to analyze
        detector_preset: Preset configuration for the detector
        
    Returns:
        RegimeDetectorBase: Configured and updated detector
    """
    logger.info(f"Running regime detection for {symbol} with preset: {detector_preset}")
    
    # Create regime detector using factory
    detector = RegimeDetectorFactory.create_preset_detector(
        preset=detector_preset,
        debug=True  # Enable debug output
    )
    
    # Reset data handler to start of data
    data_handler.reset()
    
    # Process data with detector
    bar_count = 0
    
    while True:
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break
            
        # Update detector with bar data
        detector.update(bar)
        bar_count += 1
        
        if bar_count % 500 == 0:
            logger.info(f"Processed {bar_count} bars")
    
    logger.info(f"Completed regime detection with {bar_count} bars")
    
    # Print regime summary
    detector.print_regime_summary(symbol)
    
    # Reset data handler for further use
    data_handler.reset()
    
    return detector


def visualize_regimes(data_handler, symbol, detector):
    """
    Visualize price data with regime annotations.
    
    Args:
        data_handler: Data handler with loaded data
        symbol: Symbol to visualize
        detector: Regime detector with regime history
    """
    # Reset data handler
    data_handler.reset()
    
    # Extract price data and timestamps
    timestamps = []
    prices = []
    regimes = []
    
    while True:
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break
            
        # Get timestamp and price
        timestamp = bar.get_timestamp()
        close_price = bar.get_close()
        
        # Get regime at this timestamp
        regime = detector.get_regime_at(symbol, timestamp)
        
        # Store data
        timestamps.append(timestamp)
        prices.append(close_price)
        regimes.append(regime)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'regime': regimes
    })
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Plot price
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['price'], 'k-', label='Price')
    plt.title(f'{symbol} Price Chart')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()
    
    # Plot regimes as colored background
    regime_colors = {
        MarketRegime.UPTREND: 'lightgreen',
        MarketRegime.DOWNTREND: 'lightcoral',
        MarketRegime.SIDEWAYS: 'lightyellow',
        MarketRegime.VOLATILE: 'lightblue',
        MarketRegime.UNKNOWN: 'white'
    }
    
    # Find regime transitions
    transitions = []
    prev_regime = None
    
    for i, regime in enumerate(regimes):
        if regime != prev_regime:
            transitions.append((i, regime))
            prev_regime = regime
    
    # Add final point
    transitions.append((len(regimes) - 1, prev_regime))
    
    # Plot colored background for each regime segment
    for i in range(len(transitions) - 1):
        start_idx = transitions[i][0]
        end_idx = transitions[i+1][0]
        regime = transitions[i][1]
        color = regime_colors.get(regime, 'white')
        
        plt.axvspan(df.index[start_idx], df.index[end_idx], 
                    facecolor=color, alpha=0.3)
    
    # Plot regimes as separate plot
    plt.subplot(2, 1, 2)
    
    # Convert regimes to numeric values for plotting
    regime_values = {
        MarketRegime.UPTREND: 1,
        MarketRegime.SIDEWAYS: 0,
        MarketRegime.DOWNTREND: -1,
        MarketRegime.VOLATILE: 0.5,
        MarketRegime.UNKNOWN: 0
    }
    
    df['regime_value'] = [regime_values.get(r, 0) for r in df['regime']]
    
    # Plot regime values
    plt.plot(df.index, df['regime_value'], 'k-', linewidth=2)
    plt.fill_between(df.index, df['regime_value'], 0, 
                     where=df['regime_value'] > 0, color='green', alpha=0.3)
    plt.fill_between(df.index, df['regime_value'], 0, 
                     where=df['regime_value'] < 0, color='red', alpha=0.3)
    
    # Add horizontal lines
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    plt.axhline(y=1, color='g', linestyle='--', alpha=0.5)
    plt.axhline(y=-1, color='r', linestyle='--', alpha=0.5)
    
    # Configure plot
    plt.title(f'{symbol} Market Regimes')
    plt.ylabel('Regime')
    plt.grid(True)
    plt.yticks([1, 0.5, 0, -1], ['Uptrend', 'Volatile', 'Sideways', 'Downtrend'])
    
    # Show plot
    plt.tight_layout()
    plt.show()
    
    # Reset data handler
    data_handler.reset()


def optimize_for_regimes(data_handler, symbol, detector):
    """
    Perform regime-specific parameter optimization.
    
    Args:
        data_handler: Data handler with loaded data
        symbol: Symbol to optimize for
        detector: Regime detector with regime history
        
    Returns:
        dict: Optimized parameters for each regime
    """
    logger.info(f"Starting regime-specific optimization for {symbol}")
    
    # Define parameter grid for moving average strategy
    param_grid = {
        'fast_window': [3, 5, 8, 10, 15, 20],
        'slow_window': [15, 20, 30, 40, 50, 60]
    }
    
    # Create grid search optimizer
    grid_optimizer = GridSearchOptimizer()
    
    # Create regime-specific optimizer
    regime_optimizer = RegimeSpecificOptimizer(detector, grid_optimizer)
    
    # Define evaluation function (simplified for this example)
    def evaluate_strategy(params, data_handler, start_date, end_date):
        """
        Evaluate strategy parameters by running a simplified backtest.
        
        Args:
            params: Strategy parameters
            data_handler: Data handler with loaded data
            start_date: Start date for evaluation
            end_date: End date for evaluation
            
        Returns:
            float: Performance score (Sharpe ratio)
        """
        fast_window = params['fast_window']
        slow_window = params['slow_window']
        
        # Skip invalid parameter combinations
        if fast_window >= slow_window:
            return float('-inf')
        
        # Create event system for backtest
        event_bus = EventBus()
        event_tracker = EventTracker()
        
        # Track signals for performance evaluation
        event_bus.register(EventType.SIGNAL, event_tracker.track_event)
        
        # Create strategy
        strategy = MovingAverageCrossoverStrategy(
            name="ma_crossover",
            symbols=[symbol],
            fast_window=fast_window,
            slow_window=slow_window
        )
        strategy.set_event_bus(event_bus)
        
        # Reset data handler
        data_handler.reset()
        
        # Run simple backtest
        equity = 100000.0
        position = 0
        entry_price = 0
        returns = []
        dates = []
        
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
                
            # Skip bars outside date range
            timestamp = bar.get_timestamp()
            if start_date and timestamp < start_date:
                continue
            if end_date and timestamp > end_date:
                break
                
            # Process bar with strategy
            signal = strategy.on_bar(bar)
            
            # Simple execution (no slippage or trading costs)
            price = bar.get_close()
            dates.append(timestamp)
            
            # Execute trades based on signals
            if signal:
                signal_value = signal.get_signal_value()
                
                # Buy signal
                if signal_value == 1 and position <= 0:
                    # Close short position if any
                    if position < 0:
                        profit = entry_price - price
                        equity += profit * abs(position)
                        returns.append(profit / entry_price)
                    
                    # Open long position
                    position = 100
                    entry_price = price
                    
                # Sell signal
                elif signal_value == -1 and position >= 0:
                    # Close long position if any
                    if position > 0:
                        profit = price - entry_price
                        equity += profit * position
                        returns.append(profit / entry_price)
                    
                    # Open short position
                    position = -100
                    entry_price = price
            
        # Calculate performance metrics
        if len(returns) < 5:
            return float('-inf')  # Not enough trades
            
        # Calculate Sharpe ratio
        returns_array = np.array(returns)
        sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
        
        return sharpe_ratio
    
    # Run regime-specific optimization
    regime_params = regime_optimizer.optimize(
        param_grid=param_grid,
        data_handler=data_handler,
        evaluation_func=evaluate_strategy,
        min_regime_bars=100,  # Need at least 100 bars to optimize for a regime
        optimize_metric='sharpe_ratio',  # Optimize for risk-adjusted returns
        min_improvement=0.1  # Require at least 10% improvement over baseline
    )
    
    # Print optimization results
    logger.info("Regime-Based Optimization Results:")
    logger.info(f"Baseline Parameters: {regime_optimizer.results['baseline_parameters']}")
    logger.info(f"Baseline Score: {regime_optimizer.results['baseline_score']:.4f}")
    
    logger.info("\nRegime-Specific Parameters:")
    for regime, params in regime_params.items():
        is_baseline = params == regime_optimizer.results['baseline_parameters']
        logger.info(f"{regime.value}: {params}" + (" (using baseline)" if is_baseline else ""))
    
    return regime_params


def run_regime_aware_backtest(data_handler, symbol, regime_params, detector):
    """
    Run a backtest with regime-aware parameter switching.
    
    Args:
        data_handler: Data handler with loaded data
        symbol: Symbol to backtest
        regime_params: Dictionary of regime-specific parameters
        detector: Regime detector to detect regimes
        
    Returns:
        tuple: (equity_curve, signals, regime_transitions)
    """
    logger.info(f"Running regime-aware backtest for {symbol}")
    
    # Create event system
    event_bus = EventBus()
    event_tracker = EventTracker()
    
    # Track signals and fill events
    event_bus.register(EventType.SIGNAL, event_tracker.track_event)
    
    # Create base strategy with default parameters
    base_strategy = MovingAverageCrossoverStrategy(
        name="ma_crossover",
        symbols=[symbol],
        fast_window=10,  # Default parameters
        slow_window=30   # Will be overridden by regime-specific params
    )
    
    # Create regime-aware strategy wrapper
    strategy = RegimeAwareStrategy(base_strategy, detector)
    strategy.set_event_bus(event_bus)
    
    # Set regime-specific parameters
    for regime, params in regime_params.items():
        strategy.set_regime_parameters(regime, params)
    
    # Run simple backtest
    equity = 100000.0
    position = 0
    entry_price = 0
    equity_curve = []
    signals = []
    regime_transitions = []
    current_regime = None
    
    # Reset data handler
    data_handler.reset()
    detector.reset()  # Reset detector state
    
    while True:
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break
            
        # Current timestamp and price
        timestamp = bar.get_timestamp()
        price = bar.get_close()
        
        # Track regime
        regime = detector.update(bar)
        if regime != current_regime:
            regime_transitions.append({
                'timestamp': timestamp,
                'from_regime': current_regime.value if current_regime else None,
                'to_regime': regime.value
            })
            current_regime = regime
        
        # Process bar with strategy
        signal = strategy.on_bar(bar)
        
        # Record signal if generated
        if signal:
            signals.append({
                'timestamp': timestamp,
                'signal_value': signal.get_signal_value(),
                'price': price,
                'regime': regime.value
            })
            
            signal_value = signal.get_signal_value()
            
            # Execute trades based on signals
            if signal_value == 1 and position <= 0:  # Buy signal
                # Close short position if any
                if position < 0:
                    profit = entry_price - price
                    equity += profit * abs(position)
                
                # Open long position
                position = 100
                entry_price = price
                
            elif signal_value == -1 and position >= 0:  # Sell signal
                # Close long position if any
                if position > 0:
                    profit = price - entry_price
                    equity += profit * position
                
                # Open short position
                position = -100
                entry_price = price
        
        # Record equity
        equity_curve.append({
            'timestamp': timestamp,
            'equity': equity + position * (price - entry_price),
            'regime': regime.value
        })
    
    logger.info(f"Backtest completed: Final equity: ${equity:,.2f}")
    logger.info(f"Signals generated: {len(signals)}")
    logger.info(f"Regime transitions: {len(regime_transitions)}")
    
    return equity_curve, signals, regime_transitions


def compare_regime_vs_standard(data_handler, symbol, regime_params):
    """
    Compare performance of regime-aware strategy vs standard strategy.
    
    Args:
        data_handler: Data handler with loaded data
        symbol: Symbol to test
        regime_params: Dictionary of regime-specific parameters
        
    Returns:
        dict: Comparison results
    """
    logger.info(f"Comparing regime-aware vs standard strategies for {symbol}")
    
    # Get baseline parameters (what would be used without regime optimization)
    baseline_params = regime_params[MarketRegime.UNKNOWN]
    
    # Run standard backtest with baseline parameters
    event_bus = EventBus()
    event_tracker = EventTracker()
    event_bus.register(EventType.SIGNAL, event_tracker.track_event)
    
    standard_strategy = MovingAverageCrossoverStrategy(
        name="ma_crossover",
        symbols=[symbol],
        fast_window=baseline_params['fast_window'],
        slow_window=baseline_params['slow_window']
    )
    standard_strategy.set_event_bus(event_bus)
    
    # Run simple backtest for standard strategy
    equity = 100000.0
    position = 0
    entry_price = 0
    standard_equity_curve = []
    
    # Reset data handler
    data_handler.reset()
    
    while True:
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break
            
        # Current timestamp and price
        timestamp = bar.get_timestamp()
        price = bar.get_close()
        
        # Process bar with strategy
        signal = standard_strategy.on_bar(bar)
        
        # Execute trades based on signals
        if signal:
            signal_value = signal.get_signal_value()
            
            # Buy signal
            if signal_value == 1 and position <= 0:
                # Close short position if any
                if position < 0:
                    profit = entry_price - price
                    equity += profit * abs(position)
                
                # Open long position
                position = 100
                entry_price = price
                
            # Sell signal
            elif signal_value == -1 and position >= 0:
                # Close long position if any
                if position > 0:
                    profit = price - entry_price
                    equity += profit * position
                
                # Open short position
                position = -100
                entry_price = price
        
        # Record equity
        standard_equity_curve.append({
            'timestamp': timestamp,
            'equity': equity + position * (price - entry_price)
        })
    
    # Run regime-aware backtest
    # Create detector first
    detector = RegimeDetectorFactory.create_preset_detector(
        preset='advanced_sensitive',
        debug=False
    )
    
    # Process data to build regime history
    data_handler.reset()
    while True:
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break
        detector.update(bar)
    
    # Run regime-aware backtest
    regime_equity_curve, _, _ = run_regime_aware_backtest(
        data_handler=data_handler,
        symbol=symbol,
        regime_params=regime_params,
        detector=detector
    )
    
    # Calculate performance metrics
    standard_final_equity = standard_equity_curve[-1]['equity']
    standard_initial_equity = standard_equity_curve[0]['equity']
    standard_return = (standard_final_equity / standard_initial_equity) - 1
    
    regime_final_equity = regime_equity_curve[-1]['equity']
    regime_initial_equity = regime_equity_curve[0]['equity']
    regime_return = (regime_final_equity / regime_initial_equity) - 1
    
    # Calculate other metrics (e.g., drawdown, Sharpe ratio)
    # ...
    
    # Print comparison
    logger.info("\n=== Performance Comparison ===")
    logger.info(f"Standard Strategy Return: {standard_return:.2%}")
    logger.info(f"Regime-Aware Strategy Return: {regime_return:.2%}")
    logger.info(f"Improvement: {(regime_return - standard_return):.2%}")
    
    # Plot equity curves
    plt.figure(figsize=(12, 6))
    
    # Convert to DataFrames
    df_standard = pd.DataFrame(standard_equity_curve)
    df_standard.set_index('timestamp', inplace=True)
    
    df_regime = pd.DataFrame(regime_equity_curve)
    df_regime.set_index('timestamp', inplace=True)
    
    # Plot
    plt.plot(df_standard.index, df_standard['equity'], 'b-', 
             label=f'Standard Strategy: {standard_return:.2%}')
    plt.plot(df_regime.index, df_regime['equity'], 'g-', 
             label=f'Regime-Aware Strategy: {regime_return:.2%}')
    
    plt.title(f'{symbol} Strategy Comparison')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Return comparison results
    return {
        'standard': {
            'return': standard_return,
            'final_equity': standard_final_equity,
            'equity_curve': standard_equity_curve
        },
        'regime_aware': {
            'return': regime_return,
            'final_equity': regime_final_equity,
            'equity_curve': regime_equity_curve
        },
        'improvement': regime_return - standard_return
    }


def main():
    """Main function to run the example."""
    print("\n=== Regime-Based Optimization Example ===\n")
    
    # Configuration
    data_dir = os.path.expanduser("~/data")  # Update with your data directory
    symbol = "SPY"
    timeframe = "1d"
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        data_dir = "data"  # Try relative path
        
    # 1. Load sample data
    print(f"Loading data for {symbol} from {data_dir}...")
    data_source, data_handler = load_sample_data(data_dir, symbol, timeframe)
    
    # 2. Detect market regimes
    print("\nRunning regime detection...")
    detector = detect_market_regimes(data_handler, symbol)
    
    # 3. Visualize regimes
    print("\nVisualizing regimes...")
    visualize_regimes(data_handler, symbol, detector)
    
    # 4. Optimize for different regimes
    print("\nRunning regime-specific optimization...")
    regime_params = optimize_for_regimes(data_handler, symbol, detector)
    
    # 5. Compare regime-aware vs standard strategies
    print("\nComparing regime-aware vs standard strategies...")
    comparison = compare_regime_vs_standard(data_handler, symbol, regime_params)
    
    print("\nExample completed! Check the plots for visualization.")


if __name__ == "__main__":
    main()
