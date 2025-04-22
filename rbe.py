#!/usr/bin/env python
"""
Regime-Based Backtesting Example

This script demonstrates how to implement a complete backtesting system
that adapts to different market regimes using the regime detection components.
"""
import os
import sys
import logging
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Add src directory to path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import data components
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler

# Import event system components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType, Event, BarEvent, SignalEvent
from src.core.events.event_emitters import BarEmitter
from src.core.events.event_utils import EventTracker

# Import regime detection components
from src.models.filters.regime.regime_detector import MarketRegime, EnhancedRegimeDetector
from src.models.filters.regime.detector_factory import RegimeDetectorFactory
from src.models.filters.regime.regime_strategy import RegimeAwareStrategy, MultiRegimeStrategy

# Import strategy components
from src.strategy.strategies.ma_crossover import MovingAverageCrossoverStrategy
from src.strategy.strategies.mean_reversion import MeanReversionStrategy
from src.strategy.strategies.momentum import MomentumStrategy

# Import execution components
from src.execution.brokers.simulated import SimulatedBroker
from src.execution.portfolio import PortfolioManager
from src.execution.position import Position
from src.execution.execution_base import ExecutionEngine
from src.strategy.risk.risk_manager import SimpleRiskManager

# Import performance calculation
from src.analytics.performance import PerformanceCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_market_data(data_dir, symbols, start_date=None, end_date=None, timeframe='1d'):
    """
    Load market data for backtesting.
    
    Args:
        data_dir: Directory containing data files
        symbols: Symbol or list of symbols to load
        start_date: Optional start date
        end_date: Optional end date
        timeframe: Data timeframe
        
    Returns:
        tuple: (data_source, data_handler) for the loaded data
    """
    # Ensure symbols is a list
    if isinstance(symbols, str):
        symbols = [symbols]
        
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
    
    # Create bar emitter
    bar_emitter = BarEmitter("backtest_bar_emitter")
    
    # Create data handler
    data_handler = HistoricalDataHandler(data_source, bar_emitter)
    
    # Load data for each symbol
    for symbol in symbols:
        try:
            data_handler.load_data(symbol, start_date=start_date, end_date=end_date, timeframe=timeframe)
            logger.info(f"Loaded data for {symbol}")
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
    
    return data_source, data_handler, bar_emitter


def setup_regime_detector(detector_preset='advanced'):
    """
    Create and configure a regime detector.
    
    Args:
        detector_preset: Preset configuration to use
        
    Returns:
        RegimeDetectorBase: Configured detector
    """
    # Create regime detector using factory
    detector = RegimeDetectorFactory.create_preset_detector(
        preset=detector_preset,
        debug=False  # Disable debug output for backtest
    )
    
    return detector


def setup_single_strategy_backtest(symbols, detector):
    """
    Setup a backtest with a single strategy that adapts to regimes.
    
    Args:
        symbols: Symbol or list of symbols to trade
        detector: Regime detector instance
        
    Returns:
        dict: Dictionary of system components
    """
    # Ensure symbols is a list
    if isinstance(symbols, str):
        symbols = [symbols]
    
    # Create event system
    event_bus = EventBus(use_weak_refs=False)
    event_manager = EventManager(event_bus)
    
    # Create event tracker
    tracker = EventTracker(verbose=False)
    
    # Track all events
    for event_type in EventType:
        event_bus.register(event_type, tracker.track_event)
    
    # Create base moving average crossover strategy
    base_strategy = MovingAverageCrossoverStrategy(
        name="ma_crossover",
        symbols=symbols,
        fast_window=10,  # Default parameters
        slow_window=30   # Will be overridden by regime-specific params
    )
    
    # Create regime-aware strategy wrapper
    strategy = RegimeAwareStrategy(base_strategy, detector)
    
    # Set regime-specific parameters
    strategy.set_regime_parameters(MarketRegime.UPTREND, {'fast_window': 3, 'slow_window': 15})
    strategy.set_regime_parameters(MarketRegime.DOWNTREND, {'fast_window': 8, 'slow_window': 40})
    strategy.set_regime_parameters(MarketRegime.SIDEWAYS, {'fast_window': 10, 'slow_window': 30})
    strategy.set_regime_parameters(MarketRegime.VOLATILE, {'fast_window': 5, 'slow_window': 25})
    strategy.set_regime_parameters(MarketRegime.UNKNOWN, {'fast_window': 10, 'slow_window': 30})
    
    # Connect strategy to event bus
    strategy.set_event_bus(event_bus)
    
    # Create portfolio with initial capital
    initial_capital = 100000.0
    portfolio = PortfolioManager(initial_cash=initial_capital)
    portfolio.set_event_bus(event_bus)
    
    # Create risk manager
    risk_manager = SimpleRiskManager(
        portfolio=portfolio,
        fixed_size=100
    )
    risk_manager.set_event_bus(event_bus)
    
    # Create broker
    broker = SimulatedBroker()
    broker.set_event_bus(event_bus)
    
    # Create execution engine
    execution_engine = ExecutionEngine(broker)
    execution_engine.set_event_bus(event_bus)
    
    # Register components with event manager
    event_manager.register_component('strategy', strategy, [EventType.BAR])
    event_manager.register_component('risk', risk_manager, [EventType.SIGNAL])
    event_manager.register_component('execution', execution_engine, [EventType.ORDER])
    event_manager.register_component('portfolio', portfolio, [EventType.FILL])
    
    return {
        'event_bus': event_bus,
        'event_manager': event_manager,
        'tracker': tracker,
        'strategy': strategy,
        'portfolio': portfolio,
        'risk_manager': risk_manager,
        'broker': broker,
        'execution_engine': execution_engine
    }


def setup_multi_strategy_backtest(symbols, detector):
    """
    Setup a backtest with multiple strategies that switch based on regime.
    
    Args:
        symbols: Symbol or list of symbols to trade
        detector: Regime detector instance
        
    Returns:
        dict: Dictionary of system components
    """
    # Ensure symbols is a list
    if isinstance(symbols, str):
        symbols = [symbols]
    
    # Create event system
    event_bus = EventBus(use_weak_refs=False)
    event_manager = EventManager(event_bus)
    
    # Create event tracker
    tracker = EventTracker(verbose=False)
    
    # Track all events
    for event_type in EventType:
        event_bus.register(event_type, tracker.track_event)
    
    # Create different strategies for different regimes
    
    # Moving average crossover strategy for trending markets
    ma_strategy = MovingAverageCrossoverStrategy(
        name="ma_crossover",
        symbols=symbols,
        fast_window=5,
        slow_window=20
    )
    ma_strategy.set_event_bus(event_bus)
    
    # Mean reversion strategy for sideways markets
    mr_strategy = MeanReversionStrategy(
        name="mean_reversion",
        symbols=symbols,
        lookback=20,
        z_threshold=1.5
    )
    mr_strategy.set_event_bus(event_bus)
    
    # Momentum strategy for volatile markets
    momentum_strategy = MomentumStrategy(
        name="momentum",
        symbols=symbols,
        lookback=10,
        threshold=0.01
    )
    momentum_strategy.set_event_bus(event_bus)
    
    # Create multi-regime strategy
    strategy = MultiRegimeStrategy(detector)
    strategy.set_event_bus(event_bus)
    
    # Assign strategies to regimes
    strategy.set_strategy_for_regime(MarketRegime.UPTREND, ma_strategy)
    strategy.set_strategy_for_regime(MarketRegime.DOWNTREND, ma_strategy)
    strategy.set_strategy_for_regime(MarketRegime.SIDEWAYS, mr_strategy)
    strategy.set_strategy_for_regime(MarketRegime.VOLATILE, momentum_strategy)
    strategy.set_strategy_for_regime(MarketRegime.UNKNOWN, ma_strategy)  # Default
    
    # Create portfolio with initial capital
    initial_capital = 100000.0
    portfolio = PortfolioManager(initial_cash=initial_capital)
    portfolio.set_event_bus(event_bus)
    
    # Create risk manager
    risk_manager = SimpleRiskManager(
        portfolio=portfolio,
        fixed_size=100
    )
    risk_manager.set_event_bus(event_bus)
    
    # Create broker
    broker = SimulatedBroker()
    broker.set_event_bus(event_bus)
    
    # Create execution engine
    execution_engine = ExecutionEngine(broker)
    execution_engine.set_event_bus(event_bus)
    
    # Register components with event manager
    event_manager.register_component('strategy', strategy, [EventType.BAR])
    event_manager.register_component('risk', risk_manager, [EventType.SIGNAL])
    event_manager.register_component('execution', execution_engine, [EventType.ORDER])
    event_manager.register_component('portfolio', portfolio, [EventType.FILL])
    
    return {
        'event_bus': event_bus,
        'event_manager': event_manager,
        'tracker': tracker,
        'strategy': strategy,
        'portfolio': portfolio,
        'risk_manager': risk_manager,
        'broker': broker,
        'execution_engine': execution_engine
    }


def run_backtest(data_handler, bar_emitter, system_components, detector):
    """
    Run the backtest.
    
    Args:
        data_handler: Data handler with loaded data
        bar_emitter: Bar emitter for emitting bar events
        system_components: Dictionary of system components
        detector: Regime detector
        
    Returns:
        tuple: (equity_curve, regime_transitions)
    """
    # Extract system components
    event_bus = system_components['event_bus']
    portfolio = system_components['portfolio']
    broker = system_components['broker']
    tracker = system_components['tracker']
    
    # Connect bar emitter to event bus
    bar_emitter.set_event_bus(event_bus)
    bar_emitter.start()
    
    # Initialize equity curve and regime history
    equity_curve = []
    regime_transitions = []
    current_regime = {}  # symbol -> regime
    
    # Reset data handler and detector
    data_handler.reset()
    detector.reset()
    
    # Process data for each symbol
    symbols = data_handler.get_symbols()
    
    for symbol in symbols:
        bar_count = 0
        
        # Process bars
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
                
            # Current timestamp and price
            timestamp = bar.get_timestamp()
            price = bar.get_close()
            
            # Update broker's market data
            broker.update_market_data(symbol, {"price": price})
            
            # Update detector with bar data
            regime = detector.update(bar)
            
            # Track regime changes
            if symbol not in current_regime or regime != current_regime[symbol]:
                regime_transitions.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'from_regime': current_regime.get(symbol, None),
                    'to_regime': regime
                })
                current_regime[symbol] = regime
            
            # Record equity
            if bar_count % 10 == 0:  # Record every 10 bars to save memory
                equity_curve.append({
                    'timestamp': timestamp,
                    'equity': portfolio.get_equity({symbol: price}),
                    'regime': regime.value
                })
            
            bar_count += 1
            
            if bar_count % 100 == 0:
                logger.debug(f"Processed {bar_count} bars for {symbol}")
        
        logger.info(f"Completed processing {bar_count} bars for {symbol}")
    
    # Ensure final equity value is recorded
    equity_curve.append({
        'timestamp': timestamp,
        'equity': portfolio.get_equity({symbol: price}),
        'regime': regime.value
    })
    
    logger.info(f"Backtest completed: Final equity: ${portfolio.get_equity():,.2f}")
    logger.info(f"Signals generated: {tracker.get_event_count(EventType.SIGNAL)}")
    logger.info(f"Orders executed: {tracker.get_event_count(EventType.ORDER)}")
    logger.info(f"Fills received: {tracker.get_event_count(EventType.FILL)}")
    logger.info(f"Regime transitions: {len(regime_transitions)}")
    
    return equity_curve, regime_transitions


def analyze_performance(equity_curve, regime_transitions, portfolio):
    """
    Analyze backtest performance.
    
    Args:
        equity_curve: List of equity points with timestamp and value
        regime_transitions: List of regime transitions
        portfolio: Portfolio manager instance
        
    Returns:
        dict: Performance metrics
    """
    # Create performance calculator
    calculator = PerformanceCalculator()
    
    # Calculate overall performance metrics
    metrics = calculator.calculate(equity_curve, portfolio.fill_history)
    
    # Create results dictionary
    results = {
        'initial_equity': metrics['initial_equity'],
        'final_equity': metrics['final_equity'],
        'total_return': metrics['total_return'],
        'total_return_pct': metrics['total_return'] * 100,
        'annual_return': metrics['annual_return'],
        'annual_return_pct': metrics['annual_return'] * 100,
        'max_drawdown': metrics['max_drawdown'],
        'max_drawdown_pct': metrics['max_drawdown'] * 100,
        'sharpe_ratio': metrics['sharpe_ratio'],
        'sortino_ratio': metrics['sortino_ratio'],
        'calmar_ratio': metrics['calmar_ratio'],
        'trades': metrics['num_trades'] if 'num_trades' in metrics else len(portfolio.fill_history),
        'regime_transitions': len(regime_transitions)
    }
    
    # Calculate regime-specific metrics
    regime_performance = {}
    
    # Convert equity curve to DataFrame
    df = pd.DataFrame(equity_curve)
    df.set_index('timestamp', inplace=True)
    
    # Calculate performance by regime
    for regime in MarketRegime:
        # Filter equity curve for this regime
        regime_df = df[df['regime'] == regime.value]
        
        if len(regime_df) < 2:
            continue
            
        # Calculate returns for this regime
        regime_df['returns'] = regime_df['equity'].pct_change()
        
        # Calculate metrics
        total_return = (regime_df['equity'].iloc[-1] / regime_df['equity'].iloc[0]) - 1
        
        # Only calculate other metrics if we have enough data
        if len(regime_df) >= 20:
            volatility = regime_df['returns'].std() * np.sqrt(252)
            sharpe = regime_df['returns'].mean() / regime_df['returns'].std() * np.sqrt(252) if regime_df['returns'].std() > 0 else 0
        else:
            volatility = float('nan')
            sharpe = float('nan')
        
        # Store regime metrics
        regime_performance[regime.value] = {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'volatility': volatility,
            'sharpe': sharpe,
            'bars': len(regime_df)
        }
    
    # Add regime-specific metrics to results
    results['regime_performance'] = regime_performance
    
    return results


def visualize_results(equity_curve, regime_transitions, performance_metrics, title="Backtest Results"):
    """
    Visualize backtest results with regime annotations.
    
    Args:
        equity_curve: List of equity points with timestamp and value
        regime_transitions: List of regime transitions
        performance_metrics: Dictionary of performance metrics
        title: Plot title
    """
    # Convert to DataFrame
    df = pd.DataFrame(equity_curve)
    df.set_index('timestamp', inplace=True)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot equity curve
    ax1.plot(df.index, df['equity'], 'k-', linewidth=1.5)
    
    # Format x-axis for dates
    date_format = DateFormatter('%Y-%m-%d')
    ax1.xaxis.set_major_formatter(date_format)
    
    # Add title and labels
    ax1.set_title(f"{title}\nFinal Equity: ${performance_metrics['final_equity']:,.2f}, Return: {performance_metrics['total_return_pct']:.2f}%")
    ax1.set_ylabel('Equity ($)')
    ax1.grid(True)
    
    # Plot regime background colors
    regime_colors = {
        MarketRegime.UPTREND.value: 'lightgreen',
        MarketRegime.DOWNTREND.value: 'lightcoral',
        MarketRegime.SIDEWAYS.value: 'lightyellow',
        MarketRegime.VOLATILE.value: 'lightblue',
        MarketRegime.UNKNOWN.value: 'white'
    }
    
    # Plot colored background for regimes
    prev_regime = None
    prev_timestamp = None
    
    for i, row in df.iterrows():
        regime = row['regime']
        
        if regime != prev_regime and prev_timestamp is not None:
            color = regime_colors.get(prev_regime, 'white')
            ax1.axvspan(prev_timestamp, i, facecolor=color, alpha=0.3)
            
        prev_regime = regime
        prev_timestamp = i
    
    # Plot final segment
    if prev_timestamp is not None:
        color = regime_colors.get(prev_regime, 'white')
        ax1.axvspan(prev_timestamp, df.index[-1], facecolor=color, alpha=0.3)
    
    # Add regime transitions on second subplot
    # Convert regimes to numeric values for plotting
    regime_values = {
        MarketRegime.UPTREND.value: 1,
        MarketRegime.SIDEWAYS.value: 0,
        MarketRegime.DOWNTREND.value: -1,
        MarketRegime.VOLATILE.value: 0.5,
        MarketRegime.UNKNOWN.value: 0
    }
    
    df['regime_value'] = [regime_values.get(r, 0) for r in df['regime']]
    
    # Plot regime values
    ax2.plot(df.index, df['regime_value'], 'k-', linewidth=2)
    ax2.fill_between(df.index, df['regime_value'], 0, 
                   where=df['regime_value'] > 0, color='green', alpha=0.3)
    ax2.fill_between(df.index, df['regime_value'], 0, 
                   where=df['regime_value'] < 0, color='red', alpha=0.3)
    
    # Add horizontal lines
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax2.axhline(y=1, color='g', linestyle='--', alpha=0.5)
    ax2.axhline(y=-1, color='r', linestyle='--', alpha=0.5)
    
    # Configure plot
    ax2.set_yticks([1, 0.5, 0, -1])
    ax2.set_yticklabels(['Uptrend', 'Volatile', 'Sideways', 'Downtrend'])
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Regime')
    ax2.grid(True)
    
    # Format x-axis for dates
    ax2.xaxis.set_major_formatter(date_format)
    
    # Add performance metrics as text
    metrics_text = (
        f"Annual Return: {performance_metrics['annual_return_pct']:.2f}%\n"
        f"Max Drawdown: {performance_metrics['max_drawdown_pct']:.2f}%\n"
        f"Sharpe Ratio: {performance_metrics['sharpe_ratio']:.2f}\n"
        f"Sortino Ratio: {performance_metrics['sortino_ratio']:.2f}\n"
        f"Calmar Ratio: {performance_metrics['calmar_ratio']:.2f}\n"
        f"Trades: {performance_metrics['trades']}\n"
        f"Regime Changes: {performance_metrics['regime_transitions']}"
    )
    
    # Add text box with metrics
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.97, metrics_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Add regime-specific performance table
    regime_perf = performance_metrics.get('regime_performance', {})
    
    if regime_perf:
        regime_text = "Regime Performance:\n"
        
        for regime, metrics in regime_perf.items():
            regime_text += f"{regime}: {metrics['total_return_pct']:.2f}%, "
            regime_text += f"Sharpe: {metrics['sharpe']:.2f}, "
            regime_text += f"Bars: {metrics['bars']}\n"
        
        # Add text box with regime performance
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.02, 0.85, regime_text, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the example."""
    print("\n=== Regime-Based Backtesting Example ===\n")
    
    # Configuration
    data_dir = os.path.expanduser("~/data")  # Update with your data directory
    symbols = ["SPY"]
    start_date = "2020-01-01"
    end_date = "2020-12-31"
    timeframe = "1d"
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        data_dir = "data"  # Try relative path
    
    print(f"1. Loading data for {symbols} from {data_dir}...")
    data_source, data_handler, bar_emitter = load_market_data(
        data_dir, symbols, start_date, end_date, timeframe
    )
    
    print("\n2. Setting up regime detector...")
    detector = setup_regime_detector(detector_preset='advanced_sensitive')
    
    print("\n3. Setting up single-strategy backtest...")
    system_components = setup_single_strategy_backtest(symbols, detector)
    
    print("\n4. Running backtest...")
    equity_curve, regime_transitions = run_backtest(
        data_handler, bar_emitter, system_components, detector
    )
    
    print("\n5. Analyzing performance...")
    performance_metrics = analyze_performance(
        equity_curve, regime_transitions, system_components['portfolio']
    )
    
    print("\n6. Visualizing results...")
    visualize_results(
        equity_curve, regime_transitions, performance_metrics, 
        title="Regime-Aware Single Strategy Backtest"
    )
    
    print("\n7. Setting up multi-strategy backtest...")
    # Reset data handler for second backtest
    data_handler.reset()
    detector.reset()
    
    # Setup multi-strategy backtest
    system_components = setup_multi_strategy_backtest(symbols, detector)
    
    print("\n8. Running multi-strategy backtest...")
    equity_curve, regime_transitions = run_backtest(
        data_handler, bar_emitter, system_components, detector
    )
    
    print("\n9. Analyzing multi-strategy performance...")
    performance_metrics = analyze_performance(
        equity_curve, regime_transitions, system_components['portfolio']
    )
    
    print("\n10. Visualizing multi-strategy results...")
    visualize_results(
        equity_curve, regime_transitions, performance_metrics, 
        title="Regime-Based Multi-Strategy Backtest"
    )
    
    print("\nBacktest examples completed! Check the plots for visualization.")


if __name__ == "__main__":
    main()
