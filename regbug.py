#!/usr/bin/env python
# regime_optimization_fixed.py

import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import system components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType

from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler
from src.data.datetime_utils import parse_timestamp, ensure_timezone_aware

from src.models.filters.regime.regime_detector import EnhancedRegimeDetector, MarketRegime
from src.models.optimization.grid_search import GridSearchOptimizer
from src.models.optimization.regime_optimizer import RegimeSpecificOptimizer

from src.strategy.strategies.ma_crossover import MovingAverageCrossoverStrategy
from src.strategy.risk.risk_manager import SimpleRiskManager
from src.execution.brokers.simulated import SimulatedBroker
from src.execution.portfolio import PortfolioManager
from src.execution.execution_engine import ExecutionEngine

from src.analytics.performance.calculator import PerformanceCalculator

def evaluate_ma_strategy(params, data_handler, start_date=None, end_date=None):
    """Evaluate MA strategy with specific parameters."""
    # Verify we have data to work with
    symbols = data_handler.get_symbols()
    if not symbols:
        logger.error("No symbols available in data handler")
        return {'sharpe_ratio': -999, 'total_return_pct': -999}
    
    symbol = symbols[0]
    
    # Create event system
    event_bus = EventBus()
    event_manager = EventManager(event_bus)
    
    # Set up portfolio and broker
    portfolio = PortfolioManager(initial_cash=10000.0)
    broker = SimulatedBroker(fill_emitter=event_bus)
    
    # Set up risk manager and execution
    risk_manager = SimpleRiskManager(portfolio, event_bus=event_bus)
    execution_engine = ExecutionEngine(risk_manager, broker, order_emitter=event_bus)
    
    # Create strategy with parameters
    strategy = MovingAverageCrossoverStrategy(
        symbols=symbol,
        fast_window=params.get('fast_window', 10),
        slow_window=params.get('slow_window', 30)
    )
    
    # Connect to event system
    strategy.set_event_bus(event_bus)
    risk_manager.set_event_bus(event_bus)
    broker.set_event_bus(event_bus)
    portfolio.set_event_bus(event_bus)
    execution_engine.set_event_bus(event_bus)
    
    event_manager.register_component("portfolio", portfolio, [EventType.FILL])
    event_manager.register_component("risk_manager", risk_manager, [EventType.SIGNAL])
    event_manager.register_component("execution", execution_engine, [EventType.ORDER])
    event_manager.register_component("strategy", strategy, [EventType.BAR])
    
    # Reset components
    data_handler.reset()
    portfolio.reset()
    strategy.reset()
    
    # Process start_date and end_date
    if start_date:
        start_date = ensure_timezone_aware(parse_timestamp(start_date) if isinstance(start_date, str) else start_date)
    if end_date:
        end_date = ensure_timezone_aware(parse_timestamp(end_date) if isinstance(end_date, str) else end_date)
    
    # Track equity curve
    equity_curve = [{
        'timestamp': datetime.now(),
        'equity': portfolio.get_equity()
    }]
    
    # Run backtest
    while True:
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break
            
        # Apply date filtering
        timestamp = bar.get_timestamp()
        if start_date and timestamp < start_date:
            continue
        if end_date and timestamp > end_date:
            break
            
        # Record equity point
        equity_curve.append({
            'timestamp': timestamp,
            'equity': portfolio.get_equity()
        })
    
    # Calculate performance metrics
    calculator = PerformanceCalculator()
    metrics = calculator.calculate(equity_curve)
    
    # Log basic results
    logger.debug(f"Parameters: {params}, Sharpe: {metrics['sharpe_ratio']:.4f}, Return: {metrics['total_return_pct']:.2f}%")
    
    # Return performance metrics
    return metrics

def test_regime_detection():
    """Test the regime detection alone to verify it's working properly."""
    # Define symbol and timeframe
    symbol = 'SPY'
    timeframe = '1m'
    
    # Set up data
    data_dir = "./data"
    data_source = CSVDataSource(data_dir=data_dir, filename_pattern='{symbol}_{timeframe}.csv')
    data_handler = HistoricalDataHandler(data_source, bar_emitter=None)
    
    # Load data
    try:
        data_handler.load_data([symbol], timeframe=timeframe)
        logger.info(f"Loaded data for {symbol}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Configure regime detector
    regime_detector = EnhancedRegimeDetector(
        lookback_window=120, 
        trend_lookback=240,
        volatility_lookback=60,
        trend_threshold=0.01,
        volatility_threshold=0.005,
        sideways_threshold=0.008,
        debug=True
    )
    
    # Process bars to detect regimes
    data_handler.reset()
    
    bar_count = 0
    regime_changes = 0
    last_regime = None
    
    # Process first 1000 bars
    while bar_count < 1000:
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break
        
        # Process bar and update detector
        current_regime = regime_detector.update(bar)
        
        # Track regime changes
        if last_regime != current_regime:
            regime_changes += 1
            logger.info(f"Regime change at bar {bar_count}: {last_regime} -> {current_regime.value}")
            last_regime = current_regime
        
        if bar_count % 100 == 0:
            logger.info(f"Processed {bar_count} bars, current regime: {current_regime.value}")
            # Print timestamp to verify it's correct
            logger.info(f"Current timestamp: {bar.get_timestamp()}")
        
        bar_count += 1
    
    logger.info(f"Detected {regime_changes} regime changes")
    
    # Print regime distribution
    if symbol in regime_detector.regime_history:
        regime_counts = {}
        for _, regime in regime_detector.regime_history[symbol]:
            regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1
        
        logger.info("Regime distribution:")
        for regime, count in regime_counts.items():
            pct = (count / bar_count) * 100 if bar_count > 0 else 0
            logger.info(f"  {regime}: {count} bars ({pct:.1f}%)")
    
    return regime_detector.get_regime_periods(symbol)

def main():
    """Main function to run the optimization."""
    # First test regime detection to make sure it works
    logger.info("Testing regime detection...")
    regime_periods = test_regime_detection()
    
    if not regime_periods:
        logger.error("Regime detection failed. Cannot proceed with optimization.")
        return
    
    logger.info(f"Detected regimes: {list(regime_periods.keys())}")
    
    # Define symbol and timeframe
    symbol = 'SPY'
    timeframe = '1m'
    
    # Set up data
    data_dir = "./data"
    data_source = CSVDataSource(data_dir=data_dir, filename_pattern='{symbol}_{timeframe}.csv')
    data_handler = HistoricalDataHandler(data_source, bar_emitter=None)
    
    # Load data
    try:
        data_handler.load_data([symbol], timeframe=timeframe)
        logger.info(f"Loaded data for {symbol}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Configure regime detector
    regime_detector = EnhancedRegimeDetector(
        lookback_window=120, 
        trend_lookback=240,
        volatility_lookback=60,
        trend_threshold=0.01,
        volatility_threshold=0.005,
        sideways_threshold=0.008
    )
    
    # Define parameter grid
    param_grid = {
        'fast_window': [10, 20, 30, 60],
        'slow_window': [60, 120, 240, 480]
    }
    
    # Configure optimizer
    logger.info("Setting up optimization...")
    grid_optimizer = GridSearchOptimizer()
    regime_optimizer = RegimeSpecificOptimizer(
        regime_detector=regime_detector,
        grid_optimizer=grid_optimizer
    )
    
    # Run optimization
    logger.info("Starting regime-based optimization...")
    try:
        result = regime_optimizer.optimize(
            param_grid=param_grid,
            data_handler=data_handler,
            evaluation_func=evaluate_ma_strategy,
            min_regime_bars=240,  # Higher for minute data
            optimize_metric='sharpe_ratio'
        )
        
        # Display results
        logger.info("\nOptimization Results:")
        regime_params = result.get('regime_parameters', {})
        baseline_params = result.get('baseline_parameters', {})
        
        logger.info(f"Baseline Parameters: {baseline_params}")
        logger.info(f"Baseline Score: {result.get('baseline_score', 0):.4f}")
        
        logger.info("\nRegime-Specific Parameters:")
        for regime, params in regime_params.items():
            logger.info(f"{regime.value}: {params}")
        
        # Generate comprehensive report
        logger.info("\nGenerating comprehensive report...")
        report = regime_optimizer.generate_report(data_handler, symbol)
        print(report)
        
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("Optimization complete")

if __name__ == "__main__":
    main()
