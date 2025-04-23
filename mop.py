#!/usr/bin/env python
# simplified_regime_optimization.py - Focus on MA strategy regime optimization

import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import system components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType

from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler
from src.data.transformers.resampler import Resampler

from src.strategy.strategies.ma_crossover import MovingAverageCrossoverStrategy
from src.strategy.risk.risk_manager import SimpleRiskManager
from src.execution.brokers.simulated import SimulatedBroker
from src.execution.portfolio import PortfolioManager
from src.execution.execution_engine import ExecutionEngine

from src.models.filters.regime.regime_detector import EnhancedRegimeDetector, MarketRegime
from src.models.optimization.grid_search import GridSearchOptimizer
from src.models.optimization.regime_optimizer import RegimeSpecificOptimizer

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

def verify_data_path(data_dir, symbol, timeframe='1m'):
    """Check if data file exists and create a placeholder if it doesn't."""
    # Construct expected file path based on CSVDataSource conventions
    file_path = os.path.join(data_dir, f"{symbol}_{timeframe}.csv")
    
    if os.path.exists(file_path):
        logger.info(f"Data file found: {file_path}")
        return True
    
    logger.warning(f"Data file not found: {file_path}")
    return False

def main():
    # Define symbol and timeframe to use
    symbol = 'SPY'
    timeframe = '1m'  # Changed to match your file
    
    # 1. Set up data directory and verify data
    data_dir = "./data"
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        logger.info(f"Creating data directory: {data_dir}")
        os.makedirs(data_dir)
    
    # Verify data file exists
    if not verify_data_path(data_dir, symbol, timeframe):
        logger.error(f"Data file for {symbol} not found. Please make sure the data file exists.")
        logger.info(f"Expected file: {data_dir}/{symbol}_{timeframe}.csv with columns: date,open,high,low,close,volume")
        return
    
    # 2. Set up data
    logger.info("Setting up data...")
    data_source = CSVDataSource(data_dir=data_dir, filename_pattern='{symbol}_{timeframe}.csv')
    data_handler = HistoricalDataHandler(data_source, bar_emitter=None)
    
    # Try to load data - specify the timeframe
    try:
        data_handler.load_data([symbol], start_date='2022-01-01', end_date='2022-12-31', timeframe=timeframe)
        logger.info(f"Data loaded successfully for {symbol}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Verify data was loaded successfully
    symbols = data_handler.get_symbols()
    if not symbols:
        logger.error("No symbols were loaded. Check your data files and format.")
        return
    
    logger.info(f"Successfully loaded data for: {symbols}")
    
    # 3. Configure regime detector
    # Adjust parameters for minute data
    logger.info("Configuring regime detector...")
    regime_detector = EnhancedRegimeDetector(
        lookback_window=120,       # More bars for minute data
        trend_lookback=240,        # More bars for minute data
        volatility_lookback=60,    # More bars for minute data
        trend_threshold=0.01,      # Smaller for minute data
        volatility_threshold=0.005, # Smaller for minute data
        sideways_threshold=0.008   # Smaller for minute data
    )
    
    # 4. Define parameter grid for optimization
    # Adjust windows for minute data
    param_grid = {
        'fast_window': [10, 20, 30, 60],    # Larger for minute data
        'slow_window': [60, 120, 240, 480]  # Larger for minute data
    }
    
    # 5. Configure the optimizer
    logger.info("Setting up optimization...")
    grid_optimizer = GridSearchOptimizer()
    regime_optimizer = RegimeSpecificOptimizer(
        regime_detector=regime_detector,
        grid_optimizer=grid_optimizer
    )
    
    # 6. Run the optimization
    logger.info("Starting regime-based optimization...")
    try:
        result = regime_optimizer.optimize(
            param_grid=param_grid,
            data_handler=data_handler,
            evaluation_func=evaluate_ma_strategy,
            start_date=pd.to_datetime('2022-01-01'),  # Shorter period for minute data
            end_date=pd.to_datetime('2022-12-31'),
            min_regime_bars=300,             # More bars needed for minute data
            optimize_metric='sharpe_ratio'
        )
        
        # 7. Display results
        logger.info("\nOptimization Results:")
        regime_params = result.get('regime_parameters', {})
        baseline_params = result.get('baseline_parameters', {})
        
        logger.info(f"Baseline Parameters: {baseline_params}")
        logger.info(f"Baseline Score: {result.get('baseline_score', 0):.4f}")
        
        logger.info("\nRegime-Specific Parameters:")
        for regime, params in regime_params.items():
            logger.info(f"{regime.value}: {params}")
        
        # 8. Generate comprehensive report
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
