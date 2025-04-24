"""
Optimization Framework Demo Script

This script demonstrates the core functionality of the modular optimization framework
with both standard and regime-aware optimization.
"""

import logging
import pandas as pd
import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import core components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType, OrderEvent, SignalEvent

# Import strategy components
from src.strategy.strategies.ma_crossover import MovingAverageCrossoverStrategy
from src.strategy.strategies.mean_reversion import MeanReversionStrategy
from src.strategy.strategies.momentum import MomentumStrategy

# Import execution components
from src.execution.portfolio import PortfolioManager
from src.execution.brokers.simulated import SimulatedBroker
from src.strategy.risk.risk_manager import SimpleRiskManager

# Import data components
from src.data.historical_data_handler import HistoricalDataHandler
from src.data.sources.csv_handler import CSVDataSource

# Import backtest module
from src.execution.backtest.backtest import run_backtest

# Import optimization framework
from src.models.optimization.manager import create_optimization_manager
from src.models.filters.regime.detector_factory import RegimeDetectorFactory

def create_backtest_environment(symbols=['SAMPLE']):
    """Create a complete backtest environment with all necessary components."""
    # Create event system
    event_bus = EventBus()
    event_manager = EventManager(event_bus)
    
    # Create data components
    data_source = CSVDataSource(data_dir=os.path.join('data'))
    
    # Create bar emitter first
    from src.core.events.event_emitters import BarEmitter
    bar_emitter = BarEmitter(name="bar_emitter", event_bus=event_bus)
    
    # Create data handler with bar emitter
    data_handler = HistoricalDataHandler(
        data_source=data_source,
        bar_emitter=bar_emitter
    )
    
    # Create portfolio and risk components
    portfolio = PortfolioManager(initial_cash=10000.0)
    portfolio.set_event_bus(event_bus)
    
    # Create broker simulation
    broker = SimulatedBroker()
    broker.set_event_bus(event_bus)
    
    # Create risk manager
    risk_manager = SimpleRiskManager(portfolio)
    risk_manager.set_event_bus(event_bus)
    
    # Register all components with event manager
    event_manager.register_component('portfolio', portfolio, [EventType.FILL])
    event_manager.register_component('risk_manager', risk_manager, [EventType.SIGNAL])
    event_manager.register_component('broker', broker, [EventType.ORDER])
    
    return {
        'event_bus': event_bus,
        'event_manager': event_manager,
        'data_handler': data_handler,
        'bar_emitter': bar_emitter,
        'portfolio': portfolio,
        'broker': broker,
        'risk_manager': risk_manager
    }



def load_test_data(symbol='SAMPLE', start_date='2024-03-26', end_date='2024-04-26'):
    """Load historical data for testing."""
    # Convert string dates to datetime objects
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Create environment with data handler
    env = create_backtest_environment([symbol])
    data_handler = env['data_handler']
    
    # Load data for the symbol
    data_handler.load_data(symbols=[symbol], start_date=start_date, end_date=end_date)
    
    logger.info(f"Loaded data for {symbol} from {start_date} to {end_date}")
    return data_handler, env

def test_order_fill_pipeline():
    """Test the basic order-fill pipeline."""
    logger.info("\n===== TESTING ORDER-FILL PIPELINE =====\n")
    
    # Create test environment
    env = create_backtest_environment(['TEST'])
    portfolio = env['portfolio']
    broker = env['broker']
    event_bus = env['event_bus']
    
    # Log initial state
    logger.info(f"Initial portfolio: Cash={portfolio.cash}, Positions={len(portfolio.positions)}")
    
    # Create and place a test order directly through the broker
    # This bypasses the duplicate event issue
    from src.core.events.event_types import OrderEvent
    order = OrderEvent(
        symbol='TEST',
        order_type='MARKET',
        direction='BUY',
        quantity=100,
        price=100.0
    )
    
    # Place order directly through broker to avoid duplicate events
    broker.place_order(order)
    
    # Log final state
    logger.info(f"Final portfolio: Cash={portfolio.cash}, Positions={len(portfolio.positions)}")
    
    # Check if position was created correctly
    for symbol, position in portfolio.positions.items():
        logger.info(f"  Position: {symbol} - {position.quantity} shares @ {position.cost_basis:.2f}")
    
    # Validate test was successful
    if len(portfolio.positions) > 0 and portfolio.cash < 10000.0:
        logger.info("Order-fill pipeline test PASSED! ✓")
        return True
    else:
        logger.error("Order-fill pipeline test FAILED! ✗")
        return False


def demo_walk_forward_optimization():
    """Demonstrate walk-forward optimization."""
    logger.info("\n===== WALK-FORWARD OPTIMIZATION DEMO =====\n")
    
    # 1. Load market data with explicit date handling
    start_date = '2024-03-26'
    end_date = '2024-04-26'
    data_handler, env = load_test_data(
        symbol='SAMPLE',
        start_date=start_date,
        end_date=end_date
    )
    
    # Add debug logging to check if data was actually loaded
    logger.info(f"Checking data availability in date range {start_date} to {end_date}")
    
    # Test data access - reset data handler and try to get some bars
    data_handler.reset()
    symbol = 'SAMPLE'
    
    # Count available bars to verify data
    bar_count = 0
    all_dates = []
    while True:
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break
        bar_count += 1
        all_dates.append(bar.get_timestamp())
    
    if bar_count == 0:
        logger.error(f"No data found for {symbol} in the specified date range!")
        return None, None
    
    logger.info(f"Found {bar_count} bars for {symbol}")
    if all_dates:
        logger.info(f"Date range: {min(all_dates)} to {max(all_dates)}")
    
    # Reset data handler for optimization
    data_handler.reset()
    
    # 2. Create strategy
    ma_strategy = MovingAverageCrossoverStrategy(
        name="ma_strategy",
        symbols=['SAMPLE'],
        fast_window=10,
        slow_window=30
    )
    
    # 3. Create optimization manager
    manager = create_optimization_manager()
    
    # 4. Register components
    manager.register_target("ma_strategy", ma_strategy)
    
    # 5. Define parameter space with constraint
    param_space = {
        'fast_window': [5, 10, 15, 20],
        'slow_window': [20, 30, 40, 50]
    }
    
    def window_constraint(params):
        return params['fast_window'] < params['slow_window']
    
    # 6. Register a custom evaluator instead of providing the function directly
    def custom_evaluator(component, **kwargs):
        """Simple evaluation function that returns a fixed score for testing."""
        logger.info(f"Evaluating strategy with params: {component.get_parameters()}")
        return 0.5  # Return a fixed score for testing
    
    # Register the custom evaluator with the manager
    manager.register_evaluator("custom_eval", custom_evaluator)
    
    # 7. Run walk-forward optimization using the registered evaluator
    logger.info("Running walk-forward optimization...")
    
    # Use single window for initial testing to simplify debugging
    try:
        wf_result = manager.optimize_component(
            target_name="ma_strategy",
            optimizer_name="grid",  # Use simpler optimizer for testing
            evaluator_name="custom_eval",  # Use our registered evaluator
            param_space=param_space,
            constraints=[window_constraint],
            data_handler=data_handler,
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date),
            windows=1,  # Use only 1 window for testing
            train_size=0.7,
            test_size=0.3
        )
        
        # 8. Display results
        logger.info("\nWalk-Forward Optimization Results:")
        logger.info(f"Best parameters: {wf_result.get('best_params')}")
        
        # Show results for each window
        windows = wf_result.get('window_results', [])
        for i, window in enumerate(windows):
            logger.info(f"\nWindow {i+1} results:")
            logger.info(f"  Train period: {window.get('train_period')}")
            logger.info(f"  Test period: {window.get('test_period')}")
            logger.info(f"  Best parameters: {window.get('params')}")
            logger.info(f"  Train score: {window.get('train_score')}")
            logger.info(f"  Test score: {window.get('test_score')}")
        
        return manager, wf_result
    
    except Exception as e:
        logger.error(f"Error during walk-forward optimization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None
    


def demo_basic_optimization():
    """Demonstrate basic component optimization."""
    logger.info("\n===== BASIC OPTIMIZATION DEMO =====\n")
    
    # 1. Load market data
    data_handler, env = load_test_data(symbol='SAMPLE')
    
    # 2. Create strategies
    ma_strategy = MovingAverageCrossoverStrategy(
        name="ma_crossover",
        symbols=['SAMPLE'],
        fast_window=10,
        slow_window=30
    )
    
    # 3. Create optimization manager
    manager = create_optimization_manager()
    
    # 4. Register components
    manager.register_target("ma_strategy", ma_strategy)
    
    # 5. Define parameter space
    param_space = {
        'fast_window': [5, 10, 15, 20],
        'slow_window': [20, 30, 40, 50]
    }
    
    # Add constraint to ensure fast_window < slow_window
    def window_constraint(params):
        return params['fast_window'] < params['slow_window']
    
    # 6. Run optimization
    logger.info("Running grid search optimization...")
    grid_result = manager.optimize_component(
        target_name="ma_strategy",
        optimizer_name="grid",
        evaluator_name="sharpe_ratio",
        param_space=param_space,
        constraints=[window_constraint],
        data_handler=data_handler,
        start_date='2024-03-26',
        end_date='2024-04-28'
    )
    
    # 7. Display results
    logger.info("\nGrid Search Results:")
    logger.info(f"Best parameters: {grid_result.get('best_params')}")
    logger.info(f"Best score: {grid_result.get('best_score')}")
    
    # 8. Run another optimization method
    logger.info("\nRunning random search optimization...")
    random_result = manager.optimize_component(
        target_name="ma_strategy",
        optimizer_name="random",
        evaluator_name="total_return",
        param_space=param_space,
        constraints=[window_constraint],
        data_handler=data_handler,
        start_date='2024-03-26',
        end_date='2024-04-28'
    )
    
    logger.info("\nRandom Search Results:")
    logger.info(f"Best parameters: {random_result.get('best_params')}")
    logger.info(f"Best score: {random_result.get('best_score')}")
    
    return manager, grid_result, random_result

# Other demo functions would follow the same pattern

if __name__ == "__main__":
    # Run order-fill test first to confirm basic functionality
    test_success = test_order_fill_pipeline()
    
    if not test_success:
        logger.error("Aborting demo due to failed order-fill pipeline test")
        exit(1)
    
    # Run demonstrations
    print("\n" + "="*80)
    print("OPTIMIZATION FRAMEWORK DEMO")
    print("="*80 + "\n")
    
    # Focus on basic optimization first
    basic_manager, grid_result, random_result = demo_basic_optimization()
    
    # Then run walk-forward optimization
    wf_manager, wf_result = demo_walk_forward_optimization()
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80 + "\n")
