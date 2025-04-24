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

# Import strategy components
from src.strategy.strategies.ma_crossover import MovingAverageCrossoverStrategy
from src.strategy.strategies.mean_reversion import MeanReversionStrategy
from src.strategy.strategies.momentum import MomentumStrategy

# Import execution components
from src.execution.portfolio import PortfolioManager
from src.execution.brokers.simulated import SimulatedBroker
from src.execution.backtest.simulator import MarketSimulator  # This import was missing
from src.execution.backtest.coordinator import BacktestCoordinator  # This import was missing
from src.strategy.risk.risk_manager import SimpleRiskManager

# Import data components
from src.data.historical_data_handler import HistoricalDataHandler
from src.data.sources.csv_handler import CSVDataSource

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
    data_handler = HistoricalDataHandler(data_source=data_source)
    
    # Create portfolio and risk components
    portfolio = PortfolioManager(initial_cash=10000.0)
    portfolio.set_event_bus(event_bus)
    
    # Create broker simulation
    broker = SimulatedBroker()
    broker.set_event_bus(event_bus)
    
    # Create risk manager
    risk_manager = SimpleRiskManager(portfolio)
    risk_manager.set_event_bus(event_bus)
    
    # Create market simulator
    simulator = MarketSimulator(data_handler, event_bus)
    
    # Create backtest coordinator
    coordinator = BacktestCoordinator(
        event_manager=event_manager,
        data_handler=data_handler,
        portfolio=portfolio,
        broker=broker,
        risk_manager=risk_manager,
        simulator=simulator
    )
    
    # Register all components with event manager
    event_manager.register_component('portfolio', portfolio, [EventType.FILL])
    event_manager.register_component('risk_manager', risk_manager, [EventType.SIGNAL])
    event_manager.register_component('broker', broker, [EventType.ORDER])
    
    return {
        'event_bus': event_bus,
        'event_manager': event_manager,
        'data_handler': data_handler,
        'portfolio': portfolio,
        'broker': broker,
        'risk_manager': risk_manager,
        'simulator': simulator,
        'coordinator': coordinator
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
    
    # Create and place a test order
    from src.core.events.event_types import OrderEvent
    order = OrderEvent(
        symbol='TEST',
        order_type='MARKET',
        direction='BUY',
        quantity=100,
        price=100.0
    )
    
    # Place order through event bus to ensure proper event handling
    event_bus.emit(order)
    
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
    
    # 1. Load market data and create environment
    data_handler, env = load_test_data(
        symbol='SAMPLE',
        start_date='2024-03-26',
        end_date='2024-04-26'
    )
    
    event_bus = env['event_bus']
    
    # 2. Create strategy
    ma_strategy = MovingAverageCrossoverStrategy(
        name="ma_strategy",
        symbols=['SAMPLE'],
        fast_window=10,
        slow_window=30
    )
    ma_strategy.set_event_bus(event_bus)
    
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
    
    # Create custom evaluation function that uses our backtest environment
    def evaluate_strategy(strategy, data_handler, start_date, end_date, **kwargs):
        """Evaluate strategy using our properly configured backtest."""
        # Reset environment components
        env['portfolio'].reset()
        env['event_manager'].reset_components()
        
        # Register strategy
        env['event_manager'].register_component('strategy', strategy, [EventType.BAR])
        
        # Run backtest
        results = env['coordinator'].run_backtest(
            symbols=['SAMPLE'],
            start_date=start_date,
            end_date=end_date
        )
        
        # Calculate metrics
        from src.analytics.performance import calculate_sharpe_ratio
        equity_curve = results['equity_curve']
        sharpe = calculate_sharpe_ratio(equity_curve['equity'])
        
        return {'sharpe_ratio': sharpe}
    
    # 6. Run walk-forward optimization
    logger.info("Running walk-forward optimization...")
    wf_result = manager.optimize_component(
        target_name="ma_strategy",
        optimizer_name="walk_forward",
        evaluator_name="custom",  # Use custom evaluator
        evaluation_function=evaluate_strategy,  # Pass custom evaluation function
        param_space=param_space,
        constraints=[window_constraint],
        data_handler=data_handler,
        start_date=pd.to_datetime('2024-03-26'),
        end_date=pd.to_datetime('2024-04-26'),
        windows=3,
        train_size=0.7,
        test_size=0.3
    )
    
    # 7. Display results
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

# Other demo functions remain the same...

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
    
    # Focus on the walk-forward optimization first to debug
    wf_manager, wf_result = demo_walk_forward_optimization()
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80 + "\n")
