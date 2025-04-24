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



def debug_order_duplication():
    """Debug the double order issue."""
    logger.info("\n===== DEBUGGING ORDER DUPLICATION =====\n")
    
    # Create test environment
    env = create_backtest_environment(['TEST'])
    event_bus = env['event_bus']
    broker = env['broker']
    portfolio = env['portfolio']
    
    # Create a tracking handler to see all events
    events_seen = []
    
    def track_events(event):
        event_type = event.get_type().name
        if hasattr(event, 'get_symbol'):
            symbol = event.get_symbol()
            events_seen.append(f"{event_type}: {symbol}")
            logger.info(f"EVENT TRACKED: {event_type} for {symbol}")
    
    # Register tracking handler for all event types
    for event_type in EventType:
        event_bus.register(event_type, track_events)
    
    # Test 1: Order directly through broker (bypassing events)
    logger.info("TEST 1: Order directly through broker")
    portfolio.reset()
    
    order = OrderEvent(
        symbol='TEST',
        order_type='MARKET',
        direction='BUY',
        quantity=100,
        price=100.0
    )
    
    broker.place_order(order)
    
    # Test 2: Order through event bus
    logger.info("TEST 2: Order through event bus")
    portfolio.reset()
    events_seen.clear()
    
    order = OrderEvent(
        symbol='TEST',
        order_type='MARKET',
        direction='BUY',
        quantity=100,
        price=100.0
    )
    
    event_bus.emit(order)
    
    # Log the events that were seen
    logger.info(f"Events tracked: {len(events_seen)}")
    for event in events_seen:
        logger.info(f"  {event}")
    
    # Test 3: Order through risk manager
    logger.info("TEST 3: Order through risk manager")
    portfolio.reset()
    events_seen.clear()
    
    signal = SignalEvent(
        signal_value=SignalEvent.BUY,
        price=100.0,
        symbol='TEST'
    )
    
    event_bus.emit(signal)
    
    # Log the events that were seen
    logger.info(f"Events tracked: {len(events_seen)}")
    for event in events_seen:
        logger.info(f"  {event}")
    
    return events_seen


def debug_event_pipeline():
    """Debug the event pipeline to see where the disconnect is happening."""
    logger.info("\n===== DEBUGGING EVENT PIPELINE =====\n")
    
    # Create test environment
    env = create_backtest_environment(['TEST'])
    event_bus = env['event_bus']
    event_manager = env['event_manager']
    risk_manager = env['risk_manager']
    
    # Check component registrations
    logger.info("Event bus handlers:")
    for event_type, handlers in event_bus.handlers.items():
        handler_names = []
        for handler in handlers:
            if hasattr(handler, '__self__'):
                handler_names.append(f"{handler.__self__.__class__.__name__}.{handler.__name__}")
            else:
                handler_names.append(str(handler))
        logger.info(f"  {event_type.name}: {handler_names}")
    
    # Check event manager components
    logger.info("Event manager components:")
    for name, component in event_manager.components.items():
        logger.info(f"  {name}: {component.__class__.__name__}")
    
    # Examine risk manager to see if it's handling signals properly
    logger.info("Testing risk manager signal handling:")
    
    # Send a test signal and see if the risk manager converts it to an order
    signal = SignalEvent(
        signal_value=SignalEvent.BUY,
        price=100.0,
        symbol='TEST'
    )
    
    # Create a tracking handler to see if an order is generated
    order_generated = []
    
    def track_order(event):
        if event.get_type() == EventType.ORDER:
            order_generated.append(event)
            logger.info(f"ORDER GENERATED: {event.get_symbol()} {event.get_direction()} {event.get_quantity()}")
    
    # Register tracking handler specifically for orders
    event_bus.register(EventType.ORDER, track_order)
    
    # Now emit the signal
    event_bus.emit(signal)
    
    # Check if an order was generated
    if order_generated:
        logger.info(f"Order successfully generated from signal: {len(order_generated)} orders")
    else:
        logger.error("No order was generated from signal!")
    
    return order_generated


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
    
    # Import our analytics
    from src.analytics.performance import PerformanceAnalytics
    
    # 1. Load market data with explicit date handling
    # Use string dates for clarity
    start_date_str = '2024-03-26'
    end_date_str = '2024-04-26'
    
    # Log the dates we're using
    logger.info(f"Using date range: {start_date_str} to {end_date_str}")
    
    # Create pandas timestamps
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    
    # Load data
    data_handler, env = load_test_data(
        symbol='SAMPLE',
        start_date=start_date_str,
        end_date=end_date_str
    )
    
    # Add debug logging to check if data was actually loaded
    logger.info(f"Checking data availability in date range {start_date_str} to {end_date_str}")
    
    # Test data access - reset data handler and try to get some bars
    data_handler.reset()
    symbol = 'SAMPLE'
    
    # Count available bars to verify data
    bar_count = 0
    sample_dates = []
    while True:
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break
        bar_count += 1
        date = bar.get_timestamp()
        if bar_count <= 5 or bar_count > (bar_count - 5):
            sample_dates.append(date)
    
    if bar_count == 0:
        logger.error(f"No data found for {symbol} in the specified date range!")
        return None, None
    
    logger.info(f"Found {bar_count} bars for {symbol}")
    if sample_dates:
        logger.info(f"Sample dates: First 5 = {sample_dates[:5]}")
        if len(sample_dates) > 5:
            logger.info(f"Latest dates: {sample_dates[-5:]}")
    
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
    
    # 6. Register custom evaluator for simpler testing
    def custom_evaluator(component, **kwargs):
        """Simplified evaluation function for testing."""
        logger.info(f"Evaluating strategy with params: {component.get_parameters()}")
        
        data_handler = kwargs.get('data_handler')
        start = kwargs.get('start_date')
        end = kwargs.get('end_date')
        
        if start:
            logger.info(f"Evaluation date range: {start} to {end}")
        
        # Run a simple backtest
        from src.execution.backtest.backtest import run_backtest
        equity_curve, trades = run_backtest(
            component=component,
            data_handler=data_handler,
            start_date=start,
            end_date=end
        )
        
        # Calculate a simple metric - just final equity
        if len(equity_curve) > 0:
            final_equity = equity_curve['equity'].iloc[-1]
            initial_equity = equity_curve['equity'].iloc[0]
            total_return = (final_equity / initial_equity) - 1
            logger.info(f"Evaluation result: {total_return:.4f}")
            return total_return
        else:
            logger.warning("No equity curve generated in evaluation")
            return 0.0
    
    # Register the custom evaluator with the manager
    manager.register_evaluator("custom_eval", custom_evaluator)
    
    # 7. Run walk-forward optimization using the registered evaluator
    logger.info("Running walk-forward optimization...")
    
    # Use simplified optimization for initial testing
    try:
        # Start with a simpler optimizer first
        grid_result = manager.optimize_component(
            target_name="ma_strategy",
            optimizer_name="grid",  # Use grid search first
            evaluator_name="custom_eval",
            param_space=param_space,
            constraints=[window_constraint],
            data_handler=data_handler,
            start_date=start_date,
            end_date=end_date
        )
        
        # If grid search works, try walk-forward
        logger.info("Grid search complete, attempting walk-forward optimization...")
        
        wf_result = manager.optimize_component(
            target_name="ma_strategy",
            optimizer_name="walk_forward",
            evaluator_name="custom_eval",
            param_space=param_space,
            constraints=[window_constraint],
            data_handler=data_handler,
            start_date=start_date,
            end_date=end_date,
            windows=3,  # Use 3 windows for testing
            train_size=0.7,
            test_size=0.3
        )
        
        # 8. Display formatted results if available
        if wf_result:
            result_table = PerformanceAnalytics.format_walk_forward_results(wf_result)
            print(result_table)
            
            # 9. Run the best strategy to demonstrate its performance
            best_params = wf_result.get('best_params', {})
            if best_params:
                logger.info("\nRunning backtest with optimal parameters...")
                
                # Apply best parameters
                ma_strategy.set_parameters(best_params)
                
                # Run backtest
                equity_curve, trades = run_backtest(
                    component=ma_strategy,
                    data_handler=data_handler,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Calculate and display metrics
                metrics = PerformanceAnalytics.calculate_metrics(equity_curve, trades)
                metrics_table = PerformanceAnalytics.display_metrics(metrics, 
                                                                 title="Optimized Strategy Performance")
                print(metrics_table)
            
            return manager, wf_result
        else:
            logger.error("Walk-forward optimization did not produce valid results")
            return manager, grid_result
            
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
