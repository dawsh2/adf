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
from src.core.events.event_emitters import BarEmitter

# Import data components
from src.data.historical_data_handler import HistoricalDataHandler
from src.data.sources.csv_handler import CSVDataSource

# Import strategy components
from src.strategy.strategies.ma_crossover import MovingAverageCrossoverStrategy
from src.strategy.strategies.mean_reversion import MeanReversionStrategy
from src.strategy.strategies.momentum import MomentumStrategy

# Import optimization framework
from src.models.optimization.manager import create_optimization_manager
from src.models.filters.regime.detector_factory import RegimeDetectorFactory


def test_order_fill_pipeline():
    """Test that order-fill pipeline works correctly in isolation."""
    import logging
    logging.info("\n===== TESTING ORDER-FILL PIPELINE =====\n")
    
    # Create core components
    from src.core.events.event_bus import EventBus
    from src.core.events.event_types import EventType
    from src.core.events.event_utils import create_order_event
    from src.execution.portfolio import PortfolioManager
    from src.execution.brokers.simulated import SimulatedBroker
    
    # Set up event system
    event_bus = EventBus()
    
    # Create portfolio and broker
    portfolio = PortfolioManager(initial_cash=10000.0)
    portfolio.set_event_bus(event_bus)
    
    broker = SimulatedBroker(fill_emitter=event_bus)
    broker.set_event_bus(event_bus)
    
    # Register portfolio to listen for fills
    event_bus.register(EventType.FILL, portfolio.on_fill)
    
    # Output initial state
    logging.info(f"Initial portfolio: Cash={portfolio.cash}, Positions={len(portfolio.positions)}")
    
    # Create and place a test order
    test_order = create_order_event(
        symbol="TEST", 
        order_type="MARKET",
        direction="BUY",
        quantity=100,
        price=100.0
    )
    
    # Place order directly
    broker.place_order(test_order)
    
    # Output final state
    logging.info(f"Final portfolio: Cash={portfolio.cash}, Positions={len(portfolio.positions)}")
    for symbol, position in portfolio.positions.items():
        logging.info(f"  Position: {symbol} - {position.quantity} shares @ {position.cost_basis:.2f}")
    
    # Return success/failure
    pipeline_working = portfolio.cash != 10000.0 and len(portfolio.positions) > 0
    
    if pipeline_working:
        logging.info("Order-fill pipeline test PASSED! ✓")
    else:
        logging.error("Order-fill pipeline test FAILED! ✗")
    
    return pipeline_working


# Fix 1: Update the data loading function to ensure proper date handling
def load_test_data(symbol='SAMPLE', start_date='2024-03-26', end_date='2024-04-26'):
    """Load historical data for testing."""
    # Convert string dates to datetime objects with proper formatting
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Configure data directory
    data_dir = os.path.join('data')
    
    # Create CSV data source
    data_source = CSVDataSource(data_dir=data_dir)
    
    # Create event bus
    event_bus = EventBus()
    
    # Create bar emitter
    bar_emitter = BarEmitter(name="bar_emitter", event_bus=event_bus)
    
    # Create data handler with proper setup
    data_handler = HistoricalDataHandler(
        data_source=data_source, 
        bar_emitter=bar_emitter
    )
    
    # Load data for the symbol with explicit date handling
    data_handler.load_data(symbols=[symbol], start_date=start_date, end_date=end_date)
    
    logger.info(f"Loaded data for {symbol} from {start_date} to {end_date}")
    return data_handler


def demo_basic_optimization():
    """Demonstrate basic component optimization."""
    logger.info("\n===== BASIC OPTIMIZATION DEMO =====\n")
    
    # 1. Load market data
    data_handler = load_test_data(symbol='SAMPLE')
    
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
    
    # 6. Run optimization
    logger.info("Running grid search optimization...")
    grid_result = manager.optimize_component(
        target_name="ma_strategy",
        optimizer_name="grid",
        evaluator_name="sharpe_ratio",
        param_space=param_space,
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
        data_handler=data_handler,
        start_date='2024-03-26',
        end_date='2024-04-28'
    )
    
    logger.info("\nRandom Search Results:")
    logger.info(f"Best parameters: {random_result.get('best_params')}")
    logger.info(f"Best score: {random_result.get('best_score')}")
    
    return manager, grid_result, random_result


def demo_ensemble_optimization():
    """Demonstrate ensemble strategy optimization."""
    logger.info("\n===== ENSEMBLE OPTIMIZATION DEMO =====\n")
    
    # 1. Load market data
    data_handler = load_test_data(symbol='SAMPLE')
    
    # 2. Create strategies
    ma_strategy = MovingAverageCrossoverStrategy(
        name="ma_crossover",
        symbols=['SAMPLE'],
        fast_window=10,
        slow_window=30
    )
    
    mr_strategy = MeanReversionStrategy(
        name="mean_reversion",
        symbols=['SAMPLE'],
        lookback=20,
        z_threshold=1.5
    )
    
    mom_strategy = MomentumStrategy(
        name="momentum",
        symbols=['SAMPLE'],
        lookback=10,
        threshold=0.01
    )

    ma_strategy.name = "ma_crossover"
    mr_strategy.name = "mean_reversion"
    mom_strategy.name = "momentum"

    
    # 3. Create optimization manager
    manager = create_optimization_manager()
    
    # 4. Register components
    manager.register_target("ma_strategy", ma_strategy)
    manager.register_target("mr_strategy", mr_strategy)
    manager.register_target("mom_strategy", mom_strategy)
    
    # 5. Optimize individual strategies
    logger.info("Optimizing individual strategies...")
    strategies = ["ma_strategy", "mr_strategy", "mom_strategy"]
    results = manager.optimize_multiple(
        targets=strategies,
        optimizer_name="grid",
        evaluator_name="sharpe_ratio",
        data_handler=data_handler,
    )
    
    # 6. Create ensemble with equal weights
    logger.info("\nCreating ensemble strategy with equal weights...")
    ensemble = manager.create_ensemble(
        target_names=strategies,
        ensemble_name="ensemble_strategy"
    )
    
    # 7. Optimize ensemble weights
    logger.info("Optimizing ensemble weights...")
    
    # Define weight parameter space
    weight_space = {
        'weight_ma_crossover': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'weight_mean_reversion': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'weight_momentum': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    }

    # Define weight constraint (ensure weights don't all equal zero)
    def weight_constraint(params):
        total = sum(params.values())
        return total > 0  # As long as some weights are positive, we'll normalize later
    
    ensemble_result = manager.optimize_component(
        target_name="ensemble_strategy",
        optimizer_name="genetic",  # Genetic works well for weights
        evaluator_name="sharpe_ratio",
        param_space=weight_space,
        constraints=[weight_constraint],
        data_handler=data_handler,
        start_date='2018-01-01',
        end_date='2024-04-26',
        population_size=20,
        generations=5  # Using small values for demo purposes
    )
    
    logger.info("\nEnsemble Optimization Results:")
    logger.info(f"Best weights: {ensemble_result.get('best_params')}")
    logger.info(f"Best score: {ensemble_result.get('best_score')}")
    
    return manager, ensemble, ensemble_result


def demo_regime_optimization():
    """Demonstrate regime-aware optimization."""
    logger.info("\n===== REGIME-AWARE OPTIMIZATION DEMO =====\n")
    
    # 1. Load market data
    data_handler = load_test_data(symbol='SAMPLE')
    
    # 2. Create regime detector
    regime_detector = RegimeDetectorFactory.create_detector('enhanced')
    
    # 3. Create strategy
    ma_strategy = MovingAverageCrossoverStrategy(
        name="ma_strategy",
        symbols=['SAMPLE'],
        fast_window=10,
        slow_window=30
    )
    
    # 4. Create regime-aware optimization manager
    manager = create_optimization_manager(
        regime_aware=True,
        regime_detector=regime_detector
    )
    
    # 5. Register components
    manager.register_target("ma_strategy", ma_strategy)
    
    # 6. Define parameter space
    param_space = {
        'fast_window': [5, 10, 15, 20],
        'slow_window': [20, 30, 40, 50]
    }
    
    # 7. Run regime-specific optimization
    logger.info("Running regime-specific optimization...")
    regime_result = manager.optimize_for_regimes(
        target_name="ma_strategy",
        optimizer_name="grid",
        evaluator_name="sharpe_ratio",
        param_space=param_space,
        data_handler=data_handler,
        start_date='2024-03-26',
        end_date='2024-04-26',
        min_regime_bars=20,  # Lower for demo purposes
        min_improvement=0.05
    )
    
    # 8. Display results
    logger.info("\nRegime-Specific Optimization Results:")
    
    # Show baseline results
    baseline_params = regime_result.get('baseline_parameters', {})
    baseline_score = regime_result.get('baseline_score', 0.0)
    logger.info(f"Baseline parameters: {baseline_params}")
    logger.info(f"Baseline score: {baseline_score}")
    
    # Show regime-specific results
    regime_params = regime_result.get('regime_parameters', {})
    for regime, params in regime_params.items():
        regime_name = regime.value if hasattr(regime, 'value') else str(regime)
        logger.info(f"\nParameters for {regime_name} regime:")
        logger.info(f"  {params}")
    
    # 9. Create regime-aware strategy
    logger.info("\nCreating regime-aware strategy...")
    regime_strategy = manager.create_regime_aware_strategy("ma_strategy")
    
    logger.info(f"Created regime-aware strategy: {regime_strategy.name}")
    
    return manager, regime_detector, regime_strategy, regime_result

def demo_walk_forward_optimization():
    """Demonstrate walk-forward optimization."""
    logger.info("\n===== WALK-FORWARD OPTIMIZATION DEMO =====\n")
    
    # 1. Load market data
    data_handler = load_test_data(symbol='SAMPLE')
    
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
    
    # 5. Define parameter space
    param_space = {
        'fast_window': [5, 10, 15, 20],
        'slow_window': [20, 30, 40, 50]
    }
    
    # Add constraint to ensure fast_window < slow_window
    def window_constraint(params):
        return params['fast_window'] < params['slow_window']
    
    # 6. Run walk-forward optimization
    logger.info("Running walk-forward optimization...")
    wf_result = manager.optimize_component(
        target_name="ma_strategy",
        optimizer_name="walk_forward",
        evaluator_name="sharpe_ratio",
        param_space=param_space,
        constraints=[window_constraint],  # Add the constraint
        data_handler=data_handler,
        start_date='2024-03-26',
        end_date='2024-04-26',
        windows=3,  # Use 3 windows for the demo
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


if __name__ == "__main__":
    # Test order-fill pipeline before running main demo
    pipeline_ok = test_order_fill_pipeline()
    
    if not pipeline_ok:
        logging.error("Order-fill pipeline is not working correctly - fix before continuing")
    
    # Continue with the rest of your code
    print("\n" + "="*80)
    print("OPTIMIZATION FRAMEWORK DEMO")
    print("="*80 + "\n")
    
    # Run demonstrations
    basic_manager, grid_result, random_result = demo_basic_optimization()
    ensemble_manager, ensemble, ensemble_result = demo_ensemble_optimization()
    regime_manager, regime_detector, regime_strategy, regime_result = demo_regime_optimization()
    wf_manager, wf_result = demo_walk_forward_optimization()
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80 + "\n")
