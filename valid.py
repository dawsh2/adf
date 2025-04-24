"""
Simple demo script to test optimization validation.
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

# Import directly from validation.py in optimization folder
from src.models.optimization.validation import OptimizationValidator
from src.models.optimization.manager import create_optimization_manager
from src.core.events.event_bus import EventBus
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler
from src.strategy.strategies.ma_crossover import MovingAverageCrossoverStrategy

def load_test_data(symbol='SAMPLE', start_date='2024-03-26', end_date='2024-04-26'):
    """Load historical data for testing."""
    # Convert string dates to datetime objects
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
    from src.core.events.event_emitters import BarEmitter
    bar_emitter = BarEmitter(name="bar_emitter", event_bus=event_bus)
    
    # Create data handler with bar emitter
    data_handler = HistoricalDataHandler(
        data_source=data_source, 
        bar_emitter=bar_emitter
    )
    
    # Load data for the symbol
    data_handler.load_data(symbols=[symbol], start_date=start_date, end_date=end_date)
    
    logger.info(f"Loaded data for {symbol} from {start_date} to {end_date}")
    return data_handler

def run_simple_validation():
    """Run a simple optimization validation."""
    logger.info("Starting simple validation test")
    
    # 1. Set up data
    start_date = pd.to_datetime("2024-03-26")
    end_date = pd.to_datetime("2024-03-28")
    data_handler = load_test_data(symbol='SAMPLE', start_date=start_date, end_date=end_date)
    
    # 2. Create strategy
    strategy = MovingAverageCrossoverStrategy(
        name="ma_strategy",
        symbols=['SAMPLE'],
        fast_window=10,
        slow_window=30
    )
    
    # 3. Create optimization manager
    manager = create_optimization_manager()
    
    # 4. Register strategy with manager
    manager.register_target("ma_strategy", strategy)
    
    # 5. Create validator
    validator = OptimizationValidator(
        optimizer_manager=manager,
        data_handler=data_handler,
        start_date=start_date,
        end_date=end_date
    )
    
    # 6. Define small parameter space for testing
    param_space = {
        'fast_window': [5, 10, 20],
        'slow_window': [20, 30, 50]
    }
    
    # 7. Run validation
    logger.info("Running validation with parameter sweep")
    validation_result = validator.validate_component(
        component_name="ma_strategy",
        param_space=param_space,
        evaluator_name="sharpe_ratio"
    )
    
    # 8. Print report
    report = validator.generate_report()
    print("\nValidation Report:")
    print(report)
    
    # 9. Return results for further analysis
    return validator, validation_result

if __name__ == "__main__":
    logger.info("Starting optimization validation demo")
    validator, results = run_simple_validation()
    logger.info("Validation complete")
