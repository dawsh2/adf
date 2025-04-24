#!/usr/bin/env python
"""
Script to verify the execution pipeline fixes
"""
import logging
import sys
import os
import pandas as pd
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType, BarEvent

# Import data components
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler

# Import execution components
from src.execution.portfolio import PortfolioManager
from src.execution.brokers.simulated import SimulatedBroker 
from src.strategy.risk.risk_manager import SimpleRiskManager

# Import example strategy (assuming a simple moving average crossover strategy)
from src.strategy.strategies.ma_crossover import MovingAverageCrossoverStrategy

def test_execution_pipeline():
    """Test that the execution pipeline works correctly."""
    logger.info("Starting execution pipeline test")
    
    # Create event system
    event_bus = EventBus()
    event_manager = EventManager(event_bus)
    
    # Create data source
    data_dir = "data"  # Adjust path as needed
    csv_source = CSVDataSource(
        data_dir=data_dir,
        date_column="timestamp",  # Adjust based on your CSV structure
        filename_pattern="{symbol}_{timeframe}.csv"
    )
    
    # Create portfolio
    portfolio = PortfolioManager(initial_cash=10000.0)
    
    # Create broker
    broker = SimulatedBroker(fill_emitter=event_bus)
    
    # Create risk manager with direct broker connection
    risk_manager = SimpleRiskManager(portfolio, event_bus)
    risk_manager.broker = broker  # Direct connection to broker
    
    # Create strategy
    strategy = MovingAverageCrossoverStrategy(
        name="ma_crossover",
        symbols=["SAMPLE"],
        fast_window=5,
        slow_window=20
    )
    
    # Create data handler
    data_handler = HistoricalDataHandler(
        data_source=csv_source,
        bar_emitter=event_bus
    )
    
    # Set event bus for components
    portfolio.set_event_bus(event_bus)
    strategy.set_event_bus(event_bus)
    
    # Register components with event manager
    event_manager.register_component('strategy', strategy, [EventType.BAR])
    event_manager.register_component('risk_manager', risk_manager, [EventType.SIGNAL])
    event_manager.register_component('portfolio', portfolio, [EventType.FILL])
    
    # Track events
    event_counts = {'bars': 0, 'signals': 0, 'orders': 0, 'fills': 0}
    
    def track_event(event):
        event_type = event.get_type()
        
        if event_type == EventType.BAR:
            event_counts['bars'] += 1
        elif event_type == EventType.SIGNAL:
            event_counts['signals'] += 1
            logger.info(f"Signal: {event.get_symbol()} {event.get_signal_value()}")
        elif event_type == EventType.ORDER:
            event_counts['orders'] += 1
            logger.info(f"Order: {event.get_symbol()} {event.get_direction()} {event.get_quantity()}")
            
            # Direct order execution for testing
            broker.place_order(event)
        elif event_type == EventType.FILL:
            event_counts['fills'] += 1
            logger.info(f"Fill: {event.get_symbol()} {event.get_direction()} {event.get_quantity()} @ {event.get_price()}")
    
    # Register event tracking
    for event_type in [EventType.BAR, EventType.SIGNAL, EventType.ORDER, EventType.FILL]:
        event_bus.register(event_type, track_event)
    
    # Load data
    logger.info("Loading data...")
    start_date = pd.to_datetime("2024-03-26")
    end_date = pd.to_datetime("2024-03-27")
    
    data_handler.load_data(
        symbols=["SAMPLE"],
        start_date=start_date,
        end_date=end_date,
        timeframe="1m"
    )
    
    # Reset components
    strategy.reset()
    risk_manager.reset()
    portfolio.reset()
    data_handler.reset()
    
    # Run simulation
    logger.info("Running simulation...")
    symbol = "SAMPLE"
    
    bar_count = 0
    while True:
        # Get next bar
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break
        
        bar_count += 1
        
        # Update market data in broker
        market_data = {
            'price': bar.get_close(),
            'timestamp': bar.get_timestamp()
        }
        broker.update_market_data(symbol, market_data)
        
        # Process bar
        event_bus.emit(bar)
        
        # Log progress
        if bar_count % 100 == 0:
            logger.info(f"Processed {bar_count} bars")
    
    # Log results
    logger.info(f"Simulation complete - processed {bar_count} bars")
    logger.info(f"Event counts: {event_counts}")
    logger.info(f"Portfolio cash: {portfolio.cash}")
    logger.info(f"Portfolio positions: {len(portfolio.positions)}")
    
    # Check positions
    if portfolio.positions:
        logger.info("Final positions:")
        for sym, position in portfolio.positions.items():
            logger.info(f"  {sym}: {position.quantity} shares @ {position.cost_basis:.2f}")
    
    # Verification result
    pipeline_working = event_counts['fills'] > 0
    if pipeline_working:
        logger.info("PASSED: Execution pipeline is working correctly!")
    else:
        logger.error("FAILED: No fills were generated!")
    
    return pipeline_working

if __name__ == "__main__":
    result = test_execution_pipeline()
    sys.exit(0 if result else 1)
