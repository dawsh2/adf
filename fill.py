#!/usr/bin/env python
"""
Test script to verify the event system fix.
This specifically tests the order-fill pipeline with enhanced logging.
"""
import logging
import sys
import os

# Configure more detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# You may need to adjust this path depending on your project structure
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_order_fill_pipeline():
    """Test that order-fill pipeline works correctly with enhanced logging."""
    logging.info("\n===== TESTING ORDER-FILL PIPELINE =====\n")
    
    # Import core components
    from src.core.events.event_bus import EventBus
    from src.core.events.event_types import EventType, FillEvent
    from src.core.events.event_utils import create_order_event
    from src.execution.portfolio import PortfolioManager
    from src.execution.brokers.simulated import SimulatedBroker
    
    # Set up event system
    event_bus = EventBus()
    
    # Create portfolio and broker with explicit connections
    portfolio = PortfolioManager(initial_cash=10000.0)
    portfolio.set_event_bus(event_bus)
    
    # Create broker with direct fill emitter
    broker = SimulatedBroker(fill_emitter=event_bus)
    broker.set_event_bus(event_bus)
    
    # Register portfolio to listen for fills
    event_bus.register(EventType.FILL, portfolio.on_fill)
    
    # Add debug tracking of all events
    events_tracked = {
        'signals': 0,
        'orders': 0,
        'fills': 0
    }
    
    def track_event(event):
        """Track all events for debugging."""
        event_type = event.get_type()
        if event_type == EventType.SIGNAL:
            events_tracked['signals'] += 1
        elif event_type == EventType.ORDER:
            events_tracked['orders'] += 1
            logging.info(f"Order tracked: {event.get_symbol()} {event.get_direction()} {event.get_quantity()}")
        elif event_type == EventType.FILL:
            events_tracked['fills'] += 1
            logging.info(f"Fill tracked: {event.get_symbol()} {event.get_direction()} {event.get_quantity()}")
    
    # Register tracking for all event types
    for event_type in [EventType.SIGNAL, EventType.ORDER, EventType.FILL]:
        event_bus.register(event_type, track_event)
    
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
    logging.info("Placing test order...")
    broker.place_order(test_order)
    
    # Output final state
    logging.info(f"Final portfolio: Cash={portfolio.cash}, Positions={len(portfolio.positions)}")
    for symbol, position in portfolio.positions.items():
        logging.info(f"  Position: {symbol} - {position.quantity} shares @ {position.cost_basis:.2f}")
    
    # Output event tracking stats
    logging.info(f"Events tracked: {events_tracked}")
    
    # Return success/failure
    pipeline_working = portfolio.cash != 10000.0 and len(portfolio.positions) > 0
    
    if pipeline_working:
        logging.info("Order-fill pipeline test PASSED! ✓")
    else:
        logging.error("Order-fill pipeline test FAILED! ✗")
    
    return pipeline_working

# Add additional test for the full backtest pipeline
def test_backtest_pipeline():
    """Test the complete backtest pipeline with enhanced logging."""
    logging.info("\n===== TESTING BACKTEST PIPELINE =====\n")
    
    import pandas as pd
    
    # Import components needed for backtest
    from src.core.events.event_bus import EventBus
    from src.core.events.event_manager import EventManager
    from src.core.events.event_types import EventType
    from src.data.sources.csv_handler import CSVDataSource
    from src.data.historical_data_handler import HistoricalDataHandler
    from src.strategy.strategies.ma_crossover import MovingAverageCrossoverStrategy
    from src.execution.backtest.backtest import run_backtest
    
    # Create strategy with small windows to ensure signals
    strategy = MovingAverageCrossoverStrategy(
        name="ma_crossover",
        symbols=["SAMPLE"],
        fast_window=5,  # Use small windows to generate more signals
        slow_window=20
    )
    
    # Create data source and handler
    data_dir = os.path.join('data')
    data_source = CSVDataSource(data_dir=data_dir)
    
    # Create data handler
    data_handler = HistoricalDataHandler(
        data_source=data_source,
        bar_emitter=None  # Will be set in backtest
    )
    
    # Set date range - use a known range with data
    start_date = pd.to_datetime("2024-03-26")
    end_date = pd.to_datetime("2024-03-28")
    
    # Load data
    data_handler.load_data(
        symbols=["SAMPLE"],
        start_date=start_date,
        end_date=end_date,
        timeframe="1m"
    )
    
    # Run backtest with enhanced logging
    logging.info(f"Running backtest from {start_date} to {end_date}")
    
    # Define event tracking callback
    fills_executed = []
    
    def track_fills(event):
        if event.get_type() == EventType.FILL:
            fills_executed.append(event)
            logging.info(f"Fill executed during backtest: {event.get_symbol()} {event.get_direction()} {event.get_quantity()}")
    
    # Create a temporary event bus to register our tracking
    temp_bus = EventBus()
    temp_bus.register(EventType.FILL, track_fills)
    
    # Run backtest
    equity_curve, trades = run_backtest(
        component=strategy,
        data_handler=data_handler,
        start_date=start_date,
        end_date=end_date
    )
    
    # Log results
    logging.info(f"Backtest complete! Generated {len(trades)} trades")
    logging.info(f"Initial equity: ${equity_curve['equity'].iloc[0]:.2f}")
    logging.info(f"Final equity: ${equity_curve['equity'].iloc[-1]:.2f}")
    
    # Check if backtest was successful
    backtest_working = len(trades) > 0
    
    if backtest_working:
        logging.info("Backtest pipeline test PASSED! ✓")
    else:
        logging.error("Backtest pipeline test FAILED! ✗")
    
    return backtest_working, equity_curve, trades

if __name__ == "__main__":
    # Run tests
    order_fill_success = test_order_fill_pipeline()
    
    # Only run backtest if order-fill test passes
    if order_fill_success:
        backtest_success, equity_curve, trades = test_backtest_pipeline()
        
        if backtest_success:
            # Calculate simple performance metrics
            initial_equity = equity_curve['equity'].iloc[0] 
            final_equity = equity_curve['equity'].iloc[-1]
            total_return = (final_equity / initial_equity - 1) * 100
            
            print("\n===== TEST RESULTS =====")
            print(f"All pipelines functioning correctly!")
            print(f"Backtest performance: {total_return:.2f}% return")
            print(f"Total trades executed: {len(trades)}")
            print("===== FIX SUCCESSFUL =====\n")
            sys.exit(0)
        else:
            print("\n===== TEST RESULTS =====")
            print(f"Order-fill pipeline fixed, but backtest pipeline still failing")
            print("===== PARTIAL FIX =====\n")
            sys.exit(1)
    else:
        print("\n===== TEST RESULTS =====")
        print(f"Order-fill pipeline still failing")
        print("===== FIX FAILED =====\n")
        sys.exit(1)
