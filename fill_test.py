"""
Order Fill Test Script

This script tests the order-fill pipeline in isolation to verify that:
1. Orders are being correctly placed
2. Fills are being generated from those orders
3. The portfolio is being updated based on fills
"""

import logging
import datetime
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Using DEBUG level to see detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("order_fill_test")

# Import necessary components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType, SignalEvent, OrderEvent, FillEvent
from src.core.events.event_utils import create_signal_event, create_order_event, create_fill_event, EventTracker

from src.execution.portfolio import PortfolioManager
from src.strategy.risk.risk_manager import SimpleRiskManager
from src.execution.brokers.simulated import SimulatedBroker
from src.execution.brokers.broker_base import BrokerBase


class TestOrderHandler:
    """Simple component that emits orders in response to signals."""
    
    def __init__(self, name="test_order_handler"):
        self.name = name
        self.event_bus = None
        self.orders_created = 0
    
    def set_event_bus(self, event_bus):
        self.event_bus = event_bus
        return self
    
    def on_signal(self, event):
        """Create a test order from a signal."""
        if not isinstance(event, SignalEvent):
            return
        
        # Create simple market order based on signal
        symbol = event.get_symbol()
        price = event.get_price()
        
        # Determine order direction
        if event.get_signal_value() == SignalEvent.BUY:
            direction = "BUY"
        elif event.get_signal_value() == SignalEvent.SELL:
            direction = "SELL"
        else:
            return  # Ignore neutral signals
        
        # Create order
        order = create_order_event(
            symbol=symbol,
            order_type="MARKET",
            direction=direction,
            quantity=100,  # Fixed quantity for testing
            price=price
        )
        
        # Emit order
        if self.event_bus:
            self.event_bus.emit(order)
            self.orders_created += 1
            logger.info(f"Order created: {order.get_symbol()} {order.get_direction()} {order.get_quantity()} @ {order.get_price()}")


def run_order_fill_test():
    """Run the order-fill test."""
    logger.info("Starting order-fill pipeline test")
    
    # 1. Create components
    event_bus = EventBus()
    event_manager = EventManager(event_bus)
    
    # Create event tracker to monitor events
    tracker = EventTracker(name="test_tracker", verbose=True)
    
    # Register tracker with all event types
    for event_type in EventType:
        event_bus.register(event_type, tracker.track_event)
    
    # 2. Create portfolio and execution components
    portfolio = PortfolioManager(initial_cash=10000.0)
    
    # Create broker with explicit fill emitter reference
    broker = SimulatedBroker(fill_emitter=event_bus)
    
    # Create test order handler (instead of using risk manager for simplicity)
    order_handler = TestOrderHandler()
    
    # 3. Set event bus for all components
    portfolio.set_event_bus(event_bus)
    broker.set_event_bus(event_bus)
    order_handler.set_event_bus(event_bus)
    
    # 4. Register components with event manager
    event_manager.register_component('portfolio', portfolio, [EventType.FILL])
    event_manager.register_component('order_handler', order_handler, [EventType.SIGNAL])
    
    # 5. Create and emit test signals
    symbols = ['AAPL', 'MSFT', 'GOOG']
    signals_count = 0
    
    for symbol in symbols:
        # Create a buy signal
        buy_signal = create_signal_event(
            signal_value=SignalEvent.BUY,
            price=100.0,
            symbol=symbol,
            rule_id="test_rule"
        )
        
        # Emit the signal
        event_bus.emit(buy_signal)
        signals_count += 1
        logger.info(f"Signal emitted: {symbol} BUY @ 100.0")
        
        # Manually place the order with the broker (simulating what would happen in execution engine)
        # This step helps isolate if the issue is with the broker or the event handling
        if tracker.get_event_count(EventType.ORDER) > 0:
            order = tracker.get_last_event(EventType.ORDER)
            logger.info(f"Manually placing order with broker: {order.get_symbol()} {order.get_direction()}")
            broker.place_order(order)
    
    # 6. Check results
    # Count events
    signals = tracker.get_event_count(EventType.SIGNAL)
    orders = tracker.get_event_count(EventType.ORDER)
    fills = tracker.get_event_count(EventType.FILL)
    
    logger.info("\nTest Results:")
    logger.info(f"Signals created: {signals_count}")
    logger.info(f"Signals tracked: {signals}")
    logger.info(f"Orders created: {order_handler.orders_created}")
    logger.info(f"Orders tracked: {orders}")
    logger.info(f"Fills generated: {fills}")
    
    # Check portfolio
    logger.info(f"Portfolio initial cash: 10000.0")
    logger.info(f"Portfolio current cash: {portfolio.cash}")
    logger.info(f"Portfolio positions: {len(portfolio.positions)}")
    
    for symbol, position in portfolio.positions.items():
        logger.info(f"  {symbol}: {position.quantity} shares @ {position.cost_basis}")
    
    # Evaluate success/failure
    if signals > 0 and orders > 0 and fills > 0 and len(portfolio.positions) > 0:
        logger.info("\nTest PASSED: The entire order-fill pipeline is working!")
    else:
        logger.info("\nTest FAILED: Issues detected in the order-fill pipeline.")
        
        # Diagnose the specific issue
        if signals == 0:
            logger.error("Signals were not properly created or emitted")
        elif orders == 0:
            logger.error("Orders were not generated from signals")
        elif fills == 0:
            logger.error("Fills were not generated from orders - likely broker issue")
        elif len(portfolio.positions) == 0:
            logger.error("Portfolio not updated from fills - likely portfolio issue")
    
    return {
        'signals': signals,
        'orders': orders,
        'fills': fills,
        'positions': len(portfolio.positions),
        'portfolio': portfolio
    }


# Run the test if executed directly
if __name__ == "__main__":
    results = run_order_fill_test()
    
    # Check if SimulatedBroker is properly emitting fills
    if results['orders'] > 0 and results['fills'] == 0:
        logger.error("\nDetailed diagnosis:")
        logger.error("The SimulatedBroker is not properly emitting fill events.")
        logger.error("Check the following:")
        logger.error("1. Is emit_fill being called in place_order?")
        logger.error("2. Is the event_bus properly set on the broker?")
        logger.error("3. Is FillEvent being created correctly?")
    
    # Check if portfolio is processing fills
    if results['fills'] > 0 and results['positions'] == 0:
        logger.error("\nDetailed diagnosis:")
        logger.error("Fills are being generated but not processed by the Portfolio.")
        logger.error("Check the following:")
        logger.error("1. Is the Portfolio correctly registered for FILL events?")
        logger.error("2. Is the on_fill method correctly updating positions?")
