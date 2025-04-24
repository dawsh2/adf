"""
Integration test to verify event connections between system components.

This test focuses specifically on the event chain from signal to order to fill,
ensuring each component correctly processes and forwards events.
"""

import logging
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import necessary components
from src.core.events.event_bus import EventBus
from src.core.events.event_types import EventType, SignalEvent, OrderEvent, FillEvent
from src.core.events.event_utils import create_signal_event
from src.execution.portfolio import PortfolioManager
from src.execution.brokers.simulated import SimulatedBroker
from src.strategy.risk.risk_manager import SimpleRiskManager

def test_event_chain():
    """
    Test that events flow correctly through the system:
    Signal → Risk Manager → Order → Broker → Fill → Portfolio
    """
    logger.info("=== TESTING EVENT SYSTEM INTEGRATION ===")
    
    # Create event system
    event_bus = EventBus()
    
    # Create and connect components
    portfolio = PortfolioManager(initial_cash=10000.0)
    broker = SimulatedBroker(fill_emitter=event_bus)
    risk_manager = SimpleRiskManager(portfolio=portfolio, event_bus=event_bus)
    
    # Set the broker on the risk manager (direct connection)
    risk_manager.broker = broker
    
    # Event counters
    event_counts = {
        'signal': 0,
        'order': 0,
        'fill': 0
    }
    
    # Event trackers
    def track_signal(event):
        if event.get_type() == EventType.SIGNAL:
            event_counts['signal'] += 1
            logger.info(f"Signal received: {event.get_symbol()} {event.get_signal_value()}")
    
    def track_order(event):
        if event.get_type() == EventType.ORDER:
            event_counts['order'] += 1
            logger.info(f"Order generated: {event.get_symbol()} {event.get_direction()} {event.get_quantity()}")
    
    def track_fill(event):
        if event.get_type() == EventType.FILL:
            event_counts['fill'] += 1
            logger.info(f"Fill executed: {event.get_symbol()} {event.get_direction()} {event.get_quantity()} @ {event.get_price()}")
    
    # Register event handlers for monitoring
    event_bus.register(EventType.SIGNAL, track_signal)
    event_bus.register(EventType.ORDER, track_order) 
    event_bus.register(EventType.FILL, track_fill)
    
    # Register components with event bus
    event_bus.register(EventType.SIGNAL, risk_manager.on_signal)
    event_bus.register(EventType.ORDER, broker.place_order)
    event_bus.register(EventType.FILL, portfolio.on_fill)
    
    # Create test market data
    symbol = "TEST"
    price = 100.0
    broker.update_market_data(symbol, {'price': price})
    
    # Create and emit test signal
    signal = create_signal_event(
        signal_value=SignalEvent.BUY,
        price=price,
        symbol=symbol,
        timestamp=datetime.datetime.now()
    )
    
    # Initial portfolio state
    logger.info(f"Initial portfolio - Cash: {portfolio.cash}, Positions: {len(portfolio.positions)}")
    
    # Emit signal
    logger.info(f"Emitting BUY signal for {symbol} @ {price}")
    event_bus.emit(signal)
    
    # Check event flow
    logger.info(f"Event counts: Signals={event_counts['signal']}, Orders={event_counts['order']}, Fills={event_counts['fill']}")
    
    # Validate results
    if event_counts['signal'] == 1 and event_counts['order'] == 1 and event_counts['fill'] == 1:
        logger.info("✓ Complete event chain verified: Signal → Order → Fill")
        
        # Check position was created
        position = portfolio.get_position(symbol)
        if position and position.quantity > 0:
            logger.info(f"✓ Position created: {position.quantity} shares of {symbol}")
            logger.info(f"✓ Portfolio cash updated: {portfolio.cash:.2f}")
            
            # Final portfolio state
            logger.info(f"Final portfolio - Cash: {portfolio.cash}, Position value: {portfolio.get_position_value()}")
            
            return True
        else:
            logger.error("✗ Position was not created in portfolio")
            return False
    else:
        logger.error("✗ Event chain broken!")
        logger.error(f"Missing events: Signals={1-event_counts['signal']}, Orders={1-event_counts['order']}, Fills={1-event_counts['fill']}")
        return False

if __name__ == "__main__":
    success = test_event_chain()
    
    if success:
        print("\nINTEGRATION TEST PASSED!")
        print("Event system is properly connected.")
    else:
        print("\nINTEGRATION TEST FAILED!")
        print("Event system is not properly connected.")
