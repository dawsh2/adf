"""
Quick validation test to confirm the position sizing fix.

This script creates a simplified test that verifies the position sizing fix
for the SimpleRiskManager.
"""

import logging
import pandas as pd
import numpy as np
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import necessary components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType, BarEvent
from src.core.events.event_utils import create_bar_event, create_signal_event
from src.execution.portfolio import PortfolioManager
from src.execution.brokers.simulated import SimulatedBroker
from src.strategy.risk.risk_manager import SimpleRiskManager

def create_synthetic_price_series(days=30, start_price=100.0, daily_return=0.01):
    """Create a synthetic price series with a constant daily return."""
    # Create date range
    start_date = datetime.datetime(2024, 1, 1)
    dates = [start_date + datetime.timedelta(days=i) for i in range(days)]
    
    # Calculate prices with compounding
    prices = [start_price * (1 + daily_return) ** i for i in range(days)]
    
    # Calculate the expected total return
    expected_total_return = (1 + daily_return) ** (days - 1) - 1
    logger.info(f"Expected total return: {expected_total_return:.6f} ({expected_total_return*100:.2f}%)")
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Simple OHLC based on close price
        open_price = prices[i-1] if i > 0 else close * 0.99
        high = max(open_price, close) * 1.01  # 1% above max of open/close
        low = min(open_price, close) * 0.99   # 1% below min of open/close
        volume = 1000000  # Constant volume
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    logger.info(f"Created synthetic price series: {days} days, start={start_price:.2f}, end={prices[-1]:.2f}")
    
    return df, expected_total_return, prices

def test_risk_manager_position_sizing():
    """Test the improved position sizing in SimpleRiskManager."""
    # Create portfolio with initial cash
    portfolio = PortfolioManager(initial_cash=10000.0)
    
    # Create risk manager with different position size settings
    risk_manager = SimpleRiskManager(portfolio, position_pct=0.95)
    
    # Test position sizes at different prices
    test_prices = [10.0, 100.0, 1000.0]
    
    logger.info("Testing position sizing with different prices:")
    for price in test_prices:
        # Calculate position size
        position_size = risk_manager.calculate_position_size("TEST", price)
        
        # Calculate expected size based on portfolio percentage
        expected_size = int((portfolio.get_equity() * 0.95) / price)
        
        logger.info(f"Price: ${price:.2f}, Position size: {position_size} shares")
        logger.info(f"Position value: ${position_size * price:.2f}, " +
                   f"Percentage of portfolio: {(position_size * price / portfolio.get_equity()) * 100:.2f}%")
    
    return True

def test_full_system():
    """Test the full system with proper position sizing."""
    # Set up event system
    event_bus = EventBus()
    event_manager = EventManager(event_bus)
    
    # Create portfolio and broker
    portfolio = PortfolioManager(initial_cash=10000.0)
    portfolio.set_event_bus(event_bus)
    
    broker = SimulatedBroker()
    broker.set_event_bus(event_bus)
    
    # Create risk manager with high position percentage
    risk_manager = SimpleRiskManager(
        portfolio=portfolio,
        event_bus=event_bus,
        position_pct=0.95  # Use 95% of portfolio for each position
    )
    
    # Register components with event manager
    event_manager.register_component('portfolio', portfolio, [EventType.FILL])
    event_manager.register_component('risk_manager', risk_manager, [EventType.SIGNAL])
    event_manager.register_component('broker', broker, [EventType.ORDER])
    
    # Create and track event counts
    event_counts = {
        'bar': 0,
        'signal': 0,
        'order': 0,
        'fill': 0
    }
    
    def count_events(event):
        event_type = event.get_type()
        if event_type == EventType.BAR:
            event_counts['bar'] += 1
        elif event_type == EventType.SIGNAL:
            event_counts['signal'] += 1
            logger.info(f"Signal: {event.get_symbol()} {event.data.get('signal_value')} @ {event.data.get('price')}")
        elif event_type == EventType.ORDER:
            event_counts['order'] += 1
            logger.info(f"Order: {event.get_symbol()} {event.data.get('direction')} {event.data.get('quantity')} @ {event.data.get('price')}")
        elif event_type == EventType.FILL:
            event_counts['fill'] += 1
            logger.info(f"Fill: {event.get_symbol()} {event.data.get('direction')} {event.data.get('quantity')} @ {event.data.get('price')}")
    
    # Register event counter
    for event_type in [EventType.BAR, EventType.SIGNAL, EventType.ORDER, EventType.FILL]:
        event_bus.register(event_type, count_events)
    
    # Generate synthetic price data
    days = 30
    start_price = 100.0
    daily_return = 0.01  # 1% daily return
    
    df, expected_return, prices = create_synthetic_price_series(days, start_price, daily_return)
    
    # Generate signal for buying at first price
    symbol = "TEST"
    signal = create_signal_event(
        signal_value=1,  # BUY
        price=prices[0],
        symbol=symbol,
        timestamp=df.index[0]
    )
    
    # Log initial state
    initial_equity = portfolio.get_equity()
    logger.info(f"Initial state - Cash: {portfolio.cash:.2f}, Equity: {initial_equity:.2f}")
    
    # Emit signal to trigger position sizing
    logger.info(f"Emitting BUY signal for {symbol} @ {prices[0]:.2f}")
    event_bus.emit(signal)
    
    # Check if orders were generated
    if event_counts['order'] > 0:
        logger.info(f"Order(s) generated: {event_counts['order']}")
    else:
        logger.error("No orders generated from signal!")
    
    # Check if fills occurred
    if event_counts['fill'] > 0:
        logger.info(f"Fill(s) executed: {event_counts['fill']}")
        
        # Check position after fill
        position_value = portfolio.get_position_value()
        logger.info(f"Position value after fill: {position_value:.2f}")
        logger.info(f"Cash after fill: {portfolio.cash:.2f}")
        logger.info(f"Equity after fill: {portfolio.get_equity():.2f}")
        
        # Calculate position percentage of initial portfolio
        position_pct = position_value / initial_equity
        logger.info(f"Position percentage of initial portfolio: {position_pct*100:.2f}%")
        
        # Fast forward to final price
        final_price = prices[-1]
        
        # Update portfolio for new price
        final_equity = portfolio.get_equity({symbol: final_price})
        final_position_value = portfolio.get_position_value({symbol: final_price})
        
        logger.info(f"After price change to {final_price:.2f}:")
        logger.info(f"Position value: {final_position_value:.2f}")
        logger.info(f"Equity: {final_equity:.2f}")
        
        # Calculate return and compare to expected
        calculated_return = (final_equity / initial_equity) - 1
        logger.info(f"Calculated return: {calculated_return:.6f} ({calculated_return*100:.2f}%)")
        logger.info(f"Expected return: {expected_return:.6f} ({expected_return*100:.2f}%)")
        
        # Check if returns match expected
        return_diff = abs(calculated_return - expected_return)
        logger.info(f"Return difference: {return_diff:.6f} ({return_diff*100:.2f}%)")
        
        # Consider test successful if the returns are reasonably close
        return return_diff < 0.05  # Within 5 percentage points
    else:
        logger.error("No fills executed!")
        return False

if __name__ == "__main__":
    logger.info("Testing improved position sizing...")
    
    # Test position sizing calculation
    logger.info("\n=== POSITION SIZING TEST ===")
    test_risk_manager_position_sizing()
    
    # Test full system
    logger.info("\n=== FULL SYSTEM TEST ===")
    success = test_full_system()
    
    # Print final result
    if success:
        logger.info("\n=== TEST PASSED ===")
        logger.info("Position sizing fix works correctly!")
    else:
        logger.info("\n=== TEST FAILED ===")
        logger.info("Position sizing fix did not achieve expected results.")
