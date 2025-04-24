"""
Final fixed validation test to verify the trading system works correctly.

This script creates a test that validates the complete trading pipeline
and ensures market values are properly updated as prices change.
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
from src.core.events.event_types import EventType, BarEvent, SignalEvent, OrderEvent, FillEvent
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

def run_validation_test():
    """Run a comprehensive validation test of the trading system."""
    logger.info("=== STARTING VALIDATION TEST ===")
    
    # Set up event system
    event_bus = EventBus()
    event_manager = EventManager(event_bus)
    
    # Create portfolio with initial cash
    portfolio = PortfolioManager(initial_cash=10000.0)
    
    # Create bar emitter (needed by some components)
    from src.core.events.event_emitters import BarEmitter
    bar_emitter = BarEmitter("test_emitter", event_bus)
    bar_emitter.start()  # Start the emitter
    
    # Create broker with explicit fill emitter
    broker = SimulatedBroker(fill_emitter=event_bus)
    
    # Create risk manager
    risk_manager = SimpleRiskManager(
        portfolio=portfolio,
        event_bus=event_bus,
        position_pct=0.95  # Use 95% of portfolio for each position
    )
    # IMPORTANT: Set the broker reference in risk manager
    risk_manager.broker = broker
    
    # Register components with event manager - proper event flow connections
    event_manager.register_component('portfolio', portfolio, [EventType.FILL])
    event_manager.register_component('risk_manager', risk_manager, [EventType.SIGNAL])
    
    # Direct registration of order handler to ensure it's properly connected
    event_bus.register(EventType.ORDER, broker.place_order)
    
    # Initialize event counters
    event_counts = {
        'bar': 0,
        'signal': 0, 
        'order': 0,
        'fill': 0
    }
    
    # Event tracking handler
    def track_event(event):
        event_type = event.get_type()
        if event_type == EventType.BAR:
            event_counts['bar'] += 1
            if event_counts['bar'] == 1 or event_counts['bar'] % 5 == 0 or event_counts['bar'] == 30:
                # Log every 5th bar for visibility
                symbol = event.get_symbol() if hasattr(event, 'get_symbol') else "N/A"
                price = event.get_close() if hasattr(event, 'get_close') else 0
                
                # Get portfolio equity and position value to track changes
                market_prices = {symbol: price}
                current_equity = portfolio.get_equity(market_prices)
                position_value = portfolio.get_position_value()
                
                logger.info(f"Bar {event_counts['bar']}/30: Price=${price:.2f}, " +
                           f"Cash: ${portfolio.cash:.2f}, Position Value: ${position_value:.2f}, " +
                           f"Equity: ${current_equity:.2f}")
                
        elif event_type == EventType.SIGNAL:
            event_counts['signal'] += 1
            symbol = event.get_symbol() if hasattr(event, 'get_symbol') else "N/A"
            signal_val = event.data.get('signal_value') if hasattr(event, 'data') else "N/A"
            price = event.data.get('price') if hasattr(event, 'data') else 0
            logger.info(f"Signal: {symbol} {signal_val} @ {price:.2f}")
            
        elif event_type == EventType.ORDER:
            event_counts['order'] += 1
            symbol = event.get_symbol() if hasattr(event, 'get_symbol') else "N/A"
            direction = event.data.get('direction') if hasattr(event, 'data') else "N/A"
            quantity = event.data.get('quantity') if hasattr(event, 'data') else 0
            price = event.data.get('price') if hasattr(event, 'data') else 0
            logger.info(f"Order: {symbol} {direction} {quantity} @ {price:.2f}")
            
        elif event_type == EventType.FILL:
            event_counts['fill'] += 1
            symbol = event.get_symbol() if hasattr(event, 'get_symbol') else "N/A"
            direction = event.get_direction() if hasattr(event, 'get_direction') else "N/A"
            quantity = event.get_quantity() if hasattr(event, 'get_quantity') else 0
            price = event.get_price() if hasattr(event, 'get_price') else 0
            logger.info(f"Fill: {symbol} {direction} {quantity} @ {price:.2f}")
            
            # Log portfolio state after fill
            position_value = portfolio.get_position_value()
            current_equity = portfolio.get_equity()
            logger.info(f"After fill - Cash: ${portfolio.cash:.2f}, Position Value: ${position_value:.2f}, " + 
                      f"Equity: ${current_equity:.2f}")
            
    # Register tracker for all event types
    for event_type in [EventType.BAR, EventType.SIGNAL, EventType.ORDER, EventType.FILL]:
        event_bus.register(event_type, track_event)
    
    # Generate synthetic data
    days = 30
    symbol = "TEST"
    df, expected_return, prices = create_synthetic_price_series(days=days)
    
    # Initial portfolio state
    initial_equity = portfolio.get_equity()
    logger.info(f"Initial state - Cash: {portfolio.cash:.2f}, Equity: {initial_equity:.2f}")
    
    # Generate SIGNAL for buying at the first price
    signal = create_signal_event(
        signal_value=SignalEvent.BUY,  # BUY signal
        price=prices[0],
        symbol=symbol,
        timestamp=df.index[0]
    )
    
    # Emit the signal to trigger order generation
    event_bus.emit(signal)
    logger.info(f"Generated BUY signal for {symbol} @ {prices[0]:.2f}")
    
    # Loop through price bars to simulate time passing
    for i, (date, row) in enumerate(df.iterrows()):
        # Create and emit bar event
        bar = create_bar_event(
            symbol=symbol,
            timestamp=date,
            open_price=row['open'],
            high_price=row['high'],
            low_price=row['low'],
            close_price=row['close'],
            volume=row['volume']
        )
        
        # Update broker's market price data (critical for valuation)
        broker.update_market_data(symbol, {
            'price': row['close'],
            'timestamp': date
        })
        
        # IMPORTANT: Update portfolio's market prices directly
        market_prices = {symbol: row['close']}
        portfolio.update_market_data(market_prices)
        
        # Emit the bar event
        event_bus.emit(bar)
    
    # Log final state
    logger.info(f"Event summary: {event_counts}")
    
    # Calculate final values using the last price
    final_price = prices[-1]
    market_prices = {symbol: final_price}
    
    # Get final portfolio state with explicit market prices
    final_equity = portfolio.get_equity(market_prices)
    position_value = portfolio.get_position_value(market_prices)
    
    logger.info(f"Final state - Cash: ${portfolio.cash:.2f}, " +
               f"Position Value: ${position_value:.2f}, Equity: ${final_equity:.2f}")
    
    # Check if fills occurred
    if event_counts['fill'] > 0:
        # Calculate actual return
        actual_return = (final_equity / initial_equity) - 1
        logger.info(f"Actual return: {actual_return:.6f} ({actual_return*100:.2f}%)")
        logger.info(f"Expected return: {expected_return:.6f} ({expected_return*100:.2f}%)")
        
        # Check if returns are reasonably close
        return_diff = abs(actual_return - expected_return)
        logger.info(f"Return difference: {return_diff:.6f} ({return_diff*100:.2f}%)")
        
        # Get position details
        positions = portfolio.get_position_details(market_prices)
        logger.info(f"Final portfolio positions: {positions}")
        
        # Test passes if fills occurred and return is close to expected
        test_passed = return_diff < 0.05  # Within 5 percentage points
        
        if test_passed:
            logger.info("VALIDATION TEST PASSED!")
        else:
            logger.error("VALIDATION TEST FAILED: Returns do not match expected values")
        
        return test_passed
    else:
        # Get position details for debugging
        positions = portfolio.get_position_details()
        logger.info(f"Final portfolio positions: {positions}")
        
        logger.error(f"Test failed: Signal count={event_counts['signal']}, Fill count={event_counts['fill']}")
        logger.error("VALIDATION TEST FAILED!")
        return False

if __name__ == "__main__":
    success = run_validation_test()
    
    if not success:
        print("\nVALIDATION TEST FAILED!")
        print("There are still issues in the trading system.")
    else:
        print("\nVALIDATION TEST PASSED!")
        print("The trading system is working correctly.")
