#!/usr/bin/env python
"""
Event System Demo

This script demonstrates the usage of the synchronous event system components.
"""

import sys
import os
import logging
import datetime
import time
import random
from typing import Dict, Any, List

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.events.event_types import EventType, Event, BarEvent, SignalEvent
from core.events.event_bus import EventBus
from core.events.event_manager import EventManager
from core.events.event_utils import create_bar_event, create_signal_event
# Import the correct class names
from core.events.event_handlers import LoggingHandler, FilterHandler, ChainHandler
from core.events.event_emitters import EventGeneratorEmitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleStrategy:
    """Simple moving average crossover strategy for demonstration."""
    
    def __init__(self, symbol, fast_window=5, slow_window=15):
        self.symbol = symbol
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.prices = []
        self.event_bus = None
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
    
    def on_bar(self, event):
        """Handle bar events."""
        if event.get_symbol() != self.symbol:
            return
            
        # Add price to history
        price = event.get_close()
        self.prices.append(price)
        
        # Keep only necessary history
        if len(self.prices) > self.slow_window + 10:
            self.prices = self.prices[-(self.slow_window + 10):]
            
        # Calculate moving averages
        if len(self.prices) >= self.slow_window:
            fast_ma = sum(self.prices[-self.fast_window:]) / self.fast_window
            slow_ma = sum(self.prices[-self.slow_window:]) / self.slow_window
            
            # Generate signal on crossover
            if len(self.prices) > self.slow_window:
                prev_fast_ma = sum(self.prices[-(self.fast_window+1):-1]) / self.fast_window
                prev_slow_ma = sum(self.prices[-(self.slow_window+1):-1]) / self.slow_window
                
                # Buy signal: fast MA crosses above slow MA
                if fast_ma > slow_ma and prev_fast_ma <= prev_slow_ma:
                    logger.info(f"BUY SIGNAL: Fast MA ({fast_ma:.2f}) crossed above Slow MA ({slow_ma:.2f})")
                    signal = create_signal_event(
                        SignalEvent.BUY, price, self.symbol, 'ma_crossover',
                        metadata={'fast_ma': fast_ma, 'slow_ma': slow_ma}
                    )
                    if self.event_bus:
                        self.event_bus.emit(signal)
                
                # Sell signal: fast MA crosses below slow MA
                elif fast_ma < slow_ma and prev_fast_ma >= prev_slow_ma:
                    logger.info(f"SELL SIGNAL: Fast MA ({fast_ma:.2f}) crossed below Slow MA ({slow_ma:.2f})")
                    signal = create_signal_event(
                        SignalEvent.SELL, price, self.symbol, 'ma_crossover',
                        metadata={'fast_ma': fast_ma, 'slow_ma': slow_ma}
                    )
                    if self.event_bus:
                        self.event_bus.emit(signal)
    
    def reset(self):
        """Reset the strategy state."""
        self.prices = []


class SimpleExecutor:
    """Simple trade executor for demonstration."""
    
    def __init__(self):
        self.positions = {}  # symbol -> position
        self.trades = []
        self.event_bus = None
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
    
    def on_signal(self, event):
        """Handle signal events."""
        symbol = event.get_symbol()
        price = event.get_price()
        signal = event.get_signal_value()
        
        logger.info(f"Received signal: {signal} for {symbol} at {price}")
        
        if signal == SignalEvent.BUY:
            # Open long position
            self.positions[symbol] = {
                'direction': 'LONG',
                'entry_price': price,
                'quantity': 100,
                'entry_time': event.get_timestamp()
            }
            logger.info(f"Opened LONG position for {symbol} at {price}")
            
            # Log trade
            self.trades.append({
                'symbol': symbol,
                'direction': 'BUY',
                'quantity': 100,
                'price': price,
                'timestamp': event.get_timestamp()
            })
            
        elif signal == SignalEvent.SELL:
            # Close any open long position
            if symbol in self.positions and self.positions[symbol]['direction'] == 'LONG':
                entry_price = self.positions[symbol]['entry_price']
                quantity = self.positions[symbol]['quantity']
                profit = (price - entry_price) * quantity
                
                logger.info(f"Closed LONG position for {symbol} at {price}, Profit: {profit:.2f}")
                
                # Log trade
                self.trades.append({
                    'symbol': symbol,
                    'direction': 'SELL',
                    'quantity': quantity,
                    'price': price,
                    'timestamp': event.get_timestamp(),
                    'profit': profit
                })
                
                # Open short position
                self.positions[symbol] = {
                    'direction': 'SHORT',
                    'entry_price': price,
                    'quantity': 100,
                    'entry_time': event.get_timestamp()
                }
                logger.info(f"Opened SHORT position for {symbol} at {price}")
                
                # Log trade
                self.trades.append({
                    'symbol': symbol,
                    'direction': 'SELL_SHORT',
                    'quantity': 100,
                    'price': price,
                    'timestamp': event.get_timestamp()
                })
    
    def reset(self):
        """Reset the executor state."""
        self.positions = {}
        self.trades = []


def generate_random_bar_events(symbol='AAPL', base_price=150.0, volatility=1.0):
    """Generate random bar events for testing."""
    import random
    
    # Random walk price
    last_close = getattr(generate_random_bar_events, 'last_close', base_price)
    change = random.gauss(0, volatility)
    close = max(last_close + change, 0.01)  # Ensure price is positive
    
    # Generate high, low, open
    high = max(close, last_close) + random.uniform(0, volatility)
    low = min(close, last_close) - random.uniform(0, volatility)
    open_price = last_close
    
    # Save last close for next time
    generate_random_bar_events.last_close = close
    
    # Create bar event
    timestamp = datetime.datetime.now()
    volume = random.randint(1000, 10000)
    
    return create_bar_event(
        symbol, timestamp, open_price, high, low, close, volume
    )


def demonstrate_filters():
    """Demonstrate event filtering."""
    event_bus = EventBus()
    
    # Create filter handler that only passes AAPL events
    apple_events = []
    msft_events = []
    
    def handle_apple(event):
        apple_events.append(event)
        return True
        
    def handle_msft(event):
        msft_events.append(event)
        return True
    
    # Create filter functions
    def is_apple(event):
        return hasattr(event, 'get_symbol') and event.get_symbol() == 'AAPL'
        
    def is_msft(event):
        return hasattr(event, 'get_symbol') and event.get_symbol() == 'MSFT'
    
    # Create filter handlers - use functions directly, not objects with handle methods
    apple_filter = FilterHandler("apple_filter", is_apple, handle_apple)
    msft_filter = FilterHandler("msft_filter", is_msft, handle_msft)
    
    # Create chain handler for both filters
    chain_handler = ChainHandler("symbol_filters", [apple_filter, msft_filter])
    
    # Register chain handler with event bus
    event_bus.register(EventType.BAR, chain_handler.handle)
    
    # Create and emit events
    aapl_event = create_bar_event('AAPL', datetime.datetime.now(), 150.0, 152.0, 149.0, 151.0, 10000)
    msft_event = create_bar_event('MSFT', datetime.datetime.now(), 250.0, 252.0, 249.0, 251.0, 8000)
    goog_event = create_bar_event('GOOG', datetime.datetime.now(), 2500.0, 2520.0, 2490.0, 2510.0, 5000)
    
    event_bus.emit(aapl_event)
    event_bus.emit(msft_event)
    event_bus.emit(goog_event)
    
    # Check results
    logger.info("=== FILTER DEMONSTRATION ===")
    logger.info(f"Apple events: {len(apple_events)}, MSFT events: {len(msft_events)}")
    logger.info(f"Apple filter processed: {apple_filter.stats['processed']}")
    logger.info(f"MSFT filter processed: {msft_filter.stats['processed']}")
    logger.info(f"Chain handler processed: {chain_handler.stats['processed']}")

def demonstrate_weakref():
    """Demonstrate weakref functionality in the event bus."""
    # Create event system components
    event_bus = EventBus(use_weak_refs=True)  # Use weak references
    
    logger.info("=== WEAKREF DEMONSTRATION ===")
    
    class TemporaryComponent:
        def __init__(self, name):
            self.name = name
            self.event_count = 0
        
        def handle_event(self, event):
            self.event_count += 1
            logger.info(f"Component {self.name} received event: {event.get_type().name}")
    
    # Create components
    comp1 = TemporaryComponent("Component1")
    comp2 = TemporaryComponent("Component2")
    
    # Register event handlers
    event_bus.register(EventType.BAR, comp1.handle_event)
    event_bus.register(EventType.BAR, comp2.handle_event)
    
    # Check active handlers
    stats = event_bus.get_stats()
    logger.info(f"Active handlers before GC: {stats['active_handlers']}")
    
    # Send an event to both components
    event1 = Event(EventType.BAR, {'test': 'data1'})
    event_bus.emit(event1)
    
    # Both should have received it
    logger.info(f"Component1 event count: {comp1.event_count}")
    logger.info(f"Component2 event count: {comp2.event_count}")
    
    # Delete one component and force garbage collection
    logger.info("Deleting Component1 and forcing garbage collection...")
    import gc
    del comp1
    gc.collect()
    
    # Force cleanup
    event_bus.cleanup()
    
    # Check active handlers
    stats = event_bus.get_stats()
    logger.info(f"Active handlers after GC: {stats['active_handlers']}")
    
    # Send another event
    event2 = Event(EventType.BAR, {'test': 'data2'})
    event_bus.emit(event2)
    
    # Only comp2 should receive it
    logger.info(f"Component2 event count: {comp2.event_count}")


def main():
    """Main function to demonstrate event system."""
    # Create event system components
    event_bus = EventBus(use_weak_refs=True)  # Use weak references by default
    event_manager = EventManager(event_bus)
    
    # Create strategy and executor
    strategy = SimpleStrategy('AAPL', fast_window=3, slow_window=7)
    executor = SimpleExecutor()
    
    # Create logging handler
    logging_handler = LoggingHandler("main_logger")
    
    # Register components with event manager
    event_manager.register_component('strategy', strategy, [EventType.BAR])
    event_manager.register_component('executor', executor, [EventType.SIGNAL])
    
    # Register logging handler for all events
    event_bus.register(EventType.BAR, logging_handler.handle)
    event_bus.register(EventType.SIGNAL, logging_handler.handle)
    
    # Create event emitter for random bar data
    emitter = EventGeneratorEmitter(
        "random_bars", 
        lambda: generate_random_bar_events('AAPL', 150.0, 1.0),
        event_bus,
        interval=0.5  # Generate a bar every 0.5 seconds
    )
    
    try:
        logger.info("Starting event system demo...")
        
        # Start emitter
        emitter.start()
        
        # Run for a while
        time.sleep(15)  # Run for 15 seconds
        
        # Demonstrate filter handlers
        demonstrate_filters()
        
        # Run for a bit more
        time.sleep(5)  # Run for 5 more seconds
        
        # Demonstrate weakref functionality
        demonstrate_weakref()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Stop emitter
        emitter.stop()
        
        # Display statistics
        logger.info("\n=== FINAL STATISTICS ===")
        logger.info(f"Total bars emitted: {emitter.stats['emitted']}")
        logger.info(f"Total events processed by logger: {logging_handler.stats['processed']}")
        logger.info(f"Event counts by type: {event_bus.event_counts}")
        logger.info(f"Active handlers: {event_bus.get_stats()['active_handlers']}")
        logger.info(f"Total trades: {len(executor.trades)}")
        
        # Display trade summary
        if executor.trades:
            total_profit = sum(trade.get('profit', 0) for trade in executor.trades)
            logger.info(f"Total profit: {total_profit:.2f}")
            
            # Display last few trades
            logger.info("Last 5 trades:")
            for trade in executor.trades[-5:]:
                direction = trade['direction']
                symbol = trade['symbol']
                price = trade['price']
                profit = trade.get('profit', 'N/A')
                logger.info(f"  {direction} {symbol} @ {price:.2f} - Profit: {profit}")


if __name__ == "__main__":
    main()
