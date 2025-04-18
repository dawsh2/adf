#!/usr/bin/env python
"""
Asynchronous Event System Demo

This script demonstrates the usage of the asynchronous event system components.
"""

import sys
import os
import logging
import datetime
import time
import asyncio
import random
from typing import Dict, Any, List

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.events.event_types import EventType, Event, BarEvent, SignalEvent, WebSocketEvent
from core.events.event_bus import EventBus
from core.events.event_manager import EventManager
from core.events.event_utils import create_bar_event, create_signal_event
from core.events.event_handlers import (
    AsyncLoggingHandler, AsyncFilterHandler, AsyncChainHandler, WebSocketHandler
)
from core.events.event_emitters import WebSocketEmitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AsyncStrategy:
    """Simple async moving average crossover strategy for demonstration."""
    
    def __init__(self, symbol, fast_window=5, slow_window=15):
        self.symbol = symbol
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.prices = []
        self.event_bus = None
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
    
    async def on_bar(self, event):
        """Handle bar events asynchronously."""
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
            # Simulate some async work
            await asyncio.sleep(0.01)
            
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
                        await asyncio.create_task(self.event_bus.emit_async(signal))
                
                # Sell signal: fast MA crosses below slow MA
                elif fast_ma < slow_ma and prev_fast_ma >= prev_slow_ma:
                    logger.info(f"SELL SIGNAL: Fast MA ({fast_ma:.2f}) crossed below Slow MA ({slow_ma:.2f})")
                    signal = create_signal_event(
                        SignalEvent.SELL, price, self.symbol, 'ma_crossover',
                        metadata={'fast_ma': fast_ma, 'slow_ma': slow_ma}
                    )
                    if self.event_bus:
                        await asyncio.create_task(self.event_bus.emit_async(signal))
    
    async def reset(self):
        """Reset the strategy state asynchronously."""
        self.prices = []
        await asyncio.sleep(0.01)  # Simulate async work


class AsyncExecutor:
    """Simple async trade executor for demonstration."""
    
    def __init__(self):
        self.positions = {}  # symbol -> position
        self.trades = []
        self.event_bus = None
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
    
    async def on_signal(self, event):
        """Handle signal events asynchronously."""
        symbol = event.get_symbol()
        price = event.get_price()
        signal = event.get_signal_value()
        
        logger.info(f"Received signal: {signal} for {symbol} at {price}")
        
        # Simulate some async work
        await asyncio.sleep(0.02)
        
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
    
    async def reset(self):
        """Reset the executor state asynchronously."""
        self.positions = {}
        self.trades = []
        await asyncio.sleep(0.01)  # Simulate async work


async def generate_random_bar_events_async(symbol='AAPL', base_price=150.0, volatility=1.0):
    """Generate random bar events for testing."""
    
    # Random walk price
    last_close = getattr(generate_random_bar_events_async, 'last_close', base_price)
    change = random.gauss(0, volatility)
    close = max(last_close + change, 0.01)  # Ensure price is positive
    
    # Generate high, low, open
    high = max(close, last_close) + random.uniform(0, volatility)
    low = min(close, last_close) - random.uniform(0, volatility)
    open_price = last_close
    
    # Save last close for next time
    generate_random_bar_events_async.last_close = close
    
    # Create bar event
    timestamp = datetime.datetime.now()
    volume = random.randint(1000, 10000)
    
    return create_bar_event(
        symbol, timestamp, open_price, high, low, close, volume
    )


class MockWebSocketEmitter(WebSocketEmitter):
    """Mock WebSocket emitter for testing."""
    
    def __init__(self, name, event_bus=None):
        super().__init__(name, "wss://example.com/ws", event_bus=event_bus)
        self.connected = False
    
    async def start(self):
        """Start the mock WebSocket emitter."""
        await super().start()
        
        # Create a task to simulate WebSocket messages
        self.create_task(self._simulate_messages())
    
    async def _simulate_messages(self):
        """Simulate WebSocket messages."""
        # Emit a connection event
        await self.emit_async(WebSocketEvent(
            connection_id=self.name,
            state=WebSocketEvent.CONNECTED
        ))
        
        self.connected = True
        
        # Simulate messages
        while self.running:
            if self.connected:
                # Generate a random bar event
                bar = await generate_random_bar_events_async('AAPL', 150.0, 1.0)
                
                # Create a WebSocket message event
                message_event = WebSocketEvent(
                    connection_id=self.name,
                    state=WebSocketEvent.MESSAGE,
                    data={
                        'type': 'bar',
                        'symbol': bar.get_symbol(),
                        'open': bar.get_open(),
                        'high': bar.get_high(),
                        'low': bar.get_low(),
                        'close': bar.get_close(),
                        'volume': bar.get_volume(),
                        'timestamp': bar.get_timestamp().isoformat()
                    }
                )
                
                # Emit the message event
                await self.emit_async(message_event)
                
                # Emit the bar event directly for demonstration
                await self.emit_async(bar)
                
            # Wait for the next message
            await asyncio.sleep(0.5)
    
    async def stop(self):
        """Stop the mock WebSocket emitter."""
        self.connected = False
        
        # Emit a disconnection event
        await self.emit_async(WebSocketEvent(
            connection_id=self.name,
            state=WebSocketEvent.DISCONNECTED
        ))
        
        await super().stop()


class WebSocketBarHandler(WebSocketHandler):
    """Handler for WebSocket bar events."""
    
    def __init__(self, name, event_bus=None):
        super().__init__(name)
        self.event_bus = event_bus
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
    
    async def _process_message(self, connection_id, data):
        """Process a WebSocket message."""
        try:
            # Check if it's a bar message
            if isinstance(data, dict) and data.get('type') == 'bar':
                # Create a bar event
                bar = create_bar_event(
                    symbol=data.get('symbol'),
                    timestamp=datetime.datetime.fromisoformat(data.get('timestamp')),
                    open_price=data.get('open'),
                    high_price=data.get('high'),
                    low_price=data.get('low'),
                    close_price=data.get('close'),
                    volume=data.get('volume')
                )
                
                # Emit the bar event
                if self.event_bus:
                    await self.event_bus.emit_async(bar)
                
                return True
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}", exc_info=True)
            
        return False


async def demonstrate_websocket():
    """Demonstrate WebSocket functionality."""
    # Create event system components
    event_bus = EventBus(use_weak_refs=True)
    event_manager = EventManager(event_bus)
    
    # Create WebSocket handler
    ws_handler = WebSocketBarHandler("ws_handler", event_bus)
    
    # Create logging handler
    logging_handler = AsyncLoggingHandler("ws_logger")
    
    # Register handlers with event bus
    event_bus.register_async(EventType.WEBSOCKET, ws_handler.handle)
    event_bus.register_async(EventType.BAR, logging_handler.handle)
    
    # Create WebSocket emitter
    emitter = MockWebSocketEmitter("mock_ws", event_bus)
    
    try:
        logger.info("Starting WebSocket demo...")
        
        # Start emitter
        await emitter.start()
        
        # Run for a while
        await asyncio.sleep(10)  # Run for 10 seconds
        
    finally:
        # Stop emitter
        await emitter.stop()
        
        # Display statistics
        logger.info("\n=== WEBSOCKET STATISTICS ===")
        logger.info(f"Total messages emitted: {emitter.stats['emitted']}")
        logger.info(f"Total events processed by logger: {logging_handler.stats['processed']}")
        logger.info(f"WebSocket errors: {emitter.stats['errors']}")


async def demonstrate_async_trading():
    """Demonstrate async trading functionality."""
    # Create event system components
    event_bus = EventBus(use_weak_refs=True)
    event_manager = EventManager(event_bus)
    
    # Create strategy and executor
    strategy = AsyncStrategy('AAPL', fast_window=3, slow_window=7)
    executor = AsyncExecutor()
    
    # Create logging handler
    logging_handler = AsyncLoggingHandler("trading_logger")
    
    # Register components with event manager
    event_manager.register_component('strategy', strategy, [EventType.BAR])
    event_manager.register_component('executor', executor, [EventType.SIGNAL])
    
    # Register logging handler with event bus
    event_bus.register_async(EventType.BAR, logging_handler.handle)
    event_bus.register_async(EventType.SIGNAL, logging_handler.handle)
    
    # Run simulated trading
    try:
        logger.info("Starting async trading demo...")
        
        # Generate and emit random bars
        for _ in range(30):  # Generate 30 bars
            bar = await generate_random_bar_events_async('AAPL', 150.0, 1.0)
            await event_bus.emit_async(bar)
            await asyncio.sleep(0.2)
        
    finally:
        # Display statistics
        logger.info("\n=== TRADING STATISTICS ===")
        logger.info(f"Total events processed by logger: {logging_handler.stats['processed']}")
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


async def main():
    """Main async function."""
    try:
        logger.info("=== ASYNCHRONOUS EVENT SYSTEM DEMO ===")
        
        # Demonstrate async trading
        logger.info("\n=== ASYNC TRADING DEMONSTRATION ===")
        await demonstrate_async_trading()
        
        # Demonstrate WebSocket functionality
        logger.info("\n=== WEBSOCKET DEMONSTRATION ===")
        await demonstrate_websocket()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
