"""
Fixed simulated broker with proper fill emission handling.
"""
import datetime
import random
import logging
from typing import Dict, Any, Optional

from src.core.events.event_types import OrderEvent, FillEvent
from src.core.events.event_utils import create_fill_event

logger = logging.getLogger(__name__)

class SimulatedBroker:
    """
    Simulated broker for backtesting.
    Executes orders with configurable slippage and delay.
    """
    
    def __init__(self, slippage_model=None, delay_model=None, fill_emitter=None):
        """
        Initialize the simulated broker.
        
        Args:
            slippage_model: Model for price slippage (optional)
            delay_model: Model for execution delay (optional)
            fill_emitter: Event emitter for fill events
        """
        self.slippage_model = slippage_model or DefaultSlippageModel()
        self.delay_model = delay_model or DefaultDelayModel()
        self.fill_emitter = fill_emitter
        self.event_bus = None  # Set by event manager
        self.orders = {}  # order_id -> order
        self.market_data = {}  # symbol -> price data
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
        return self
    
    def place_order(self, order):
        """
        Simulate order execution.
        
        Args:
            order: Order to execute
        """
        if not isinstance(order, OrderEvent):
            logger.warning(f"Ignoring non-OrderEvent: {type(order)}")
            return
        
        # Store order
        order_id = order.get_id()
        self.orders[order_id] = order
        
        # Get execution details
        symbol = order.get_symbol()
        direction = order.get_direction()
        quantity = order.get_quantity()
        requested_price = order.get_price()
        
        # Log for debugging
        logger.debug(f"Processing order: {symbol} {direction} {quantity} @ {requested_price}")
        
        # Get current market price
        market_price = self.get_market_price(symbol)
        if market_price is None:
            logger.warning(f"No market price available for {symbol}, using requested price")
            market_price = requested_price or 100.0  # Default price if none available
        
        # Calculate execution price with slippage
        execution_price = self.slippage_model.apply_slippage(
            market_price, direction, quantity, symbol
        )
        
        # Create fill event
        fill = create_fill_event(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            price=execution_price,
            commission=self._calculate_commission(execution_price, quantity),
            timestamp=order.get_timestamp() or datetime.datetime.now()
        )
        
        # Log fill creation
        logger.debug(f"Created fill: {symbol} {direction} {quantity} @ {execution_price}")
        
        # Emit fill event
        self._emit_fill(fill)
    
    def cancel_order(self, order_id):
        """Cancel a simulated order."""
        if order_id in self.orders:
            del self.orders[order_id]
            return True
        return False
    
    def get_market_price(self, symbol):
        """Get the current market price for a symbol."""
        if symbol in self.market_data and 'price' in self.market_data[symbol]:
            return self.market_data[symbol]['price']
        # In a real implementation, this would get the price from market data
        # For simplicity, just return a default
        return 100.0
    
    def update_market_data(self, symbol, data):
        """Update market data for a symbol."""
        self.market_data[symbol] = data
    
    def _emit_fill(self, fill):
        """
        Emit a fill event - Fixed implementation to ensure fills are properly emitted.
        
        Args:
            fill: Fill event to emit
        """
        # Log that we're emitting a fill
        logger.debug(f"Emitting fill: {fill.get_symbol()} {fill.get_direction()} {fill.get_quantity()} @ {fill.get_price()}")
        
        try:
            # Try all possible emission methods to ensure it works
            
            # First try the specific fill_emitter if provided
            fill_emitted = False
            if self.fill_emitter:
                try:
                    if hasattr(self.fill_emitter, 'emit'):
                        self.fill_emitter.emit(fill)
                        fill_emitted = True
                        logger.debug("Fill emitted via fill_emitter.emit")
                    else:
                        # Assume fill_emitter is directly callable
                        self.fill_emitter(fill)
                        fill_emitted = True
                        logger.debug("Fill emitted via direct fill_emitter call")
                except Exception as e:
                    logger.error(f"Error emitting fill via fill_emitter: {e}")
            
            # Then try event_bus as backup
            if not fill_emitted and self.event_bus:
                try:
                    self.event_bus.emit(fill)
                    fill_emitted = True
                    logger.debug("Fill emitted via event_bus")
                except Exception as e:
                    logger.error(f"Error emitting fill via event_bus: {e}")
            
            # Log error if no emission method worked
            if not fill_emitted:
                logger.error("Failed to emit fill event - no working emitter found!")
        except Exception as e:
            logger.error(f"Unexpected error in fill emission: {e}")
    
    def _calculate_commission(self, price, quantity):
        """Calculate commission for a trade."""
        # Simple commission model: max($1.0, 0.1% of trade value)
        return max(1.0, 0.001 * price * quantity)


class DefaultSlippageModel:
    """Default slippage model for simulated broker."""
    
    def apply_slippage(self, price, direction, quantity, symbol=None):
        """Apply slippage to a price."""
        # Simple slippage model: 0.1% in the adverse direction
        slippage_factor = 0.001
        
        if direction == 'BUY':
            # For buys, price goes up
            return price * (1 + slippage_factor)
        elif direction == 'SELL':
            # For sells, price goes down
            return price * (1 - slippage_factor)
        
        return price


class DefaultDelayModel:
    """Default delay model for simulated broker."""
    
    def calculate_delay(self, order):
        """Calculate execution delay for an order."""
        # Simple delay model: 0-10ms random delay
        return random.randint(0, 10) / 1000.0
