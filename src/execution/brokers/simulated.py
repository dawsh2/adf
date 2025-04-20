import datetime
import random
from typing import Dict, Any, Optional

from src.core.events.event_types import OrderEvent, FillEvent
from src.core.events.event_utils import create_fill_event

class SimulatedBroker(BrokerBase):
    """
    Simulated broker for backtesting.
    Executes orders with configurable slippage and delay.
    """
    
    def __init__(self, slippage_model=None, delay_model=None, fill_emitter=None):
        super().__init__(fill_emitter)
        self.slippage_model = slippage_model or DefaultSlippageModel()
        self.delay_model = delay_model or DefaultDelayModel()
        self.orders = {}  # order_id -> order
        self.market_data = {}  # symbol -> price data
    
    def place_order(self, order):
        """Simulate order execution."""
        if not isinstance(order, OrderEvent):
            return
        
        # Store order
        order_id = order.get_id()
        self.orders[order_id] = order
        
        # Get execution details
        symbol = order.get_symbol()
        direction = order.get_direction()
        quantity = order.get_quantity()
        requested_price = order.get_price()
        
        # Get current market price
        market_price = self.get_market_price(symbol)
        if market_price is None:
            # Can't execute without price
            return
        
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
        
        # Emit fill event
        self.emit_fill(fill)
    
    def cancel_order(self, order_id):
        """Cancel a simulated order."""
        if order_id in self.orders:
            del self.orders[order_id]
            return True
        return False
    
    def get_market_price(self, symbol):
        """Get the current market price for a symbol."""
        # In a real implementation, this would get the price from market data
        # For simplicity, just return the last known price or a default
        return self.market_data.get(symbol, {}).get('price', 100.0)
    
    def update_market_data(self, symbol, data):
        """Update market data for a symbol."""
        self.market_data[symbol] = data
    
    def _calculate_commission(self, price, quantity):
        """Calculate commission for a trade."""
        # Simple commission model
        return max(1.0, 0.005 * price * quantity)


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
