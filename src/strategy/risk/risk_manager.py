from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from src.core.events.event_types import SignalEvent, OrderEvent, EventType
from src.core.events.event_utils import create_order_event

class RiskManagerBase(ABC):
    """Abstract base class for risk managers."""
    
    @abstractmethod
    def on_signal(self, signal_event):
        """Process a signal event and produce an order if appropriate."""
        pass
    
    @abstractmethod
    def evaluate_trade(self, symbol, direction, quantity, price):
        """Evaluate if a trade complies with risk rules."""
        pass


class RiskManager(RiskManagerBase):
    """
    Evaluates trades against risk rules and calculates position sizes.
    Converts signals to orders after applying risk constraints.
    """
    
    def __init__(self, portfolio, position_sizer, risk_limits=None, event_bus=None):
        self.portfolio = portfolio
        self.position_sizer = position_sizer
        self.risk_limits = risk_limits or {}
        self.event_bus = event_bus
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
        return self
    
    def on_signal(self, signal_event):
        """Process a signal event and produce an order if appropriate."""
        if not isinstance(signal_event, SignalEvent):
            return
        
        # Extract signal details
        symbol = signal_event.get_symbol()
        signal_value = signal_event.get_signal_value()
        price = signal_event.get_price()
        
        # Determine order direction
        if signal_value == SignalEvent.BUY:
            direction = 'BUY'
        elif signal_value == SignalEvent.SELL:
            direction = 'SELL'
        else:
            return  # Ignore neutral signals
        
        # Calculate position size
        quantity = self.position_sizer.calculate_position_size(
            symbol, direction, price, self.portfolio, signal_event
        )
        
        # Apply risk limits
        quantity = self._apply_risk_limits(symbol, direction, quantity, price)
        
        # Create and emit order if quantity is valid
        if quantity > 0:
            order = create_order_event(
                symbol=symbol,
                order_type='MARKET',  # Default to market orders
                direction=direction,
                quantity=quantity,
                price=price,
                timestamp=signal_event.get_timestamp()
            )
            
            # Emit order event
            if self.event_bus:
                self.event_bus.emit(order)
    
    def _apply_risk_limits(self, symbol, direction, quantity, price):
        """Apply risk limits to the calculated quantity."""
        if quantity <= 0:
            return 0
        
        # Get current position
        current_position = self.portfolio.get_position(symbol)
        position_qty = current_position.quantity if current_position else 0
        
        # Apply maximum position size limit
        max_position_size = self.risk_limits.get('max_position_size', float('inf'))
        if direction == 'BUY':
            new_position_size = position_qty + quantity
            if new_position_size > max_position_size:
                quantity = max(0, max_position_size - position_qty)
        
        # Apply maximum portfolio exposure limit
        max_exposure = self.risk_limits.get('max_exposure', 1.0)
        portfolio_value = self.portfolio.get_equity()
        trade_value = quantity * price
        
        if trade_value > portfolio_value * max_exposure:
            quantity = int((portfolio_value * max_exposure) / price)
        
        # Apply minimum trade size
        min_trade_size = self.risk_limits.get('min_trade_size', 1)
        if quantity < min_trade_size:
            return 0
        
        return quantity
    
    def evaluate_trade(self, symbol, direction, quantity, price):
        """Evaluate if a trade complies with risk rules."""
        # Similar to _apply_risk_limits but returns a boolean
        adjusted_quantity = self._apply_risk_limits(symbol, direction, quantity, price)
        return adjusted_quantity >= quantity
