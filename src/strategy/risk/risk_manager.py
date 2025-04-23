from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from src.core.events.event_types import SignalEvent, OrderEvent, EventType
from src.core.events.event_utils import create_order_event

import logging
logger = logging.getLogger(__name__)

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


class SimpleRiskManager(RiskManagerBase):
    """
    Simplified risk manager that handles long and short positions.
    Only allows one position at a time (either long or short).
    """
    
    def __init__(self, portfolio, event_bus=None, fixed_size=100):
        """Initialize the simplified risk manager."""
        self.portfolio = portfolio
        self.event_bus = event_bus
        self.fixed_size = fixed_size
        self.position_state = {}  # symbol -> position state (0=neutral, 1=long, -1=short)
        self.orders = []  # For tracking generated orders
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
        return self

    def evaluate_trade(self, symbol, direction, quantity, price):
        """
        Evaluate if a trade complies with risk rules.
        
        Args:
            symbol: Symbol to trade
            direction: Trade direction ('BUY' or 'SELL')
            quantity: Trade quantity
            price: Trade price
            
        Returns:
            bool: True if trade is allowed, False otherwise
        """
        # In this simplified implementation, always allow trades
        # You could add more sophisticated risk checks here
        return True


    
    
    def on_signal(self, event):
        """Process a signal event and produce an order if appropriate."""
        if not isinstance(event, SignalEvent):
            return

        # Extract signal details
        symbol = event.get_symbol()
        signal_value = event.get_signal_value()
        price = event.get_price()
        timestamp = event.get_timestamp()

        # Initialize position state if not exists
        if symbol not in self.position_state:
            self.position_state[symbol] = 0

        # Get current position state (0=neutral, 1=long, -1=short)
        current_state = self.position_state[symbol]

        # Log current state and signal for debugging
        # logger.debug(f"Processing signal for {symbol}: Current position: {current_state}, " 
        #             f"Signal: {'BUY' if signal_value == SignalEvent.BUY else 'SELL'}")

        # CASE 1: BUY signal
        if signal_value == SignalEvent.BUY:
            # Case 1A: Currently no position
            if current_state == 0:
                # Create BUY order for new long position
                order = create_order_event(
                    symbol=symbol,
                    order_type="MARKET",
                    direction="BUY",
                    quantity=self.fixed_size,
                    price=price,
                    timestamp=timestamp
                )
                # Only update state AFTER order is successfully emitted
                if self._emit_order(order):
                    self.position_state[symbol] = 1  # Mark as long
                    # logger.info(f"Opening LONG position for {symbol}: {self.fixed_size} @ {price:.2f}")

            # Case 1B: Currently short - close short position then go long
            elif current_state == -1:
                # Close short position (buy to cover)
                cover_order = create_order_event(
                    symbol=symbol,
                    order_type="MARKET",
                    direction="BUY",  # Buy to cover short
                    quantity=self.fixed_size,
                    price=price,
                    timestamp=timestamp
                )
                # Only update state AFTER order is successfully emitted
                cover_success = self._emit_order(cover_order)
                if cover_success:
                    # logger.info(f"Closing SHORT position for {symbol}: {self.fixed_size} @ {price:.2f}")

                    # Then go long with another order
                    long_order = create_order_event(
                        symbol=symbol,
                        order_type="MARKET",
                        direction="BUY",
                        quantity=self.fixed_size,
                        price=price,
                        timestamp=timestamp
                    )
                    if self._emit_order(long_order):
                        self.position_state[symbol] = 1  # Mark as long
                        # logger.info(f"Opening LONG position for {symbol}: {self.fixed_size} @ {price:.2f}")
                    else:
                        self.position_state[symbol] = 0  # Mark as neutral if second order fails

            # Case 1C: Already long - do nothing
            else:  current_state == 1
                # logger.info(f"Ignoring BUY signal for {symbol}: already in LONG position")

        # CASE 2: SELL signal
        elif signal_value == SignalEvent.SELL:
            # Case 2A: Currently long - close long then go short
            if current_state == 1:
                # Create SELL order to close long
                close_order = create_order_event(
                    symbol=symbol,
                    order_type="MARKET",
                    direction="SELL",
                    quantity=self.fixed_size,
                    price=price,
                    timestamp=timestamp
                )
                # Only update state AFTER order is successfully emitted
                close_success = self._emit_order(close_order)
                if close_success:
                    # logger.info(f"Closing LONG position for {symbol}: {self.fixed_size} @ {price:.2f}")

                    # Then go short with another order
                    short_order = create_order_event(
                        symbol=symbol,
                        order_type="MARKET",
                        direction="SELL",
                        quantity=self.fixed_size,
                        price=price,
                        timestamp=timestamp
                    )
                    if self._emit_order(short_order):
                        self.position_state[symbol] = -1  # Mark as short
                        # logger.info(f"Opening SHORT position for {symbol}: {self.fixed_size} @ {price:.2f}")
                    else:
                        self.position_state[symbol] = 0  # Mark as neutral if second order fails

            # Case 2B: Currently no position - open short position
            elif current_state == 0:
                # Create SELL order to go short
                order = create_order_event(
                    symbol=symbol,
                    order_type="MARKET",
                    direction="SELL",
                    quantity=self.fixed_size,
                    price=price,
                    timestamp=timestamp
                )
                self.position_state[symbol] = -1  # Mark as short if order succeeds
                # if self._emit_order(order):
                    # logger.info(f"Opening SHORT position for {symbol}: {self.fixed_size} @ {price:.2f}")

            # Case 2C: Already short - do nothing
            # else: current_state == -1
                # logger.info(f"Ignoring SELL signal for {symbol}: already in SHORT position")

        # After signal processing, double-check position state for consistency
        updated_state = self.position_state[symbol]
        # if updated_state != current_state:
        #     logger.info(f"Position state changed for {symbol}: {current_state} -> {updated_state}")
    

    def _emit_order(self, order):
        """
        Emit order event and track it.
        Returns success status.
        """
        if not order:
            return False

        # Add to order list
        self.orders.append(order)

        # Initialize success flag
        success = False

        # Try direct broker placement first if available
        if hasattr(self, 'broker') and self.broker:
            try:
                self.broker.place_order(order)
                logging.info(f"Order directly placed with broker: {order.get_symbol()} {order.get_direction()} {order.get_quantity()} @ {order.get_price():.2f}")
                success = True
            except Exception as e:
                logging.error(f"Failed to place order with broker: {e}")
                # Fall back to event bus

        # If broker placement failed or not available, use event bus
        if not success and self.event_bus:
            try:
                self.event_bus.emit(order)
                logging.info(f"Order emitted via event bus: {order.get_symbol()} {order.get_direction()} {order.get_quantity()} @ {order.get_price():.2f}")
                success = True
            except Exception as e:
                logging.error(f"Failed to emit order via event bus: {e}")

        return success
        

        
    def reset(self):
        """Reset the risk manager state."""
        self.position_state = {}
        self.orders = []
    
