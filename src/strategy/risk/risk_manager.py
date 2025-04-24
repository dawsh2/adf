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

class SimpleRiskManager:
    """
    Simplified risk manager that handles long and short positions.
    Only allows one position at a time (either long or short).
    """
    
    def __init__(self, portfolio, event_bus=None, position_pct=0.95, max_risk_pct=2.0):
        """
        Initialize the simplified risk manager.
        
        Args:
            portfolio: Portfolio manager
            event_bus: Event bus
            position_pct: Percentage of portfolio to invest (0.95 = 95%)
            max_risk_pct: Maximum risk percentage per trade (2% default)
        """
        self.portfolio = portfolio
        self.event_bus = event_bus
        self.position_pct = position_pct  # NEW: Set position as a percentage of portfolio
        self.max_risk_pct = max_risk_pct
        self.position_state = {}  # symbol -> position state (0=neutral, 1=long, -1=short)
        self.broker = None  # Direct broker connection
        self.orders = []  # For tracking generated orders
        
        # Debug flag for validation
        self.debug = False
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
        return self
    
    def set_debug(self, debug=True):
        """Enable or disable debug mode."""
        self.debug = debug
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
        # Calculate risk metrics
        portfolio_value = self.portfolio.get_equity()
        trade_value = quantity * price
        trade_risk_pct = (trade_value / portfolio_value) * 100
        
        # Check against maximum risk
        if trade_risk_pct > self.max_risk_pct:
            if self.debug:
                logger.warning(f"Trade rejected: Risk too high ({trade_risk_pct:.2f}% > {self.max_risk_pct}%)")
            return False
            
        # Check for sufficient cash for buys
        if direction == 'BUY' and trade_value > self.portfolio.cash:
            if self.debug:
                logger.warning(f"Trade rejected: Insufficient cash ({self.portfolio.cash:.2f} < {trade_value:.2f})")
            return False
            
        # In this simplified implementation, allow the trade
        return True

    def calculate_position_size(self, symbol, price, signal_strength=1.0):
        """Calculate position size - simplified to always return 1."""
        return 1  # Always return 1 share regardless of portfolio size


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
        if self.debug:
            logger.debug(f"Processing signal for {symbol}: Current position: {current_state}, " +
                       f"Signal: {'BUY' if signal_value == SignalEvent.BUY else 'SELL'}")

        # Calculate position size based on risk parameters
        # Get confidence value from signal if available (default to 1.0)
        confidence = event.data.get('confidence', 1.0) if hasattr(event, 'data') else 1.0
        position_size = self.calculate_position_size(symbol, price, signal_strength=confidence)

        # CASE 1: BUY signal
        if signal_value == SignalEvent.BUY:
            # Only act if not already long
            if current_state != 1:
                if current_state == -1:
                    # Currently short - close position and go long in one step
                    # First, close short position
                    close_order = create_order_event(
                        symbol=symbol,
                        order_type="MARKET",
                        direction="BUY",  # Buy to cover short
                        quantity=position_size,
                        price=price,
                        timestamp=timestamp
                    )
                    close_success = self._emit_order(close_order)

                    if close_success and self.debug:
                        logger.debug(f"Closing SHORT position for {symbol}: {position_size} @ {price:.2f}")

                    # Then go long 
                    long_order = create_order_event(
                        symbol=symbol,
                        order_type="MARKET",
                        direction="BUY",
                        quantity=position_size,
                        price=price,
                        timestamp=timestamp
                    )
                    if self._emit_order(long_order):
                        self.position_state[symbol] = 1  # Mark as long
                        if self.debug:
                            logger.debug(f"Opening LONG position for {symbol}: {position_size} @ {price:.2f}")
                else:
                    # Currently neutral - just go long
                    order = create_order_event(
                        symbol=symbol,
                        order_type="MARKET",
                        direction="BUY",
                        quantity=position_size,
                        price=price,
                        timestamp=timestamp
                    )
                    order_success = self._emit_order(order)
                    if order_success:
                        self.position_state[symbol] = 1  # Mark as long
                        if self.debug:
                            logger.debug(f"Opening LONG position for {symbol}: {position_size} @ {price:.2f}")
            else:
                # Already long - do nothing
                if self.debug:
                    logger.debug(f"Ignoring BUY signal for {symbol}: already in LONG position")

        # CASE 2: SELL signal
        elif signal_value == SignalEvent.SELL:
            # Only act if not already short
            if current_state != -1:
                if current_state == 1:
                    # Currently long - close position and go short in one step
                    # First, close long position
                    close_order = create_order_event(
                        symbol=symbol,
                        order_type="MARKET",
                        direction="SELL",
                        quantity=position_size,
                        price=price,
                        timestamp=timestamp
                    )
                    close_success = self._emit_order(close_order)

                    if close_success and self.debug:
                        logger.debug(f"Closing LONG position for {symbol}: {position_size} @ {price:.2f}")

                    # Then go short
                    short_order = create_order_event(
                        symbol=symbol,
                        order_type="MARKET",
                        direction="SELL",
                        quantity=position_size,
                        price=price,
                        timestamp=timestamp
                    )
                    if self._emit_order(short_order):
                        self.position_state[symbol] = -1  # Mark as short
                        if self.debug:
                            logger.debug(f"Opening SHORT position for {symbol}: {position_size} @ {price:.2f}")
                else:
                    # Currently neutral - just go short
                    order = create_order_event(
                        symbol=symbol,
                        order_type="MARKET",
                        direction="SELL",
                        quantity=position_size,
                        price=price,
                        timestamp=timestamp
                    )
                    order_success = self._emit_order(order)
                    if order_success:
                        self.position_state[symbol] = -1  # Mark as short
                        if self.debug:
                            logger.debug(f"Opening SHORT position for {symbol}: {position_size} @ {price:.2f}")
            else:
                # Already short - do nothing
                if self.debug:
                    logger.debug(f"Ignoring SELL signal for {symbol}: already in SHORT position")
    


    def _emit_order(self, order):
        """
        Emit order event and track it.
        Returns success status.
        """
        if not order:
            return False

        # Add to order list for tracking
        self.orders.append(order)

        # Track success
        success = False

        try:
            # Try event bus first
            if self.event_bus:
                if self.debug:
                    logger.debug(f"Emitting order via event bus: {order.get_symbol()} {order.get_direction()}")
                self.event_bus.emit(order)
                success = True
            # Direct broker placement as backup
            elif self.broker:
                if self.debug:
                    logger.debug(f"Placing order with broker directly: {order.get_symbol()} {order.get_direction()}")
                self.broker.place_order(order)
                success = True
            else:
                logger.warning("No event bus or broker available - order not emitted!")
        except Exception as e:
            logger.error(f"Failed to emit order: {e}")
            success = False

        return success
        
    def reset(self):
        """Reset the risk manager state."""
        self.position_state = {}
        self.orders = []



"""
Simple passthrough risk manager that directly converts signals to orders.
This is designed for testing and validation purposes.
"""

class SimplePassthroughRiskManager:
    """
    A simplified risk manager that directly passes signals to orders without filtering.
    Designed for testing and validation to ensure a 1:1 mapping between signals and trades.
    """
    
    def __init__(self, portfolio, event_bus=None):
        """
        Initialize the passthrough risk manager.
        
        Args:
            portfolio: Portfolio manager (used for reference only)
            event_bus: Event bus for emitting orders
        """
        self.portfolio = portfolio
        self.event_bus = event_bus
        self.broker = None  # Direct broker connection
        
        # Track generated orders for debugging
        self.orders = []
        
        # Track current position state for reference only (not used for decisions)
        self.position_state = {}
        
        # Debug flag
        self.debug = False
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
        return self
    
    def set_debug(self, debug=True):
        """Enable or disable debug mode."""
        self.debug = debug
        return self

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

        # Calculate position size (always 1 for simplicity)
        position_size = 1

        # CASE 1: BUY signal
        if signal_value == SignalEvent.BUY:
            # If currently short, close position first
            if current_state == -1:
                close_order = create_order_event(
                    symbol=symbol,
                    order_type="MARKET",
                    direction="BUY",  # Buy to cover short
                    quantity=1,
                    price=price,
                    timestamp=timestamp
                )

                # Emit order to close short position
                self._emit_order(close_order)
                self.position_state[symbol] = 0  # Update to neutral

                # Log position closure
                if self.debug:
                    logger.debug(f"Closed SHORT position for {symbol}: 1 @ {price:.2f}")

            # Now open long position if not already long
            if current_state != 1:  # If not already long
                long_order = create_order_event(
                    symbol=symbol,
                    order_type="MARKET",
                    direction="BUY",
                    quantity=1,
                    price=price,
                    timestamp=timestamp
                )

                # Emit order to open long position
                if self._emit_order(long_order):
                    self.position_state[symbol] = 1  # Mark as long
                    if self.debug:
                        logger.debug(f"Opened LONG position for {symbol}: 1 @ {price:.2f}")
            else:
                # Already long - do nothing
                if self.debug:
                    logger.debug(f"Ignoring BUY signal for {symbol}: already in LONG position")

        # CASE 2: SELL signal
        elif signal_value == SignalEvent.SELL:
            # If currently long, close position first
            if current_state == 1:
                close_order = create_order_event(
                    symbol=symbol,
                    order_type="MARKET",
                    direction="SELL",  # Sell to close long
                    quantity=1,
                    price=price,
                    timestamp=timestamp
                )

                # Emit order to close long position
                self._emit_order(close_order)
                self.position_state[symbol] = 0  # Update to neutral

                # Log position closure
                if self.debug:
                    logger.debug(f"Closed LONG position for {symbol}: 1 @ {price:.2f}")

            # Now open short position if not already short
            if current_state != -1:  # If not already short
                short_order = create_order_event(
                    symbol=symbol,
                    order_type="MARKET",
                    direction="SELL",
                    quantity=1,
                    price=price,
                    timestamp=timestamp
                )

                # Emit order to open short position
                if self._emit_order(short_order):
                    self.position_state[symbol] = -1  # Mark as short
                    if self.debug:
                        logger.debug(f"Opened SHORT position for {symbol}: 1 @ {price:.2f}")
            else:
                # Already short - do nothing
                if self.debug:
                    logger.debug(f"Ignoring SELL signal for {symbol}: already in SHORT position")
    

    
    def calculate_position_size(self, symbol, price, signal_strength=1.0):
        """Always returns 1 share for position sizing."""
        return 1

    def _emit_order(self, order):
        """
        Emit order event and track it.
        Returns success status.
        """
        if not order:
            return False

        # Add to order list for tracking
        self.orders.append(order)

        # Track success
        success = False

        try:
            # Try event bus first
            if self.event_bus:
                if self.debug:
                    logger.debug(f"Emitting order via event bus: {order.get_symbol()} {order.get_direction()}")
                self.event_bus.emit(order)
                success = True
            # Direct broker placement as backup
            elif self.broker:
                if self.debug:
                    logger.debug(f"Placing order with broker directly: {order.get_symbol()} {order.get_direction()}")
                self.broker.place_order(order)
                success = True
            else:
                logger.warning("No event bus or broker available - order not emitted!")
        except Exception as e:
            logger.error(f"Failed to emit order: {e}")
            success = False

        return success
    
    def reset(self):
        """Reset the risk manager state."""
        self.position_state = {}
        self.orders = []        
