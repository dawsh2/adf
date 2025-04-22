"""
Momentum Strategy

This module implements a simple momentum strategy that generates
buy signals when price rises by a certain percentage and sell signals
when price falls by a certain percentage.
"""
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Union

from src.core.events.event_types import BarEvent, SignalEvent, EventType
from src.core.events.event_utils import create_signal_event

logger = logging.getLogger(__name__)

class MomentumStrategy:
    """
    Momentum Strategy.
    
    Generates signals based on price momentum:
    - Buy when price rises by a certain percentage
    - Sell when price falls by a certain percentage
    """
    
    def __init__(self, name="momentum", symbols=None, lookback=10, threshold=0.01, price_key='close'):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            symbols: Symbol or list of symbols to trade
            lookback: Window for momentum calculation
            threshold: Percentage change threshold for signal generation
            price_key: Price data to use ('close', 'open', etc.)
        """
        self.name = name
        self.symbols = symbols if symbols is not None else []
        
        # Convert single symbol to list
        if isinstance(self.symbols, str):
            self.symbols = [self.symbols]
            
        # Strategy parameters
        self.lookback = lookback
        self.threshold = threshold
        self.price_key = price_key
        
        # State data
        self.prices = {symbol: [] for symbol in self.symbols}
        self.signals = {symbol: 0 for symbol in self.symbols}  # Current signal for each symbol
        self.event_bus = None
        
        logger.info(f"Initialized MomentumStrategy: lookback={lookback}, threshold={threshold}")
    
    def set_event_bus(self, event_bus):
        """Set the event bus for signal emission."""
        self.event_bus = event_bus
        return self
    
    def set_parameters(self, params):
        """
        Update strategy parameters.
        
        Args:
            params: Dictionary of parameters to update
        """
        if 'lookback' in params:
            self.lookback = params['lookback']
        if 'threshold' in params:
            self.threshold = params['threshold']
        if 'price_key' in params:
            self.price_key = params['price_key']
            
        logger.debug(f"Updated parameters: lookback={self.lookback}, threshold={self.threshold}")
    
    def get_parameters(self):
        """Get current strategy parameters."""
        return {
            'lookback': self.lookback,
            'threshold': self.threshold,
            'price_key': self.price_key
        }
    
    def on_bar(self, event):
        """
        Process a bar event and generate signals.
        
        Args:
            event: BarEvent to process
            
        Returns:
            Optional[SignalEvent]: Generated signal event or None
        """
        if not isinstance(event, BarEvent):
            return None
            
        symbol = event.get_symbol()
        
        # Check if we should process this symbol
        if symbol not in self.symbols:
            return None
            
        # Extract price data
        if self.price_key == 'close' or not hasattr(event, f'get_{self.price_key}'):
            price = event.get_close()
        else:
            price = getattr(event, f'get_{self.price_key}')()
            
        # Update price history
        self.prices[symbol].append(price)
        
        # Keep history limited to what we need
        if len(self.prices[symbol]) > self.lookback * 2:
            self.prices[symbol] = self.prices[symbol][-self.lookback * 2:]
            
        # Need enough data for calculations
        if len(self.prices[symbol]) <= self.lookback:
            return None
            
        # Calculate momentum (percentage change over lookback period)
        current_price = self.prices[symbol][-1]
        past_price = self.prices[symbol][-self.lookback - 1]
        
        # Avoid division by zero
        if past_price == 0:
            return None
            
        momentum = (current_price / past_price) - 1
        
        # Current position from previous signals
        current_position = self.signals[symbol]
        
        # Check for momentum thresholds
        if momentum > self.threshold and current_position <= 0:
            # Positive momentum - buy signal
            signal = self._create_signal(symbol, SignalEvent.BUY, price, event.get_timestamp())
            self.signals[symbol] = 1  # Update position state
            return signal
            
        elif momentum < -self.threshold and current_position >= 0:
            # Negative momentum - sell signal
            signal = self._create_signal(symbol, SignalEvent.SELL, price, event.get_timestamp())
            self.signals[symbol] = -1  # Update position state
            return signal
            
        return None
    
    def _create_signal(self, symbol, signal_type, price, timestamp=None):
        """
        Create a signal event.
        
        Args:
            symbol: Symbol to signal for
            signal_type: Signal type constant (BUY, SELL, NEUTRAL)
            price: Current price
            timestamp: Optional timestamp
            
        Returns:
            SignalEvent: Created signal event
        """
        # Create signal
        signal = create_signal_event(
            signal_value=signal_type,
            price=price,
            symbol=symbol,
            rule_id=self.name,
            confidence=1.0,
            metadata={
                'lookback': self.lookback,
                'threshold': self.threshold
            },
            timestamp=timestamp
        )
        
        # Emit if we have an event bus
        if self.event_bus:
            self.event_bus.emit(signal)
            
        return signal
    
    def reset(self):
        """Reset the strategy state."""
        self.prices = {symbol: [] for symbol in self.symbols}
        self.signals = {symbol: 0 for symbol in self.symbols}
