"""
Mean Reversion Trading Strategy

A strategy that buys when prices fall too far below a moving average
and sells/shorts when prices rise too far above a moving average.
"""
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Union

from src.core.events.event_types import BarEvent, SignalEvent, EventType
from src.core.events.event_utils import create_signal_event

logger = logging.getLogger(__name__)

class MeanReversionStrategy:
    """
    Mean Reversion Trading Strategy.
    
    Generates signals based on price deviations from a moving average:
    - Buy when price falls below MA - (MA * threshold)
    - Sell when price rises above MA + (MA * threshold)
    """
    
    def __init__(self, name="mean_reversion", symbols=None, window=20, threshold=0.02, allow_short=True):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            symbols: Symbol or list of symbols to trade
            window: Moving average window
            threshold: Threshold for trading signals (as percentage of MA)
            allow_short: Whether to allow short selling
        """
        self.name = name
        self.symbols = symbols if symbols is not None else []
        
        # Convert single symbol to list
        if isinstance(self.symbols, str):
            self.symbols = [self.symbols]
            
        # Strategy parameters
        self.window = window
        self.threshold = threshold
        self.allow_short = allow_short
        
        # State data
        self.prices = {symbol: [] for symbol in self.symbols}
        self.signals = {symbol: 0 for symbol in self.symbols}  # Current signal for each symbol (1=long, -1=short, 0=none)
        self.mas = {symbol: None for symbol in self.symbols}   # Current moving average
        self.event_bus = None
        
        logger.info(f"Initialized MeanReversionStrategy: window={window}, threshold={threshold}")
    
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
        if 'window' in params:
            self.window = params['window']
        if 'threshold' in params:
            self.threshold = params['threshold']
        if 'allow_short' in params:
            self.allow_short = params['allow_short']
            
        logger.debug(f"Updated parameters: window={self.window}, threshold={self.threshold}")
    
    def get_parameters(self):
        """Get current strategy parameters."""
        return {
            'window': self.window,
            'threshold': self.threshold,
            'allow_short': self.allow_short
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
            
        # Extract price data (using close price)
        price = event.get_close()
            
        # Update price history
        self.prices[symbol].append(price)
        
        # Keep history limited to what we need
        if len(self.prices[symbol]) > self.window * 2:
            self.prices[symbol] = self.prices[symbol][-self.window * 2:]
            
        # Need enough data for moving average
        if len(self.prices[symbol]) < self.window:
            return None
            
        # Calculate moving average
        prices = self.prices[symbol]
        ma = np.mean(prices[-self.window:])
        
        # Store current MA
        self.mas[symbol] = ma
        
        # Calculate threshold values
        lower_band = ma * (1 - self.threshold)
        upper_band = ma * (1 + self.threshold)
        
        # Current position state (-1=short, 0=none, 1=long)
        current_signal = self.signals.get(symbol, 0)
        
        # Generate trading signals based on price vs bands
        if price < lower_band:
            # Price below lower band - buy signal
            if current_signal <= 0:  # If not already long
                signal = self._create_signal(symbol, SignalEvent.BUY, price, event.get_timestamp())
                self.signals[symbol] = 1  # Record current signal state
                return signal
        elif price > upper_band:
            # Price above upper band - sell signal
            if current_signal >= 0 and self.allow_short:  # If not already short and shorting allowed
                signal = self._create_signal(symbol, SignalEvent.SELL, price, event.get_timestamp())
                self.signals[symbol] = -1  # Record current signal state
                return signal
                
        # No signal
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
                'ma': self.mas[symbol],
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
        self.mas = {symbol: None for symbol in self.symbols}
