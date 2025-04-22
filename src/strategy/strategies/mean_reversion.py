"""
Mean Reversion Strategy

This module implements a simple mean reversion strategy that generates
buy signals when price is significantly below its moving average and
sell signals when price is significantly above its moving average.
"""
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Union

from src.core.events.event_types import BarEvent, SignalEvent, EventType
from src.core.events.event_utils import create_signal_event

logger = logging.getLogger(__name__)

class MeanReversionStrategy:
    """
    Mean Reversion Strategy.
    
    Generates signals based on deviations from the mean:
    - Buy when price is significantly below the moving average (oversold)
    - Sell when price is significantly above the moving average (overbought)
    """
    
    def __init__(self, name="mean_reversion", symbols=None, lookback=20, z_threshold=1.5, price_key='close'):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            symbols: Symbol or list of symbols to trade
            lookback: Window for moving average and standard deviation
            z_threshold: Number of standard deviations for signals
            price_key: Price data to use ('close', 'open', etc.)
        """
        self.name = name
        self.symbols = symbols if symbols is not None else []
        
        # Convert single symbol to list
        if isinstance(self.symbols, str):
            self.symbols = [self.symbols]
            
        # Strategy parameters
        self.lookback = lookback
        self.z_threshold = z_threshold
        self.price_key = price_key
        
        # State data
        self.prices = {symbol: [] for symbol in self.symbols}
        self.signals = {symbol: 0 for symbol in self.symbols}  # Current signal for each symbol
        self.event_bus = None
        
        logger.info(f"Initialized MeanReversionStrategy: lookback={lookback}, z_threshold={z_threshold}")
    
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
        if 'z_threshold' in params:
            self.z_threshold = params['z_threshold']
        if 'price_key' in params:
            self.price_key = params['price_key']
            
        logger.debug(f"Updated parameters: lookback={self.lookback}, z_threshold={self.z_threshold}")
    
    def get_parameters(self):
        """Get current strategy parameters."""
        return {
            'lookback': self.lookback,
            'z_threshold': self.z_threshold,
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
        if len(self.prices[symbol]) < self.lookback:
            return None
            
        # Calculate mean and standard deviation
        prices = self.prices[symbol][-self.lookback:]
        mean = np.mean(prices)
        std = np.std(prices)
        
        # Avoid division by zero
        if std == 0:
            return None
            
        # Calculate z-score (number of standard deviations from mean)
        z_score = (price - mean) / std
        
        # Current position from previous signals
        current_position = self.signals[symbol]
        
        # Check for overbought/oversold conditions
        if z_score < -self.z_threshold and current_position <= 0:
            # Price is significantly below mean - buy signal
            signal = self._create_signal(symbol, SignalEvent.BUY, price, event.get_timestamp())
            self.signals[symbol] = 1  # Update position state
            return signal
            
        elif z_score > self.z_threshold and current_position >= 0:
            # Price is significantly above mean - sell signal
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
                'z_threshold': self.z_threshold
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
