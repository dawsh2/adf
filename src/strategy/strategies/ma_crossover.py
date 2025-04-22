"""
Moving Average Crossover Strategy

This module implements a classic moving average crossover strategy that
generates buy signals when fast MA crosses above slow MA and sell signals
when fast MA crosses below slow MA.
"""
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Union

from src.core.events.event_types import BarEvent, SignalEvent, EventType
from src.core.events.event_utils import create_signal_event

logger = logging.getLogger(__name__)

class MovingAverageCrossoverStrategy:
    """
    Moving Average Crossover Strategy.
    
    Generates signals based on the crossover of two moving averages.
    - Buy when fast MA crosses above slow MA
    - Sell when fast MA crosses below slow MA
    """
    
    def __init__(self, name="ma_crossover", symbols=None, fast_window=10, slow_window=30, price_key='close'):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            symbols: Symbol or list of symbols to trade
            fast_window: Fast moving average window
            slow_window: Slow moving average window
            price_key: Price data to use for MA calculation ('close', 'open', etc.)
        """
        self.name = name
        self.symbols = symbols if symbols is not None else []
        
        # Convert single symbol to list
        if isinstance(self.symbols, str):
            self.symbols = [self.symbols]
            
        # Strategy parameters
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.price_key = price_key
        
        # Validation
        if self.fast_window >= self.slow_window:
            logger.warning(f"Fast window ({fast_window}) should be less than slow window ({slow_window})")
        
        # State data
        self.prices = {symbol: [] for symbol in self.symbols}
        self.signals = {symbol: 0 for symbol in self.symbols}  # Current signal for each symbol
        self.event_bus = None
        
        logger.info(f"Initialized MovingAverageCrossoverStrategy: fast_window={fast_window}, slow_window={slow_window}")
    
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
        if 'fast_window' in params:
            self.fast_window = params['fast_window']
        if 'slow_window' in params:
            self.slow_window = params['slow_window']
        if 'price_key' in params:
            self.price_key = params['price_key']
        
        # Re-validate
        if self.fast_window >= self.slow_window:
            logger.warning(f"Fast window ({self.fast_window}) should be less than slow window ({self.slow_window})")
            
        logger.debug(f"Updated parameters: fast_window={self.fast_window}, slow_window={self.slow_window}")
    
    def get_parameters(self):
        """Get current strategy parameters."""
        return {
            'fast_window': self.fast_window,
            'slow_window': self.slow_window,
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
        max_window = max(self.fast_window, self.slow_window)
        if len(self.prices[symbol]) > max_window * 2:
            self.prices[symbol] = self.prices[symbol][-max_window * 2:]
            
        # Need enough data for both moving averages
        if len(self.prices[symbol]) < self.slow_window:
            return None
            
        # Calculate moving averages
        prices = self.prices[symbol]
        fast_ma = np.mean(prices[-self.fast_window:])
        slow_ma = np.mean(prices[-self.slow_window:])
        
        # If we have enough history, check the previous MAs for crossover
        if len(prices) > self.slow_window:
            prev_prices = prices[:-1]
            prev_fast_ma = np.mean(prev_prices[-self.fast_window:])
            prev_slow_ma = np.mean(prev_prices[-self.slow_window:])
            
            # Check for crossover
            # Current fast MA > slow MA and previously fast MA < slow MA = Buy
            if fast_ma > slow_ma and prev_fast_ma <= prev_slow_ma:
                signal = self._create_signal(symbol, SignalEvent.BUY, price, event.get_timestamp())
                self.signals[symbol] = 1  # Record current signal state
                return signal
                
            # Current fast MA < slow MA and previously fast MA > slow MA = Sell
            elif fast_ma < slow_ma and prev_fast_ma >= prev_slow_ma:
                signal = self._create_signal(symbol, SignalEvent.SELL, price, event.get_timestamp())
                self.signals[symbol] = -1  # Record current signal state
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
                'fast_ma': self.fast_window,
                'slow_ma': self.slow_window
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
