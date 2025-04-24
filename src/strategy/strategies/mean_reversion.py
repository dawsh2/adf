"""
Modified Mean Reversion Strategy that ignores position state when generating signals.

This version generates signals based solely on z-score crossings,
regardless of current position state. It's designed for validation testing.
"""
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Union

from src.core.events.event_types import BarEvent, SignalEvent, EventType
from src.core.events.event_utils import create_signal_event

logger = logging.getLogger(__name__)

class MeanReversionStrategy:
    """
    Modified Mean Reversion Strategy for validation testing.
    
    Generates signals based solely on z-score crossings without position filtering:
    - Buy when price crosses below -z_threshold
    - Sell when price crosses above +z_threshold
    """
    
    def __init__(self, name="mean_reversion_modified", symbols=None, lookback=20, z_threshold=1.5, price_key='close'):
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
        self.z_scores = {symbol: [] for symbol in self.symbols}
        self.signals = {symbol: 0 for symbol in self.symbols}  # Current signal for each symbol
        self.event_bus = None
        
        # Track signal history for diagnostic purposes
        self.signal_history = []
        
        logger.info(f"Initialized ModifiedMeanReversionStrategy: lookback={lookback}, z_threshold={z_threshold}")
    
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
        Process a bar event and generate signals based solely on z-score crossings.
        
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
        
        # Update z-score history
        self.z_scores[symbol].append(z_score)
        if len(self.z_scores[symbol]) > self.lookback * 2:
            self.z_scores[symbol] = self.z_scores[symbol][-self.lookback * 2:]
        
        # Need at least 2 z-scores to detect crossings
        if len(self.z_scores[symbol]) < 2:
            return None
        
        # Get current and previous z-scores
        current_z = z_score
        previous_z = self.z_scores[symbol][-2]
        
        # MODIFIED: Generate signals based solely on z-score crossings without position filtering
        
        # Check for buy signal (crossing below -z_threshold)
        if previous_z >= -self.z_threshold and current_z < -self.z_threshold:
            # Track signal in history
            self.signal_history.append({
                'timestamp': event.get_timestamp(),
                'symbol': symbol,
                'z_score': current_z,
                'price': price,
                'type': 'BUY',
                'crossing': f"{previous_z:.2f} → {current_z:.2f}"
            })
            
            # Generate buy signal
            signal = self._create_signal(symbol, SignalEvent.BUY, price, event.get_timestamp())
            self.signals[symbol] = 1  # Track current signal state
            return signal
            
        # Check for sell signal (crossing above +z_threshold)
        elif previous_z <= self.z_threshold and current_z > self.z_threshold:
            # Track signal in history
            self.signal_history.append({
                'timestamp': event.get_timestamp(),
                'symbol': symbol,
                'z_score': current_z,
                'price': price,
                'type': 'SELL',
                'crossing': f"{previous_z:.2f} → {current_z:.2f}"
            })
            
            # Generate sell signal
            signal = self._create_signal(symbol, SignalEvent.SELL, price, event.get_timestamp())
            self.signals[symbol] = -1  # Track current signal state
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
        self.z_scores = {symbol: [] for symbol in self.symbols}
        self.signals = {symbol: 0 for symbol in self.symbols}
        self.signal_history = []
    
    def get_signal_history(self):
        """Get the signal generation history for analysis."""
        return self.signal_history
