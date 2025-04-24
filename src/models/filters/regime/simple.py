"""
Simple Regime-Filtered Strategy Wrapper

This module provides a minimal implementation of regime filtering for strategies.
It applies simple moving average slope detection to identify regimes and
filter signals accordingly.
"""
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union

from src.core.events.event_types import BarEvent, SignalEvent
from src.core.events.event_utils import create_signal_event

logger = logging.getLogger(__name__)


class SimpleRegimeFilteredStrategy:
    """
    A simple regime-filtered strategy wrapper that only allows BUY signals.
    """
    
    def __init__(self, base_strategy, ma_window=50):
        """
        Initialize the wrapper with a base strategy.
        
        Args:
            base_strategy: Strategy to wrap with regime filtering
            ma_window: Window size parameter (not used in this simplified version)
        """
        self.strategy = base_strategy
        self.name = f"regime_filtered_{self.strategy.name}"
        self.ma_window = ma_window  # Store but don't use
        
        # Get symbols from base strategy
        self.symbols = self.strategy.symbols if hasattr(self.strategy, 'symbols') else []
        
        # Signal filtering stats
        self.filtered_signals = 0
        self.passed_signals = 0
        
        # Event bus for signal emission
        self.event_bus = None
        
        print(f"Initialized simplified regime filter wrapper - ONLY ALLOWING BUY SIGNALS")
    
    def set_event_bus(self, event_bus):
        """Set the event bus for signal emission."""
        self.event_bus = event_bus
        self.strategy.set_event_bus(event_bus)
        return self
    
    def on_bar(self, event):
        """
        Process a bar event and apply filtering to signals.
        
        Args:
            event: BarEvent to process
            
        Returns:
            Optional[SignalEvent]: Filtered signal event or None
        """
        # Get signal from base strategy
        signal = self.strategy.on_bar(event)
        
        # If no signal, just pass through
        if not signal:
            return None
            
        # Check signal type
        from src.core.events.event_types import SignalEvent
        
        signal_value = signal.get_signal_value()
        
        # Simple filter: ONLY allow BUY signals, filter out SELL signals
        if signal_value == SignalEvent.BUY:
            # Pass BUY signal
            self.passed_signals += 1
            print(f"PASSING BUY signal")
            return signal
        else:
            # Filter SELL signal
            self.filtered_signals += 1
            print(f"FILTERING SELL signal")
            return None
    
    def get_regime_stats(self):
        """Get statistics about regime filtering."""
        return {
            'filtered_signals': self.filtered_signals,
            'passed_signals': self.passed_signals
        }
    
    def reset(self):
        """Reset the strategy state."""
        self.strategy.reset()
        self.filtered_signals = 0
        self.passed_signals = 0
    
    # Forward parameters to base strategy
    def get_parameters(self):
        """Get parameters from base strategy."""
        params = self.strategy.get_parameters() if hasattr(self.strategy, 'get_parameters') else {}
        params['ma_window'] = self.ma_window  # Add ma_window to parameters
        return params
    
    def set_parameters(self, params):
        """Set parameters for base strategy and this wrapper."""
        if 'ma_window' in params:
            self.ma_window = params['ma_window']
            
        # Forward all other parameters to base strategy
        base_params = {k: v for k, v in params.items() if k != 'ma_window'}
        if base_params and hasattr(self.strategy, 'set_parameters'):
            self.strategy.set_parameters(base_params)

# class SimpleRegimeFilteredStrategy:
#     """
#     Simple regime-filtered strategy wrapper.
    
#     This class wraps around an existing strategy and adds basic regime filtering:
#     - Detects trend direction using a moving average
#     - Only allows buy signals in uptrends
#     - Only allows sell signals in downtrends
#     - Allows both signals in sideways markets
#     """
    
#     def __init__(self, base_strategy, ma_window=50):
#         """
#         Initialize the wrapper with a base strategy.
        
#         Args:
#             base_strategy: Strategy to wrap with regime filtering
#             ma_window: Window for the moving average used in regime detection
#         """
#         self.strategy = base_strategy
#         self.ma_window = ma_window
#         self.name = f"regime_filtered_{self.strategy.name}"
        
#         # Get symbols from base strategy
#         self.symbols = self.strategy.symbols if hasattr(self.strategy, 'symbols') else []
        
#         # Tracking state
#         self.prices = {symbol: [] for symbol in self.symbols}
#         self.mas = {symbol: [] for symbol in self.symbols}
#         self.regimes = {symbol: 'unknown' for symbol in self.symbols}  # 'uptrend', 'downtrend', 'sideways'
        
#         # Signal filtering stats
#         self.filtered_signals = 0
#         self.passed_signals = 0
        
#         # Event bus for signal emission
#         self.event_bus = None
        
#         logger.info(f"Initialized SimpleRegimeFilteredStrategy with MA window={ma_window}")
    
#     def set_event_bus(self, event_bus):
#         """Set the event bus for signal emission."""
#         self.event_bus = event_bus
#         self.strategy.set_event_bus(event_bus)
#         return self


#     def detect_regime(self, symbol, price):
#         """
#         Detect market regime based on moving average slope.

#         Args:
#             symbol: Symbol to analyze
#             price: Current price

#         Returns:
#             str: Detected regime ('uptrend', 'downtrend', 'sideways')
#         """
#         # Add price to history
#         self.prices[symbol].append(price)

#         # Keep prices history limited
#         if len(self.prices[symbol]) > self.ma_window * 2:
#             self.prices[symbol] = self.prices[symbol][-self.ma_window * 2:]

#         # If we don't have enough prices, assume sideways regime
#         if len(self.prices[symbol]) < self.ma_window:
#             return 'sideways'  # Changed from unknown to sideways

#         # Calculate simple moving average
#         ma_value = np.mean(self.prices[symbol][-self.ma_window:])
#         self.mas[symbol].append(ma_value)

#         # Keep MA history limited
#         if len(self.mas[symbol]) > 10:  # Just need a few points to detect slope
#             self.mas[symbol] = self.mas[symbol][-10:]

#         # If we don't have enough MA points, assume sideways regime
#         if len(self.mas[symbol]) < 3:
#             return 'sideways'  # Changed from unknown to sideways

#         # Detect trend based on MA slope
#         ma_slope = self.mas[symbol][-1] - self.mas[symbol][-3]

#         # Use a moderate threshold - 0.15% of price
#         slope_threshold = 0.0015 * ma_value  # 0.15% of price

#         if ma_slope > slope_threshold:
#             regime = 'uptrend'
#         elif ma_slope < -slope_threshold:
#             regime = 'downtrend'
#         else:
#             regime = 'sideways'

#         return regime


#     def is_signal_allowed(self, signal_value, regime):
#         """
#         Determine if a signal is allowed in the current regime.

#         Args:
#             signal_value: Signal value (BUY, SELL)
#             regime: Current regime ('uptrend', 'downtrend', 'sideways', 'unknown')

#         Returns:
#             bool: True if signal is allowed, False if filtered
#         """
#         # More balanced filtering:

#         # Allow buy signals in uptrends and sideways markets
#         if signal_value == SignalEvent.BUY and (regime == 'uptrend' or regime == 'sideways'):
#             return True

#         # Allow sell signals in downtrends and sideways markets
#         if signal_value == SignalEvent.SELL and (regime == 'downtrend' or regime == 'sideways'):
#             return True

#         # Filter out: buy in downtrend, sell in uptrend
#         return False


 

    
#     def on_bar(self, event):
#         """
#         Process a bar event, detect regime, and apply filtering to signals.
        
#         Args:
#             event: BarEvent to process
            
#         Returns:
#             Optional[SignalEvent]: Filtered signal event or None
#         """
#         if not isinstance(event, BarEvent):
#             return None
            
#         symbol = event.get_symbol()
        
#         # Check if we should process this symbol
#         if symbol not in self.symbols:
#             return None
            
#         # First, update regime detection
#         price = event.get_close()
#         regime = self.detect_regime(symbol, price)
#         self.regimes[symbol] = regime
        
#         # Get signal from base strategy
#         signal = self.strategy.on_bar(event)
        
#         # If no signal, just pass through
#         if not signal:
#             return None
            
#         # Apply regime filtering
#         signal_value = signal.get_signal_value()
        
#         if self.is_signal_allowed(signal_value, regime):
#             # Signal passes regime filter
#             self.passed_signals += 1
#             logger.debug(f"Signal {signal_value} allowed in regime {regime} for {symbol}")
#             return signal
#         else:
#             # Signal blocked by regime filter
#             self.filtered_signals += 1
#             logger.debug(f"Signal {signal_value} filtered out in regime {regime} for {symbol}")
#             return None
    
#     def get_regime_stats(self):
#         """Get statistics about regime filtering."""
#         return {
#             'filtered_signals': self.filtered_signals,
#             'passed_signals': self.passed_signals,
#             'current_regimes': self.regimes.copy()
#         }
    
#     def reset(self):
#         """Reset the strategy and tracking state."""
#         self.strategy.reset()
#         self.prices = {symbol: [] for symbol in self.symbols}
#         self.mas = {symbol: [] for symbol in self.symbols}
#         self.regimes = {symbol: 'unknown' for symbol in self.symbols}
#         self.filtered_signals = 0
#         self.passed_signals = 0
    
#     # Forward parameters to base strategy
#     def get_parameters(self):
#         """Get parameters from base strategy."""
#         params = self.strategy.get_parameters() if hasattr(self.strategy, 'get_parameters') else {}
#         params['ma_window'] = self.ma_window
#         return params
    
#     def set_parameters(self, params):
#         """Set parameters for base strategy and this wrapper."""
#         if 'ma_window' in params:
#             self.ma_window = params['ma_window']
            
#         # Forward all other parameters to base strategy
#         base_params = {k: v for k, v in params.items() if k != 'ma_window'}
#         if base_params and hasattr(self.strategy, 'set_parameters'):
#             self.strategy.set_parameters(base_params)
