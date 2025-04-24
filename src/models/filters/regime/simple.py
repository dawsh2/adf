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
            ma_window: Window size parameter (stored but not used in this implementation)
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
        
        print(f"Initialized simplified regime filter wrapper with ma_window={ma_window} - ONLY ALLOWING BUY SIGNALS")
    
    def set_event_bus(self, event_bus):
        """Set the event bus for signal emission."""
        self.event_bus = event_bus
        # Important: Prevent the base strategy from emitting directly
        if hasattr(self.strategy, 'set_event_bus'):
            self.strategy.set_event_bus(None)
        return self


    def on_bar(self, event):
        """
        Process a bar event and apply filtering to signals.
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
            print(f"PASSING BUY signal #{self.passed_signals}")

            # CRITICAL: Explicitly emit the signal on the event bus
            if self.event_bus:
                #print(f"Explicitly emitting BUY signal on event bus")
                self.event_bus.emit(signal)

            # Return the signal as well
            return signal
        else:
            # Filter SELL signal
            self.filtered_signals += 1
            print(f"FILTERING SELL signal #{self.filtered_signals}")
            return None
    
 
    
    def get_regime_stats(self):
        """Get statistics about regime filtering."""
        return {
            'filtered_signals': self.filtered_signals,
            'passed_signals': self.passed_signals
        }
    
    def reset(self):
        """Reset the strategy state."""
        if hasattr(self.strategy, 'reset'):
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
