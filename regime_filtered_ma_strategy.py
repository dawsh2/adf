"""
Regime-Filtered Moving Average Crossover Strategy

This module implements a moving average crossover strategy that filters signals
based on the detected market regime.
"""
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Union

from src.core.events.event_types import BarEvent, SignalEvent, EventType
from src.core.events.event_utils import create_signal_event
from src.strategy.strategies.ma_crossover import MovingAverageCrossoverStrategy

# Import regime detection components
from src.models.filters.regime.regime_detector import MarketRegime # 

logger = logging.getLogger(__name__)

class RegimeFilteredMAStrategy:
    """
    Moving Average Crossover Strategy with Regime Filtering.
    
    Implements a regime-filtered moving average crossover strategy that:
    - Uses a regime detector to identify market conditions
    - Only generates signals that align with the current market regime
    - Uses MA crossovers for signal generation with regime confirmation
    """
    
    def __init__(self, name="regime_filtered_ma", symbols=None, 
                fast_window=10, slow_window=30, price_key='close',
                regime_detector=None, allowed_regimes=None):
        """
        Initialize the regime-filtered strategy.
        
        Args:
            name: Strategy name
            symbols: Symbol or list of symbols to trade
            fast_window: Fast moving average window
            slow_window: Slow moving average window
            price_key: Price data to use for MA calculation ('close', 'open', etc.)
            regime_detector: Regime detector instance for detecting regimes
            allowed_regimes: Dict mapping regimes to allowed signal types
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
        
        # Create base MA crossover strategy
        self.ma_strategy = MovingAverageCrossoverStrategy(
            name=f"{name}_base",
            symbols=self.symbols,
            fast_window=fast_window,
            slow_window=slow_window,
            price_key=price_key
        )
        
        # Regime detector
        self.regime_detector = regime_detector
        
        # Set default allowed regimes if none provided
        self.allowed_regimes = allowed_regimes or {
            MarketRegime.UPTREND: [SignalEvent.BUY],
            MarketRegime.DOWNTREND: [SignalEvent.SELL],
            MarketRegime.SIDEWAYS: [SignalEvent.BUY, SignalEvent.SELL],
            MarketRegime.VOLATILE: [],  # No trading in volatile regimes
            MarketRegime.UNKNOWN: [SignalEvent.BUY, SignalEvent.SELL]
        }
        
        # Event bus for signal emission
        self.event_bus = None
        
        # Tracking state
        self.current_regimes = {symbol: MarketRegime.UNKNOWN for symbol in self.symbols}
        self.filtered_signals = 0
        self.passed_signals = 0
        
        logger.info(f"Initialized RegimeFilteredMAStrategy: fast_window={fast_window}, slow_window={slow_window}")
    
    def set_event_bus(self, event_bus):
        """Set the event bus for signal emission."""
        self.event_bus = event_bus
        self.ma_strategy.set_event_bus(event_bus)
        return self
    
    def set_parameters(self, params):
        """
        Update strategy parameters.
        
        Args:
            params: Dictionary of parameters to update
        """
        # Update local parameters
        if 'fast_window' in params:
            self.fast_window = params['fast_window']
        if 'slow_window' in params:
            self.slow_window = params['slow_window']
        if 'price_key' in params:
            self.price_key = params['price_key']
            
        # Update base MA strategy
        ma_params = {k: v for k, v in params.items() 
                    if k in ['fast_window', 'slow_window', 'price_key']}
        if ma_params:
            self.ma_strategy.set_parameters(ma_params)
            
        # Update allowed regimes if provided
        if 'allowed_regimes' in params:
            self.allowed_regimes = params['allowed_regimes']
            
        logger.debug(f"Updated parameters: fast_window={self.fast_window}, slow_window={self.slow_window}")
    
    def get_parameters(self):
        """Get current strategy parameters."""
        params = {
            'fast_window': self.fast_window,
            'slow_window': self.slow_window,
            'price_key': self.price_key,
            'allowed_regimes': self.allowed_regimes
        }
        return params

    # In regime_filtered_ma_strategy.py
    def on_bar(self, event):
        """
        Process a bar event, detect regime, and apply filtering to signals.

        Args:
            event: BarEvent to process

        Returns:
            Optional[SignalEvent]: Filtered signal event or None
        """
        if not isinstance(event, BarEvent):
            return None

        symbol = event.get_symbol()

        # Check if we should process this symbol
        if symbol not in self.symbols:
            return None

        # First, update regime detection if we have a detector
        if self.regime_detector:
            regime = self.regime_detector.update(event)
            self.current_regimes[symbol] = regime

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Current regime for {symbol}: {regime.value}")
        else:
            # Default to unknown regime if no detector
            regime = MarketRegime.UNKNOWN
            self.current_regimes[symbol] = regime

        # Get signal from base MA strategy (don't emit this signal yet)
        # We're only using the base strategy to generate potential signals
        # The event bus emission happens after regime filtering
        original_event_bus = self.ma_strategy.event_bus
        self.ma_strategy.event_bus = None  # Temporarily disable event bus

        signal = self.ma_strategy.on_bar(event)

        # Restore original event bus to base strategy
        self.ma_strategy.event_bus = original_event_bus

        # If no signal generated, nothing to filter
        if not signal:
            return None

        # Apply regime filtering
        signal_value = signal.get_signal_value()
        allowed_signals = self.allowed_regimes.get(regime, [])

        # Check if signal is allowed in current regime
        if signal_value in allowed_signals:
            # Signal passes regime filter
            self.passed_signals += 1
            logger.debug(f"Signal {signal_value} allowed in regime {regime.value} for {symbol}")

            # Emit signal through event bus
            if self.event_bus:
                self.event_bus.emit(signal)
                logger.debug(f"Emitted signal event to event bus for {symbol}")

            return signal
        else:
            # Signal blocked by regime filter
            self.filtered_signals += 1
            logger.debug(f"Signal {signal_value} filtered out in regime {regime.value} for {symbol}")
            return None
    
 
    
    def get_regime_stats(self):
        """Get statistics about regime filtering."""
        return {
            'filtered_signals': self.filtered_signals,
            'passed_signals': self.passed_signals,
            'current_regimes': {symbol: regime.value for symbol, regime in self.current_regimes.items()}
        }
    
    def reset(self):
        """Reset the strategy state."""
        self.ma_strategy.reset()
        if self.regime_detector:
            self.regime_detector.reset()
        self.current_regimes = {symbol: MarketRegime.UNKNOWN for symbol in self.symbols}
        self.filtered_signals = 0
        self.passed_signals = 0
