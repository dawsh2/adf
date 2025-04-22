"""
Regime-Aware Strategy Module

This module provides regime-aware strategy components that adapt their behavior
based on the current market regime detected by a regime detector.
"""
import logging
import datetime
import random
from typing import Dict, Any, Optional, List, Union, Type

from src.core.events.event_types import BarEvent
from .regime_detector import RegimeDetectorBase, MarketRegime

logger = logging.getLogger(__name__)

class RegimeAwareStrategy:
    """
    Strategy wrapper that adapts parameters based on market regime.
    
    This component wraps around a base strategy and switches its parameters
    dynamically based on the current market regime detected by a regime detector.
    """
    
    def __init__(self, base_strategy, regime_detector: RegimeDetectorBase):
        """
        Initialize the regime-aware strategy wrapper.
        
        Args:
            base_strategy: Base strategy to enhance with regime awareness
            regime_detector: Regime detector instance
        """
        self.strategy = base_strategy
        self.regime_detector = regime_detector
        self.event_bus = None
        
        # Construct name from base strategy
        if hasattr(base_strategy, 'name'):
            self.name = f"regime_aware_{base_strategy.name}"
        else:
            self.name = f"regime_aware_strategy"
        
        # Get symbols from base strategy
        if hasattr(base_strategy, 'symbols'):
            self.symbols = base_strategy.symbols
        else:
            self.symbols = []
        
        # Parameter sets for different regimes
        self.regime_parameters = {
            MarketRegime.UPTREND: {},
            MarketRegime.DOWNTREND: {},
            MarketRegime.SIDEWAYS: {},
            MarketRegime.VOLATILE: {},
            MarketRegime.UNKNOWN: {}  # Default parameters
        }
        
        # Current active parameters
        self.active_regime = {symbol: MarketRegime.UNKNOWN for symbol in self.symbols}
        
        # Regime transition stats
        self.regime_transitions = []
        self.parameter_changes = []
    
    def set_event_bus(self, event_bus):
        """
        Set the event bus.
        
        Args:
            event_bus: Event bus instance
            
        Returns:
            self: For method chaining
        """
        self.event_bus = event_bus
        
        # Propagate to base strategy if it has set_event_bus method
        if hasattr(self.strategy, 'set_event_bus'):
            self.strategy.set_event_bus(event_bus)
            
        return self
    
    def set_regime_parameters(self, regime: MarketRegime, parameters: Dict[str, Any]):
        """
        Set parameters for a specific regime.
        
        Args:
            regime: MarketRegime to set parameters for
            parameters: Parameter dict for this regime
        """
        self.regime_parameters[regime] = parameters
        
        # If this is for the current regime of any symbol, apply immediately
        for symbol in self.symbols:
            if self.active_regime[symbol] == regime:
                logger.info(f"Applying new {regime.value} parameters to {symbol}: {parameters}")
                
                if hasattr(self.strategy, 'set_parameters'):
                    self.strategy.set_parameters(parameters)
                else:
                    # If no set_parameters method, try direct attribute assignment
                    for param_name, param_value in parameters.items():
                        if hasattr(self.strategy, param_name):
                            setattr(self.strategy, param_name, param_value)
                
                # Record parameter change
                self._record_parameter_change(symbol, regime, parameters)

    def on_bar(self, event):
        """
        Process a bar event with regime-specific parameters.

        Args:
            event: Bar event to process

        Returns:
            Signal event or None
        """
        if not isinstance(event, BarEvent):
            return None
            
        symbol = event.get_symbol()
        timestamp = event.get_timestamp()

        if symbol not in self.symbols:
            return None

        # Detect current regime
        current_regime = self.regime_detector.update(event)

        # Check if regime changed
        if current_regime != self.active_regime[symbol]:
            # Record the regime change
            self._record_regime_transition(symbol, self.active_regime[symbol], current_regime, timestamp)
            
            # Switch parameters
            self._switch_parameters(symbol, current_regime, timestamp)

        # Log active parameters occasionally
        if random.random() < 0.01:  # Log roughly 1% of the time
            logger.debug(f"Active parameters for {symbol}: " +
                       f"{self._get_current_parameters()}, Regime: {current_regime.value}")

        # Process with current parameters by delegating to base strategy
        if hasattr(self.strategy, 'on_bar'):
            return self.strategy.on_bar(event)
        
        return None
    
    def _switch_parameters(self, symbol: str, regime: MarketRegime, timestamp=None):
        """
        Switch strategy parameters based on regime.
        
        Args:
            symbol: Symbol that experienced regime change
            regime: New regime for the symbol
            timestamp: Optional timestamp of the change
        """
        logger.info(f"Regime change for {symbol}: {self.active_regime[symbol].value} -> {regime.value}")
        
        # Update active regime
        self.active_regime[symbol] = regime
        
        # Get parameters for this regime
        parameters = self.regime_parameters.get(regime, {})
        
        if parameters:
            logger.info(f"Switching to {regime.value} parameters for {symbol}: {parameters}")
            
            # Apply parameters to base strategy
            if hasattr(self.strategy, 'set_parameters'):
                self.strategy.set_parameters(parameters)
            else:
                # If no set_parameters method, try direct attribute assignment
                for param_name, param_value in parameters.items():
                    if hasattr(self.strategy, param_name):
                        setattr(self.strategy, param_name, param_value)
            
            # Record parameter change
            self._record_parameter_change(symbol, regime, parameters, timestamp)
        else:
            logger.warning(f"No parameters defined for {regime.value} regime")
    
    def _record_regime_transition(self, symbol: str, old_regime: MarketRegime, new_regime: MarketRegime, timestamp=None):
        """
        Record a regime transition for analysis.
        
        Args:
            symbol: Symbol with regime change
            old_regime: Previous regime
            new_regime: New regime
            timestamp: Optional timestamp of transition
        """
        if timestamp is None:
            timestamp = datetime.datetime.now()
            
        transition = {
            'timestamp': timestamp,
            'symbol': symbol,
            'from_regime': old_regime.value,
            'to_regime': new_regime.value
        }
        
        self.regime_transitions.append(transition)
    
    def _record_parameter_change(self, symbol: str, regime: MarketRegime, parameters: Dict[str, Any], timestamp=None):
        """
        Record a parameter change for analysis.
        
        Args:
            symbol: Symbol with parameter change
            regime: Regime causing the change
            parameters: New parameters
            timestamp: Optional timestamp of change
        """
        if timestamp is None:
            timestamp = datetime.datetime.now()
            
        change = {
            'timestamp': timestamp,
            'symbol': symbol,
            'regime': regime.value,
            'parameters': parameters
        }
        
        self.parameter_changes.append(change)
    
    def _get_current_parameters(self) -> Dict[str, Any]:
        """
        Get current strategy parameters.
        
        Returns:
            dict: Current parameter values
        """
        if hasattr(self.strategy, 'get_parameters'):
            return self.strategy.get_parameters()
        
        # If no get_parameters method, try to infer from regime parameters
        # for the active regimes
        active_params = {}
        for symbol, regime in self.active_regime.items():
            if regime in self.regime_parameters:
                active_params.update(self.regime_parameters[regime])
                
        return active_params
    
    def reset(self):
        """Reset the strategy and detector."""
        # Reset base strategy
        if hasattr(self.strategy, 'reset'):
            self.strategy.reset()
            
        # Reset regime detector
        self.regime_detector.reset()
        
        # Reset local state
        self.active_regime = {symbol: MarketRegime.UNKNOWN for symbol in self.symbols}
        self.regime_transitions = []
        self.parameter_changes = []
    
    def get_regime_stats(self) -> Dict[str, Any]:
        """
        Get statistics on regime transitions and parameter changes.
        
        Returns:
            dict: Regime statistics
        """
        result = {
            'transitions': len(self.regime_transitions),
            'parameter_changes': len(self.parameter_changes),
            'current_regimes': {symbol: regime.value for symbol, regime in self.active_regime.items()},
            'transition_history': self.regime_transitions[-10:],  # Last 10 transitions
            'parameter_history': self.parameter_changes[-10:]  # Last 10 parameter changes
        }
        
        # Calculate regime distribution
        regime_counts = {}
        for transition in self.regime_transitions:
            regime = transition['to_regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
        result['regime_distribution'] = regime_counts
        
        return result


class MultiRegimeStrategy:
    """
    Strategy that uses different sub-strategies depending on market regime.
    
    Instead of just switching parameters, this strategy completely switches
    the underlying strategy implementation based on the market regime.
    """
    
    def __init__(self, regime_detector: RegimeDetectorBase, default_strategy=None):
        """
        Initialize the multi-regime strategy.
        
        Args:
            regime_detector: Regime detector instance
            default_strategy: Optional default strategy for unknown regimes
        """
        self.regime_detector = regime_detector
        self.event_bus = None
        self.name = "multi_regime_strategy"
        self.symbols = []
        
        # Strategies for different regimes
        self.strategies = {
            MarketRegime.UPTREND: None,
            MarketRegime.DOWNTREND: None,
            MarketRegime.SIDEWAYS: None,
            MarketRegime.VOLATILE: None,
            MarketRegime.UNKNOWN: default_strategy
        }
        
        # Current active regimes
        self.active_regime = {}
        
        # Regime transition stats
        self.regime_transitions = []
        self.strategy_switches = []
    
    def set_event_bus(self, event_bus):
        """
        Set the event bus and propagate to all sub-strategies.
        
        Args:
            event_bus: Event bus instance
            
        Returns:
            self: For method chaining
        """
        self.event_bus = event_bus
        
        # Propagate to all strategies
        for regime, strategy in self.strategies.items():
            if strategy and hasattr(strategy, 'set_event_bus'):
                strategy.set_event_bus(event_bus)
                
        return self
    
    def set_strategy_for_regime(self, regime: MarketRegime, strategy):
        """
        Set the strategy to use for a specific regime.
        
        Args:
            regime: Market regime
            strategy: Strategy instance for this regime
        """
        self.strategies[regime] = strategy
        
        # Propagate event bus if set
        if self.event_bus and hasattr(strategy, 'set_event_bus'):
            strategy.set_event_bus(self.event_bus)
            
        # Update symbols list
        self._update_symbols()
        
        # If this is the active regime for any symbol, update active strategy
        for symbol, active_regime in self.active_regime.items():
            if active_regime == regime:
                logger.info(f"Updating active strategy for {symbol} to {strategy.__class__.__name__}")
                
                # Record strategy switch
                self._record_strategy_switch(symbol, regime, 
                                         strategy.__class__.__name__ if strategy else None)
    
    def on_bar(self, event):
        """
        Process a bar event with regime-specific strategy.

        Args:
            event: Bar event to process

        Returns:
            Signal event or None
        """
        if not isinstance(event, BarEvent):
            return None
            
        symbol = event.get_symbol()
        timestamp = event.get_timestamp()

        if symbol not in self.symbols:
            # Add new symbol if found in event
            self.symbols.append(symbol)
            self.active_regime[symbol] = MarketRegime.UNKNOWN
            
        # Detect current regime
        current_regime = self.regime_detector.update(event)

        # Check if regime changed
        if current_regime != self.active_regime.get(symbol, MarketRegime.UNKNOWN):
            # Record the regime change
            old_regime = self.active_regime.get(symbol, MarketRegime.UNKNOWN)
            self._record_regime_transition(symbol, old_regime, current_regime, timestamp)
            
            # Switch active strategy
            self._switch_strategy(symbol, current_regime, timestamp)

        # Get active strategy for current regime
        active_strategy = self._get_active_strategy(symbol)
        
        # Delegate to active strategy if available
        if active_strategy and hasattr(active_strategy, 'on_bar'):
            return active_strategy.on_bar(event)
        
        return None
    
    def _switch_strategy(self, symbol: str, regime: MarketRegime, timestamp=None):
        """
        Switch active strategy based on regime.
        
        Args:
            symbol: Symbol that experienced regime change
            regime: New regime for the symbol
            timestamp: Optional timestamp of the change
        """
        logger.info(f"Regime change for {symbol}: {self.active_regime.get(symbol, 'Unknown')} -> {regime.value}")
        
        # Update active regime
        self.active_regime[symbol] = regime
        
        # Get strategy for this regime
        strategy = self.strategies.get(regime)
        
        if strategy:
            logger.info(f"Switching to {strategy.__class__.__name__} for {symbol} in {regime.value} regime")
            
            # Record strategy switch
            self._record_strategy_switch(symbol, regime, strategy.__class__.__name__)
        else:
            # Fall back to default strategy
            default_strategy = self.strategies.get(MarketRegime.UNKNOWN)
            if default_strategy:
                logger.info(f"No strategy for {regime.value} regime, using default for {symbol}")
                
                # Record strategy switch to default
                self._record_strategy_switch(symbol, regime, 
                                           default_strategy.__class__.__name__ 
                                           if default_strategy else None)
            else:
                logger.warning(f"No strategy available for {regime.value} regime")
    
    def _get_active_strategy(self, symbol: str):
        """
        Get currently active strategy for a symbol.
        
        Args:
            symbol: Symbol to get strategy for
            
        Returns:
            Strategy instance or None
        """
        regime = self.active_regime.get(symbol, MarketRegime.UNKNOWN)
        strategy = self.strategies.get(regime)
        
        if not strategy:
            # Fall back to default strategy
            strategy = self.strategies.get(MarketRegime.UNKNOWN)
            
        return strategy
    
    def _record_regime_transition(self, symbol: str, old_regime: MarketRegime, new_regime: MarketRegime, timestamp=None):
        """
        Record a regime transition for analysis.
        
        Args:
            symbol: Symbol with regime change
            old_regime: Previous regime
            new_regime: New regime
            timestamp: Optional timestamp of transition
        """
        if timestamp is None:
            timestamp = datetime.datetime.now()
            
        transition = {
            'timestamp': timestamp,
            'symbol': symbol,
            'from_regime': old_regime.value if isinstance(old_regime, MarketRegime) else str(old_regime),
            'to_regime': new_regime.value
        }
        
        self.regime_transitions.append(transition)
    
    def _record_strategy_switch(self, symbol: str, regime: MarketRegime, strategy_name: str, timestamp=None):
        """
        Record a strategy switch for analysis.
        
        Args:
            symbol: Symbol with strategy switch
            regime: Regime causing the switch
            strategy_name: Name of new strategy
            timestamp: Optional timestamp of switch
        """
        if timestamp is None:
            timestamp = datetime.datetime.now()
            
        switch = {
            'timestamp': timestamp,
            'symbol': symbol,
            'regime': regime.value,
            'strategy': strategy_name
        }
        
        self.strategy_switches.append(switch)
    
    def _update_symbols(self):
        """Update symbols list based on all sub-strategies."""
        all_symbols = set()
        
        for strategy in self.strategies.values():
            if strategy and hasattr(strategy, 'symbols'):
                if isinstance(strategy.symbols, list):
                    all_symbols.update(strategy.symbols)
                elif hasattr(strategy.symbols, '__iter__'):
                    all_symbols.update(list(strategy.symbols))
                else:
                    # Single symbol
                    all_symbols.add(strategy.symbols)
        
        self.symbols = list(all_symbols)
        
        # Initialize active regime for all symbols
        for symbol in self.symbols:
            if symbol not in self.active_regime:
                self.active_regime[symbol] = MarketRegime.UNKNOWN
    
    def reset(self):
        """Reset the strategy, detector, and all sub-strategies."""
        # Reset regime detector
        self.regime_detector.reset()
        
        # Reset all sub-strategies
        for strategy in self.strategies.values():
            if strategy and hasattr(strategy, 'reset'):
                strategy.reset()
        
        # Reset local state
        self.active_regime = {symbol: MarketRegime.UNKNOWN for symbol in self.symbols}
        self.regime_transitions = []
        self.strategy_switches = []
    
    def get_regime_stats(self) -> Dict[str, Any]:
        """
        Get statistics on regime transitions and strategy switches.
        
        Returns:
            dict: Regime and strategy statistics
        """
        result = {
            'transitions': len(self.regime_transitions),
            'strategy_switches': len(self.strategy_switches),
            'current_regimes': {symbol: regime.value for symbol, regime in self.active_regime.items()},
            'transition_history': self.regime_transitions[-10:],  # Last 10 transitions
            'switch_history': self.strategy_switches[-10:]  # Last 10 strategy switches
        }
        
        # Calculate regime distribution
        regime_counts = {}
        for transition in self.regime_transitions:
            regime = transition['to_regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
        result['regime_distribution'] = regime_counts
        
        return result
