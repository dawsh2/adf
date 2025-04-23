import datetime
import pandas as pd

class EnsembleStrategy:
    def __init__(self, strategies, weights=None):
        self.strategies = strategies
        self.strategy_names = [s.name for s in strategies]
        self.weights = weights or {name: 1.0/len(strategies) for name in self.strategy_names}
        self.event_bus = None
        self.name = "ensemble_strategy"
        self.symbols = []
        
        # Extract symbols from all strategies
        for strategy in strategies:
            if hasattr(strategy, 'symbols'):
                if isinstance(strategy.symbols, list):
                    self.symbols.extend(strategy.symbols)
                else:
                    self.symbols.append(strategy.symbols)
        
        # Remove duplicates
        self.symbols = list(set(self.symbols))
    
    def set_event_bus(self, event_bus):
        """Set the event bus and propagate to child strategies."""
        self.event_bus = event_bus
        for strategy in self.strategies:
            if hasattr(strategy, 'set_event_bus'):
                strategy.set_event_bus(event_bus)
        return self
    
    def on_bar(self, event):
        """Process bar events and combine signals from all strategies."""
        signals = []
        for strategy, name in zip(self.strategies, self.strategy_names):
            if name in self.weights and self.weights[name] > 0:
                signal = strategy.on_bar(event)
                if signal:
                    signals.append((signal, self.weights[name]))
        
        # Combine signals if any were generated
        if signals:
            # Take the signal with highest weight for simplicity
            return max(signals, key=lambda x: x[1])[0]
        
        return None
    
    def get_parameters(self):
        """Get current parameters as a flat dictionary."""
        return {'weight_' + name: self.weights.get(name, 0.0) for name in self.strategy_names}
    
    def set_parameters(self, params):
        """Set parameters from a flat dictionary."""
        weights = {}
        for name in self.strategy_names:
            param_key = 'weight_' + name
            if param_key in params:
                weights[name] = params[param_key]
        
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            self.weights = {name: value/total for name, value in weights.items()}
        else:
            # Equal weights if sum is zero
            self.weights = {name: 1.0/len(self.strategy_names) for name in self.strategy_names}
    
    def reset(self):
        """Reset all child strategies."""
        for strategy in self.strategies:
            if hasattr(strategy, 'reset'):
                strategy.reset()



# src/strategy/regime_aware_strategy.py

import logging
from typing import Dict, Any, Optional, List, Union

from src.models.filters.regime.regime_detector import RegimeDetectorBase, MarketRegime

logger = logging.getLogger(__name__)

class RegimeAwareStrategy:
    """
    Strategy wrapper that adapts parameters based on market regime.
    
    This component wraps around a base strategy and switches its parameters
    dynamically based on the current market regime detected by a regime detector.
    """
    
    def __init__(self, base_strategy, regime_detector: RegimeDetectorBase, regime_parameters=None):
        """
        Initialize the regime-aware strategy wrapper.
        
        Args:
            base_strategy: Base strategy to enhance with regime awareness
            regime_detector: Regime detector instance
            regime_parameters: Dictionary mapping regimes to parameter sets
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
        self.regime_parameters = regime_parameters or {}
        
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
    
    def on_bar(self, event):
        """
        Process a bar event with regime-specific parameters.

        Args:
            event: Bar event to process

        Returns:
            Signal event or None
        """
        symbol = event.get_symbol()
        timestamp = event.get_timestamp()

        if symbol not in self.symbols:
            return None

        # Detect current regime
        current_regime = self.regime_detector.update(event)

        # Check if regime changed
        if current_regime != self.active_regime.get(symbol):
            # Record the regime change
            self._record_regime_transition(symbol, self.active_regime.get(symbol), current_regime, timestamp)
            
            # Switch parameters
            self._switch_parameters(symbol, current_regime, timestamp)

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
        logger.info(f"Regime change for {symbol}: {self.active_regime.get(symbol)} -> {regime.value}")
        
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
            'from_regime': old_regime.value if hasattr(old_regime, 'value') else str(old_regime),
            'to_regime': new_regime.value if hasattr(new_regime, 'value') else str(new_regime)
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
            import datetime
            timestamp = datetime.datetime.now()
            
        change = {
            'timestamp': timestamp,
            'symbol': symbol,
            'regime': regime.value if hasattr(regime, 'value') else str(regime),
            'parameters': parameters
        }
        
        self.parameter_changes.append(change)
    
    def get_parameters(self):
        """
        Get current strategy parameters.
        
        Returns:
            dict: Current parameter values
        """
        if hasattr(self.strategy, 'get_parameters'):
            return self.strategy.get_parameters()
        
        # If no get_parameters method, try to infer from attributes
        params = {}
        for regime, regime_params in self.regime_parameters.items():
            if regime_params:
                # Use first non-empty parameter set as a template
                for param_name in regime_params:
                    if hasattr(self.strategy, param_name):
                        params[param_name] = getattr(self.strategy, param_name)
                break
                
        return params
    
    def set_parameters(self, params):
        """
        Set base strategy parameters.
        
        Args:
            params: Parameters dictionary
        """
        if hasattr(self.strategy, 'set_parameters'):
            self.strategy.set_parameters(params)
        else:
            # If no set_parameters method, try direct attribute assignment
            for param_name, param_value in params.items():
                if hasattr(self.strategy, param_name):
                    setattr(self.strategy, param_name, param_value)
    
    def set_regime_parameters(self, regime: MarketRegime, parameters: Dict[str, Any]):
        """
        Set parameters for a specific regime.
        
        Args:
            regime: MarketRegime to set parameters for
            parameters: Parameter dict for this regime
        """
        self.regime_parameters[regime] = parameters
        
        # If this is the current regime of any symbol, apply immediately
        for symbol in self.symbols:
            if self.active_regime.get(symbol) == regime:
                logger.info(f"Applying new {regime.value} parameters to {symbol}: {parameters}")
                
                if hasattr(self.strategy, 'set_parameters'):
                    self.strategy.set_parameters(parameters)
                else:
                    # If no set_parameters method, try direct attribute assignment
                    for param_name, param_value in parameters.items():
                        if hasattr(self.strategy, param_name):
                            setattr(self.strategy, param_name, param_value)
    
    def reset(self):
        """Reset the strategy and detector."""
        # Reset base strategy
        if hasattr(self.strategy, 'reset'):
            self.strategy.reset()
            
        # Reset regime detector
        if hasattr(self.regime_detector, 'reset'):
            self.regime_detector.reset()
        
        # Reset local state
        self.active_regime = {symbol: MarketRegime.UNKNOWN for symbol in self.symbols}
        self.regime_transitions = []
        self.parameter_changes = []                
