"""
Strategy Manager implementation.

This component coordinates signals from multiple strategies,
combines them, and applies risk management rules.
"""
import logging
from typing import Dict, List, Any, Optional, Set, Union

from core.events.event_types import Event, EventType, BarEvent, SignalEvent
from core.events.event_utils import create_signal_event
# Import StrategyBase from the correct location
from models.components.base import StrategyBase

logger = logging.getLogger(__name__)

class StrategyManager:
    """
    Manager for coordinating multiple trading strategies.
    
    This component can:
    1. Route market data to multiple strategies
    2. Combine signals from different strategies
    3. Apply global risk management rules
    4. Handle strategy weighting and allocation
    """
    
    def __init__(self, name: str = "strategy_manager", event_bus = None):
        """
        Initialize the strategy manager.
        
        Args:
            name: Manager name
            event_bus: Event bus for emitting signals
        """
        self.name = name
        self.event_bus = event_bus
        self.strategies = {}  # name -> StrategyBase
        self.weights = {}     # name -> weight (0.0 to 1.0)
        self.active = {}      # name -> active status (boolean)
        self.signals = {}     # symbol -> dict of strategy signals
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
        
        # Update bus for all strategies
        for strategy in self.strategies.values():
            strategy.set_event_bus(event_bus)
    
    def add_strategy(self, strategy: StrategyBase, weight: float = 1.0, active: bool = True):
        """
        Add a strategy to the manager.
        
        Args:
            strategy: StrategyBase instance
            weight: Strategy weight (0.0 to 1.0)
            active: Whether strategy is active
        """
        if not isinstance(strategy, StrategyBase):
            raise TypeError("Strategy must be instance of StrategyBase")
            
        name = strategy.name
        
        # Set event bus
        if self.event_bus and not strategy.event_bus:
            strategy.set_event_bus(self.event_bus)
            
        # Add to collections
        self.strategies[name] = strategy
        self.weights[name] = max(0.0, min(1.0, weight))  # Clamp to 0-1
        self.active[name] = active
    
    def remove_strategy(self, name: str):
        """
        Remove a strategy from the manager.
        
        Args:
            name: Strategy name
        """
        if name in self.strategies:
            del self.strategies[name]
            del self.weights[name]
            del self.active[name]
    
    def set_strategy_weight(self, name: str, weight: float):
        """
        Set weight for a strategy.
        
        Args:
            name: Strategy name
            weight: Strategy weight (0.0 to 1.0)
        """
        if name in self.weights:
            self.weights[name] = max(0.0, min(1.0, weight))  # Clamp to 0-1
    
    def set_strategy_active(self, name: str, active: bool):
        """
        Set active status for a strategy.
        
        Args:
            name: Strategy name
            active: Whether strategy is active
        """
        if name in self.active:
            self.active[name] = active
    
    def on_bar(self, event: BarEvent):
        """
        Process a bar event with all active strategies.
        
        Args:
            event: BarEvent to process
        """
        if not isinstance(event, BarEvent):
            return
            
        symbol = event.get_symbol()
        
        # Initialize signal storage for symbol if needed
        if symbol not in self.signals:
            self.signals[symbol] = {}
            
        # Process event with all active strategies
        for name, strategy in self.strategies.items():
            if not self.active[name]:
                continue
                
            # Process bar with strategy
            signal = strategy.on_bar(event)
            
            # Store signal if generated
            if signal:
                self.signals[symbol][name] = signal
                
        # Combine signals and emit
        combined_signal = self._combine_signals(symbol)
        if combined_signal and self.event_bus:
            self.event_bus.emit(combined_signal)
    
    def _combine_signals(self, symbol: str) -> Optional[SignalEvent]:
        """
        Combine signals from all strategies for a symbol.
        
        This implementation uses a weighted vote approach.
        
        Args:
            symbol: Symbol to combine signals for
            
        Returns:
            Optional[SignalEvent]: Combined signal or None
        """
        if symbol not in self.signals or not self.signals[symbol]:
            return None
            
        # Count weighted votes for each signal type
        votes = {
            SignalEvent.BUY: 0.0,
            SignalEvent.SELL: 0.0,
            SignalEvent.NEUTRAL: 0.0
        }
        
        total_weight = 0.0
        latest_timestamp = None
        price = None
        
        # Collect votes from all strategies
        for name, signal in self.signals[symbol].items():
            if name not in self.weights or not self.active[name]:
                continue
                
            weight = self.weights[name]
            signal_value = signal.get_signal_value()
            confidence = signal.data.get('confidence', 1.0)
            
            # Add weighted vote
            votes[signal_value] += weight * confidence
            total_weight += weight
            
            # Track latest timestamp and price
            if not latest_timestamp or signal.get_timestamp() > latest_timestamp:
                latest_timestamp = signal.get_timestamp()
                price = signal.get_price()
        
        # No valid votes
        if total_weight <= 0.0 or not price:
            return None
            
        # Normalize votes
        for signal_type in votes:
            votes[signal_type] /= total_weight
            
        # Find winning signal
        max_vote = max(votes.values())
        if max_vote < 0.6:  # Require at least 60% agreement
            return None
            
        winning_signals = [s for s, v in votes.items() if v == max_vote]
        if len(winning_signals) != 1:
            return None  # Tie
            
        winning_signal = winning_signals[0]
        
        # Create combined signal
        return create_signal_event(
            signal_value=winning_signal,
            price=price,
            symbol=symbol,
            rule_id=self.name,
            confidence=max_vote,
            metadata={
                'votes': votes,
                'strategies': list(self.signals[symbol].keys())
            },
            timestamp=latest_timestamp
        )
    
    def reset(self):
        """Reset the manager and all strategies."""
        self.signals = {}
        
        # Reset all strategies
        for strategy in self.strategies.values():
            strategy.reset()
