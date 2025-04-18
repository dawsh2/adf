"""
Abstract base class for all data handlers in the trading system.
"""
from abc import ABC, abstractmethod
import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

from src.core.events.event_bus import EventBus
from src.core.events.event_types import BarEvent

class DataHandlerBase(ABC):
    """Abstract base class for data handlers."""
    
    def __init__(self, event_bus=None):
        """
        Initialize the data handler.
        
        Args:
            event_bus: Optional event bus for emitting events
        """
        self.event_bus = event_bus
    
    def set_event_bus(self, event_bus: EventBus) -> None:
        """
        Set the event bus.
        
        Args:
            event_bus: Event bus for emitting events
        """
        self.event_bus = event_bus
    
    @abstractmethod
    def load_data(self, symbols: Union[str, List[str]], start_date=None, end_date=None, timeframe='1d') -> None:
        """
        Load data for the specified symbols.
        
        Args:
            symbols: Symbol or list of symbols
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe (e.g., '1d', '1h', '5m')
        """
        pass
    
    @abstractmethod
    def get_next_bar(self, symbol: str) -> Optional[BarEvent]:
        """
        Get the next bar for a symbol.
        
        Args:
            symbol: Symbol to get bar for
            
        Returns:
            BarEvent or None if no more bars
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the data handler state."""
        pass
    
    def get_symbols(self) -> List[str]:
        """
        Get the list of available symbols.
        
        Returns:
            List of symbols
        """
        return []
    
    def get_latest_bar(self, symbol: str) -> Optional[BarEvent]:
        """
        Get the latest bar for a symbol.
        
        Args:
            symbol: Symbol to get bar for
            
        Returns:
            The latest bar or None if not available
        """
        return None
    
    def get_latest_bars(self, symbol: str, N: int = 1) -> List[BarEvent]:
        """
        Get the last N bars for a symbol.
        
        Args:
            symbol: Symbol to get bars for
            N: Number of bars to retrieve
            
        Returns:
            List of bars (empty if none available)
        """
        return []
