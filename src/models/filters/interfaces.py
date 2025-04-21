# src/models/filters/filter_base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from .result import FilterResult


class FilterTarget(ABC):
    """
    Interface for components that can be filtered.
    Any component that needs filtering should implement this.
    """
    
    @abstractmethod
    def get_filter_context(self, symbol: Optional[str] = None, 
                         timestamp: Optional[Any] = None) -> Dict[str, Any]:
        """
        Get context data needed for filter evaluation.
        
        Args:
            symbol: Optional symbol to get context for
            timestamp: Optional timestamp
            
        Returns:
            Dictionary of context data
        """
        pass
    
    @abstractmethod
    def apply_filter_result(self, result: 'FilterResult', 
                          symbol: Optional[str] = None,
                          timestamp: Optional[Any] = None) -> None:
        """
        Apply filter result to this component.
        
        Args:
            result: Filter result to apply
            symbol: Optional symbol
            timestamp: Optional timestamp
        """
        pass
