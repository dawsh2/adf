# src/models/filters/filter_base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from .result import FilterResult

class FilterBase(ABC):
    """
    Base class for all filters.
    Provides common interface and functionality.
    """
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.state = {}  # For storing filter state
    
    @abstractmethod
    def evaluate(self, context: Dict[str, Any], 
                params: Optional[Dict[str, Any]] = None) -> FilterResult:
        """
        Evaluate filter condition based on context.
        
        Args:
            context: Data context to evaluate against
            params: Optional filter parameters
            
        Returns:
            FilterResult: Result of filter evaluation
        """
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get filter parameters for optimization.
        
        Returns:
            Dictionary of parameter names to values
        """
        # Default implementation - override in subclasses
        return {}
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set filter parameters.
        
        Args:
            params: Dictionary of parameter names to values
        """
        # Default implementation - override in subclasses
        pass
    
    def reset(self) -> None:
        """
        Reset filter state.
        """
        self.state = {}


class FilterCombiner(ABC):
    """
    Interface for combining multiple filter results.
    """
    
    @abstractmethod
    def combine(self, results: List[FilterResult], 
               names: Optional[List[str]] = None) -> FilterResult:
        """
        Combine multiple filter results.
        
        Args:
            results: List of FilterResult objects
            names: Optional list of filter names
            
        Returns:
            FilterResult: Combined result
        """
        pass


class RegimeFilterBase(FilterBase):
    """Abstract base class for regime-based filters."""
    
    component_type = "filters.regime"  # More specific configuration section
    
    @classmethod
    def default_params(cls):
        """Get default parameters for regime filters."""
        return {
            'lookback': 63,  # ~3 months of trading days
            'warmup_period': 126,  # Period needed before regime detection is reliable
        }
    
    @abstractmethod
    def detect_regime(self, data):
        """
        Detect the current market regime.
        
        Args:
            data: Dictionary or DataFrame with market data
            
        Returns:
            str: Identified regime (e.g., 'bullish', 'bearish', 'sideways', 'volatile')
        """
        pass
    
    def evaluate(self, data):
        """Determine if current regime passes filter criteria."""
        # Check if we have enough data
        history = data.get('history', [])
        if len(history) < self.params.get('warmup_period', 126):
            return True  # Default to active during warmup period
        
        # Detect regime
        regime = self.detect_regime(data)
        
        # Store detected regime in state
        data['current_regime'] = regime
        
        # Check if this regime is in allowed regimes
        allowed_regimes = self.params.get('allowed_regimes', [])
        if allowed_regimes and regime not in allowed_regimes:
            return False
            
        return True
        



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


class FilterResult:
    """
    Result of a filter evaluation.
    Contains the result, reason, and additional metadata.
    """
    
    def __init__(self, passed: bool, reason: Optional[str] = None, 
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize filter result.
        
        Args:
            passed: Whether the filter passed
            reason: Optional explanation for result
            metadata: Optional additional data
        """
        self.passed = passed
        self.reason = reason or ("Passed" if passed else "Failed")
        self.metadata = metadata or {}
    
    def __bool__(self) -> bool:
        """
        Allow direct boolean evaluation.
        
        Returns:
            True if filter passed, False otherwise
        """
        return self.passed
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary.
        
        Returns:
            Dictionary representation of filter result
        """
        return {
            'passed': self.passed,
            'reason': self.reason,
            'metadata': self.metadata
        }    
