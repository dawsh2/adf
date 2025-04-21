# src/models/optimization/interfaces.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Tuple, Union

class OptimizationTarget(ABC):
    """
    Interface for components that can be optimized.
    Any component that needs parameter optimization should implement this.
    """
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current parameter dict for optimization.
        
        Returns:
            Dict of parameter names to current values
        """
        pass
    
    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set parameters from optimization result.
        
        Args:
            params: Dictionary of parameter names to values
        """
        pass
    
    @abstractmethod
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate if parameters are valid for this target.
        
        Args:
            params: Dictionary of parameter names to values
            
        Returns:
            True if parameters are valid, False otherwise
        """
        pass
