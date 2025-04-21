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




class OptimizerBase(ABC):
    """
    Base class for all optimization methods.
    """
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.best_result = None
    
    @abstractmethod
    def optimize(self, param_space: Dict[str, List[Any]], 
                fitness_function: Callable[[Dict[str, Any]], float],
                constraints: Optional[List[Callable[[Dict[str, Any]], bool]]] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Perform optimization search.
        
        Args:
            param_space: Dictionary mapping parameter names to possible values
            fitness_function: Function that evaluates parameter sets
            constraints: Optional list of constraint functions
            **kwargs: Additional optimizer-specific parameters
            
        Returns:
            Dictionary containing optimization results
        """
        pass
    
    def get_best_result(self) -> Optional[Dict[str, Any]]:
        """
        Get the best result found during optimization.
        
        Returns:
            Dictionary with best parameters and score, or None if not optimized
        """
        return self.best_result    


# src/models/optimization/result.py
from typing import Dict, Any, List, Optional
import datetime

class OptimizationResult:
    """
    Container for optimization results.
    Standardizes result format across different optimization methods.
    """
    
    def __init__(self, best_params: Dict[str, Any], best_score: float,
                optimizer_name: str, all_results: Optional[List[Dict[str, Any]]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        self.best_params = best_params
        self.best_score = best_score
        self.optimizer_name = optimizer_name
        self.all_results = all_results or []
        self.metadata = metadata or {}
        self.timestamp = datetime.datetime.now()
    
    def get_top_n(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the top N parameter sets.
        
        Args:
            n: Number of top results to return
            
        Returns:
            List of top parameter sets with scores
        """
        if not self.all_results:
            return [{'params': self.best_params, 'score': self.best_score}]
            
        # Sort by score (descending)
        sorted_results = sorted(self.all_results, 
                               key=lambda x: x.get('score', float('-inf')),
                               reverse=True)
                               
        return sorted_results[:n]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary.
        
        Returns:
            Dictionary representation of optimization result
        """
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimizer': self.optimizer_name,
            'timestamp': self.timestamp.isoformat(),
            'top_results': self.get_top_n(5),
            'metadata': self.metadata
        }    
