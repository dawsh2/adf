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
