# src/models/optimization/manager.py
from typing import Dict, Any, Optional, List, Callable, Type
import logging

from .optimizer_base import OptimizerBase
from .interfaces import OptimizationTarget
from .result import OptimizationResult

logger = logging.getLogger(__name__)

class OptimizationManager:
    """
    Manager for coordinating optimization processes.
    """
    
    def __init__(self, name: str = "optimization_manager"):
        self.name = name
        self.targets = {}  # name -> OptimizationTarget
        self.optimizers = {}  # name -> OptimizerBase
        self.sequences = {}  # name -> optimization sequence function
        self.evaluators = {}  # name -> evaluation function
        self.results = {}  # Store optimization results
        self.event_bus = None
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
        return self
    
    def register_target(self, name: str, target: OptimizationTarget) -> None:
        """
        Register an optimization target.
        
        Args:
            name: Target name
            target: OptimizationTarget instance
        """
        if not isinstance(target, OptimizationTarget):
            raise TypeError("Target must implement OptimizationTarget interface")
        self.targets[name] = target
    
    def register_optimizer(self, name: str, optimizer: OptimizerBase) -> None:
        """
        Register an optimizer.
        
        Args:
            name: Optimizer name
            optimizer: OptimizerBase instance
        """
        if not isinstance(optimizer, OptimizerBase):
            raise TypeError("Optimizer must be an OptimizerBase instance")
        self.optimizers[name] = optimizer
    
    def register_sequence(self, name: str, sequence_func: Callable) -> None:
        """
        Register an optimization sequence function.
        
        Args:
            name: Sequence name
            sequence_func: Function implementing optimization sequence
        """
        self.sequences[name] = sequence_func
    
    def register_evaluator(self, name: str, evaluator_func: Callable) -> None:
        """
        Register an evaluation function.
        
        Args:
            name: Evaluator name
            evaluator_func: Function that evaluates performance
        """
        self.evaluators[name] = evaluator_func
    
    def run_optimization(self, sequence_name: str, optimizer_name: str, 
                       target_names: List[str], evaluator_name: str,
                       **kwargs) -> Any:
        """
        Run optimization according to specified sequence.
        
        Args:
            sequence_name: Name of sequence to use
            optimizer_name: Name of optimizer to use
            target_names: List of target names to optimize
            evaluator_name: Name of evaluator to use
            **kwargs: Additional parameters
            
        Returns:
            Optimization results
        """
        # Validate inputs
        if sequence_name not in self.sequences:
            raise ValueError(f"Unknown sequence: {sequence_name}")
            
        if optimizer_name not in self.optimizers:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
        for target_name in target_names:
            if target_name not in self.targets:
                raise ValueError(f"Unknown target: {target_name}")
                
        if evaluator_name not in self.evaluators:
            raise ValueError(f"Unknown evaluator: {evaluator_name}")
        
        # Prepare components for sequence
        optimizer = self.optimizers[optimizer_name]
        targets = {name: self.targets[name] for name in target_names}
        evaluator = self.evaluators[evaluator_name]
        sequence = self.sequences[sequence_name]
        
        # Run the optimization sequence
        try:
            result = sequence(self, optimizer, targets, evaluator, **kwargs)
            
            # Store result
            result_key = f"{sequence_name}_{optimizer_name}_{'-'.join(target_names)}"
            self.results[result_key] = result
            
            # Emit event if event bus available
            if self.event_bus:
                self._emit_optimization_event(result_key, result)
                
            return result
        except Exception as e:
            logger.error(f"Error in optimization sequence {sequence_name}: {e}")
            raise
    
    def get_result(self, key: str) -> Optional[Any]:
        """
        Get optimization result by key.
        
        Args:
            key: Result key
            
        Returns:
            Optimization result or None if not found
        """
        return self.results.get(key)
    
    def _emit_optimization_event(self, key: str, result: Any) -> None:
        """
        Emit optimization result event.
        
        Args:
            key: Result key
            result: Optimization result
        """
        # Implementation depends on event system
        pass
