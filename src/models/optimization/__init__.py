"""
Optimization module initialization with validation components.
"""

# Import core classes
from .interfaces import OptimizationTarget
from .result import OptimizationResult
from .component_optimizer import (
    ComponentOptimizer,
    GridSearchOptimizer, 
    RandomSearchOptimizer,
    GeneticOptimizer,
    WalkForwardOptimizer,
    RegimeBasedOptimizer
)
from .manager import (
    OptimizationManager,
    RegimeAwareOptimizationManager,
    evaluate_backtest,
    create_optimization_manager
)

# Import the validation class from the validation package
from .validation import OptimizationValidator

# Export all components
__all__ = [
    # Base interfaces
    'OptimizationTarget',
    'OptimizationResult',
    
    # Optimizers
    'ComponentOptimizer',
    'GridSearchOptimizer',
    'RandomSearchOptimizer',
    'GeneticOptimizer',
    'WalkForwardOptimizer',
    'RegimeBasedOptimizer',
    
    # Managers
    'OptimizationManager',
    'RegimeAwareOptimizationManager',
    'evaluate_backtest',
    'create_optimization_manager',
    
    # Validation
    'OptimizationValidator'
]
