"""
Optimization Manager Module

This module provides a central manager for component optimization workflows,
coordinating different optimization techniques, components, and evaluation metrics.
"""

import logging
import datetime
from typing import Dict, List, Any, Optional, Callable, Union, Type

from src.models.filters.regime.regime_detector import RegimeDetectorBase, MarketRegime
from src.models.optimization.component_optimizer import ComponentOptimizer, GridSearchOptimizer
from src.execution.backtest.backtest import run_backtest


logger = logging.getLogger(__name__)


class OptimizationManager:
    """
    Central coordinator for optimization processes.
    
    This class manages component registration, optimization methods,
    evaluation metrics, and executes optimization workflows.
    """
    
    def __init__(self, name: str = "optimization_manager"):
        """
        Initialize the optimization manager.
        
        Args:
            name: Manager name
        """
        self.name = name
        
        # Component registry
        self.targets = {}  # name -> Component
        self.optimizers = {}  # name -> ComponentOptimizer
        self.evaluators = {}  # name -> evaluation function
        
        # Results storage
        self.results = {}  # key -> optimization result
    
    def register_target(self, name: str, target) -> 'OptimizationManager':
        """
        Register an optimization target.
        
        Args:
            name: Target name
            target: Component to optimize
            
        Returns:
            self: For method chaining
        """
        # Validate that target has required methods
        if not hasattr(target, 'get_parameters') or not callable(getattr(target, 'get_parameters')):
            raise ValueError(f"Target {name} must have get_parameters() method")
            
        if not hasattr(target, 'set_parameters') or not callable(getattr(target, 'set_parameters')):
            raise ValueError(f"Target {name} must have set_parameters() method")
        
        self.targets[name] = target
        return self
    
    def register_optimizer(self, name: str, optimizer: ComponentOptimizer) -> 'OptimizationManager':
        """
        Register an optimization method.
        
        Args:
            name: Optimizer name
            optimizer: ComponentOptimizer instance
            
        Returns:
            self: For method chaining
        """
        if not isinstance(optimizer, ComponentOptimizer):
            raise ValueError(f"Optimizer {name} must be a ComponentOptimizer instance")
            
        self.optimizers[name] = optimizer
        return self
    
    def register_evaluator(self, name: str, evaluator_func: Callable) -> 'OptimizationManager':
        """
        Register an evaluation function.
        
        Args:
            name: Evaluator name
            evaluator_func: Function that evaluates component performance
            
        Returns:
            self: For method chaining
        """
        if not callable(evaluator_func):
            raise ValueError(f"Evaluator {name} must be callable")
            
        self.evaluators[name] = evaluator_func
        return self
    
    def optimize_component(self, target_name: str, optimizer_name: str, 
                         evaluator_name: str, param_space: Optional[Dict[str, Any]] = None, 
                         **kwargs) -> Dict[str, Any]:
        """
        Optimize a single component.
        
        Args:
            target_name: Name of component to optimize
            optimizer_name: Name of optimizer to use
            evaluator_name: Name of evaluator to use
            param_space: Optional parameter space (if None, uses target's parameters)
            **kwargs: Additional parameters for optimization
            
        Returns:
            dict: Optimization result
        """
        # Validate inputs
        if target_name not in self.targets:
            raise ValueError(f"Unknown target: {target_name}")
            
        if optimizer_name not in self.optimizers:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
        if evaluator_name not in self.evaluators:
            raise ValueError(f"Unknown evaluator: {evaluator_name}")
        
        # Resolve target, optimizer, and evaluator
        target = self._resolve_target(target_name)
        optimizer = self.optimizers[optimizer_name]
        evaluator = self.evaluators[evaluator_name]
        
        # Get parameter space if not provided
        if param_space is None:
            param_space = target.get_parameters()
            
        # Log optimization setup
        logger.info(f"Optimizing {target_name} using {optimizer_name} for {evaluator_name}")
        logger.info(f"Parameter space: {param_space}")
        
        # Create evaluation wrapper
        def evaluation_wrapper(comp, **eval_kwargs):
            return evaluator(comp, **eval_kwargs)
        
        # Run optimization
        result = optimizer.optimize(
            component=target,
            param_space=param_space,
            evaluation_function=evaluation_wrapper,
            **kwargs
        )
        
        # Store result
        result_key = f"{target_name}_{optimizer_name}_{evaluator_name}"
        self.results[result_key] = result
        
        # Log optimization result
        logger.info(f"Optimization complete for {target_name}")
        logger.info(f"Best parameters: {result.get('best_params')}")
        logger.info(f"Best score: {result.get('best_score')}")
        
        return result
    
    def optimize_multiple(self, targets: List[str], optimizer_name: str, 
                        evaluator_name: str, **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Optimize multiple components with the same optimizer and evaluator.
        
        Args:
            targets: List of target names to optimize
            optimizer_name: Name of optimizer to use
            evaluator_name: Name of evaluator to use
            **kwargs: Additional parameters for optimization
            
        Returns:
            dict: Mapping from target name to optimization result
        """
        results = {}
        
        for target_name in targets:
            logger.info(f"Optimizing {target_name}...")
            result = self.optimize_component(
                target_name=target_name,
                optimizer_name=optimizer_name,
                evaluator_name=evaluator_name,
                **kwargs
            )
            results[target_name] = result
        
        return results
    
    def optimize_hierarchical(self, base_target: str, sub_targets: List[str],
                           optimizer_name: str, evaluator_name: str,
                           sequential: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Optimize a hierarchical component structure.
        
        Args:
            base_target: Name of parent component
            sub_targets: List of sub-component paths
            optimizer_name: Name of optimizer to use
            evaluator_name: Name of evaluator to use
            sequential: Whether to optimize sequentially
            **kwargs: Additional parameters for optimization
            
        Returns:
            dict: Optimization results
        """
        results = {}
        
        # Check if base target exists
        if base_target not in self.targets:
            raise ValueError(f"Unknown base target: {base_target}")
            
        # Optimize sub-components first if sequential
        if sequential:
            for sub_target in sub_targets:
                target_path = f"{base_target}.{sub_target}"
                try:
                    result = self.optimize_component(
                        target_name=target_path,
                        optimizer_name=optimizer_name,
                        evaluator_name=evaluator_name,
                        **kwargs
                    )
                    results[sub_target] = result
                except ValueError as e:
                    logger.error(f"Error optimizing {target_path}: {e}")
        
        # Then optimize the base component
        base_result = self.optimize_component(
            target_name=base_target,
            optimizer_name=optimizer_name,
            evaluator_name=evaluator_name,
            **kwargs
        )
        results['base'] = base_result
        
        return results
    
    def create_ensemble(self, target_names: List[str], ensemble_name: str, 
                      weights: Optional[Dict[str, float]] = None, **kwargs) -> Any:
        """
        Create an ensemble from multiple optimized components.
        
        Args:
            target_names: Names of components to combine
            ensemble_name: Name for the ensemble
            weights: Optional mapping from target names to weights
            **kwargs: Additional parameters for ensemble creation
            
        Returns:
            Ensemble component
        """
        from src.strategy.regime_aware_strategy import EnsembleStrategy
        
        # Resolve targets
        targets = [self._resolve_target(name) for name in target_names]
        
        # Create ensemble
        ensemble = EnsembleStrategy(
            #            name=ensemble_name,
            strategies=targets,
            weights=weights
        )
        
        # Register ensemble as target
        self.register_target(ensemble_name, ensemble)
        
        return ensemble
    
    def _resolve_target(self, target_path: str):
        """
        Resolve a target path to a component.
        
        Handles nested components with dot notation (e.g., "strategy.rules.ma_crossover").
        
        Args:
            target_path: Component path
            
        Returns:
            Component instance
        """
        if '.' not in target_path:
            if target_path not in self.targets:
                raise ValueError(f"Unknown target: {target_path}")
            return self.targets[target_path]
        
        # Handle nested path
        parts = target_path.split('.')
        root_name = parts[0]
        
        if root_name not in self.targets:
            raise ValueError(f"Unknown root target: {root_name}")
            
        component = self.targets[root_name]
        
        # Navigate through the nested structure
        for part in parts[1:]:
            if hasattr(component, part):
                component = getattr(component, part)
            elif hasattr(component, 'components') and part in component.components:
                component = component.components[part]
            elif hasattr(component, 'get_component') and callable(getattr(component, 'get_component')):
                try:
                    component = component.get_component(part)
                except:
                    raise ValueError(f"Cannot get component: {part} from {component}")
            else:
                raise ValueError(f"Cannot find component: {part} in {target_path}")
                
        return component
    
    def get_result(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a stored optimization result.
        
        Args:
            key: Result key
            
        Returns:
            dict: Optimization result or None if not found
        """
        return self.results.get(key)
    
    def get_all_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all optimization results.
        
        Returns:
            dict: All optimization results
        """
        return self.results.copy()
    
    def clear_results(self) -> None:
        """Clear all stored results."""
        self.results = {}


class RegimeAwareOptimizationManager(OptimizationManager):
    """
    Extension of OptimizationManager that integrates regime detection.
    
    This class enhances the standard optimization manager with
    regime-specific optimization capabilities.
    """
    
    def __init__(self, regime_detector: RegimeDetectorBase, name: str = "regime_optimization_manager"):
        """
        Initialize the regime-aware optimization manager.
        
        Args:
            regime_detector: RegimeDetectorBase instance for regime detection
            name: Manager name
        """
        super().__init__(name)
        self.regime_detector = regime_detector
    
    def optimize_for_regimes(self, target_name: str, optimizer_name: str, 
                           evaluator_name: str, **kwargs) -> Dict[str, Any]:
        """
        Optimize a component separately for each market regime.
        
        Args:
            target_name: Name of component to optimize
            optimizer_name: Name of optimizer to use
            evaluator_name: Name of evaluator to use
            **kwargs: Additional parameters including:
                - data_handler: Required data handler with loaded data
                - param_space: Optional parameter space
                - min_regime_bars: Minimum bars required for regime optimization
                
        Returns:
            dict: Regime-specific optimization results
        """
        # Import here to avoid circular imports
        from src.models.optimization.component_optimizer import RegimeBasedOptimizer
        
        # Validate inputs
        if target_name not in self.targets:
            raise ValueError(f"Unknown target: {target_name}")
            
        if optimizer_name not in self.optimizers:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
        if evaluator_name not in self.evaluators:
            raise ValueError(f"Unknown evaluator: {evaluator_name}")
        
        # Get components
        target = self._resolve_target(target_name)
        base_optimizer = self.optimizers[optimizer_name]
        evaluator = self.evaluators[evaluator_name]
        
        # Create regime-based optimizer
        regime_optimizer = RegimeBasedOptimizer(
            regime_detector=self.regime_detector,
            base_optimizer=base_optimizer
        )
        
        # Get parameter space if provided
        param_space = kwargs.pop('param_space', None)
        if param_space is None:
            #            param_space = target.get_parameters()
            param_space = component.get_parameters()
        
        # Create evaluation wrapper
        def evaluation_wrapper(comp, **eval_kwargs):
            return evaluator(comp, **eval_kwargs)
        
        # Run regime-specific optimization
        result = regime_optimizer.optimize(
            component=target,
            param_space=param_space,
            evaluation_function=evaluation_wrapper,
            **kwargs
        )
        
        # Store result
        result_key = f"{target_name}_{optimizer_name}_{evaluator_name}_regimes"
        self.results[result_key] = result
        
        return result
    
    def create_regime_aware_strategy(self, base_target_name: str, regime_result_key: Optional[str] = None) -> Any:
        """
        Create a regime-aware strategy from optimization results.
        
        Args:
            base_target_name: Name of base strategy to enhance
            regime_result_key: Key for regime optimization result to use
                (if None, uses the most recent regime optimization for this target)
            
        Returns:
            RegimeAwareStrategy instance
        """
        from src.strategy.regime_aware_strategy import RegimeAwareStrategy
        
        # Get base strategy
        base_strategy = self._resolve_target(base_target_name)
        
        # Find regime result to use
        if regime_result_key is None:
            # Find most recent regime result for this target
            matching_keys = [
                key for key in self.results.keys() 
                if key.startswith(f"{base_target_name}_") and key.endswith("_regimes")
            ]
            
            if not matching_keys:
                raise ValueError(f"No regime optimization results found for {base_target_name}")
                
            # Use most recent result (assuming keys are added chronologically)
            regime_result_key = matching_keys[-1]
        
        # Get regime parameters
        regime_result = self.results.get(regime_result_key)
        if not regime_result:
            raise ValueError(f"Regime optimization result not found: {regime_result_key}")
            
        regime_parameters = regime_result.get('regime_parameters', {})
        
        # Create regime-aware strategy
        regime_strategy = RegimeAwareStrategy(
            base_strategy=base_strategy,
            regime_detector=self.regime_detector,
            regime_parameters=regime_parameters
        )
        
        # Register as new target
        strategy_name = f"regime_aware_{base_target_name}"
        self.register_target(strategy_name, regime_strategy)
        
        return regime_strategy


# Sample evaluation functions for common metric
def evaluate_backtest(component, data_handler, start_date=None, end_date=None, 
                   metric='sharpe_ratio', **kwargs):
    """
    Run a backtest and evaluate a specific performance metric.
    
    Args:
        component: Component to evaluate
        data_handler: Data handler with market data
        start_date: Optional start date for backtest
        end_date: Optional end date for backtest
        metric: Name of metric to evaluate
        **kwargs: Additional parameters for backtest
        
    Returns:
        float: Metric value (higher is better)
    """
    # Run backtest to get equity curve and trades
    from src.models.optimization.component_optimizer import run_backtest
    equity_curve, trade_count = run_backtest(
        component=component,
        data_handler=data_handler,
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )
    
    # Calculate performance metrics based only on equity curve
    from src.analytics.performance import PerformanceCalculator
    calculator = PerformanceCalculator()
    
    # Modify this to avoid using trades for calculations
    metrics = calculator.calculate_from_equity(equity_curve)
    
    # Return specified metric, or 0 if not found
    metric_value = metrics.get(metric, 0.0)
    
    # Special handling for metrics where lower is better
    if metric in ['max_drawdown', 'volatility', 'downside_deviation']:
        return -metric_value  # Negate so higher is better
        
    return metric_value


# Factory function to create optimization manager

def create_optimization_manager(regime_aware: bool = False, **kwargs):
    """
    Factory function to create an appropriate optimization manager.
    
    Args:
        regime_aware: Whether to create a regime-aware manager
        **kwargs: Additional parameters for manager creation
        
    Returns:
        OptimizationManager instance
    """
    if regime_aware:
        # Create regime detector if not provided
        if 'regime_detector' not in kwargs:
            from src.models.filters.regime.detector_factory import RegimeDetectorFactory
            detector_type = kwargs.pop('detector_type', 'enhanced')
            kwargs['regime_detector'] = RegimeDetectorFactory.create_detector(detector_type)
            
        # Create regime-aware manager
        manager = RegimeAwareOptimizationManager(**kwargs)
    else:
        # Create standard manager
        manager = OptimizationManager(**kwargs)
    
    # Register default optimizers if not disabled
    if kwargs.get('register_defaults', True):
        from src.models.optimization.component_optimizer import (
            GridSearchOptimizer, RandomSearchOptimizer,
            GeneticOptimizer, WalkForwardOptimizer
        )
        
        manager.register_optimizer('grid', GridSearchOptimizer())
        manager.register_optimizer('random', RandomSearchOptimizer())
        manager.register_optimizer('genetic', GeneticOptimizer())
        manager.register_optimizer('walk_forward', WalkForwardOptimizer())
        
        # Register common evaluation functions
        manager.register_evaluator('sharpe_ratio', 
                                 lambda comp, **kwargs: evaluate_backtest(comp, metric='sharpe_ratio', **kwargs))
        manager.register_evaluator('total_return', 
                                 lambda comp, **kwargs: evaluate_backtest(comp, metric='total_return', **kwargs))
        manager.register_evaluator('win_rate', 
                                 lambda comp, **kwargs: evaluate_backtest(comp, metric='win_rate', **kwargs))
        manager.register_evaluator('drawdown', 
                                 lambda comp, **kwargs: evaluate_backtest(comp, metric='max_drawdown', **kwargs))
    
    return manager


# Export components
__all__ = [
    'OptimizationManager',
    'RegimeAwareOptimizationManager',
    'evaluate_backtest',
    'create_optimization_manager'
]
