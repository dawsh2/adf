"""
Component Optimizer Module

This module provides abstract base classes and implementations for component optimization.
The modular design allows for different optimization techniques to be used with any
component that implements the required parameter access interface.
"""

import logging
import itertools
import datetime
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Type
from abc import ABC, abstractmethod

from src.models.filters.regime.regime_detector import RegimeDetectorBase, MarketRegime
from src.models.optimization.result import OptimizationResult
from src.execution.backtest.backtest import run_backtest

logger = logging.getLogger(__name__)


class ComponentOptimizer(ABC):
    """
    Abstract base class for all component optimizers.
    
    This provides a common interface for different optimization techniques
    that can be applied to any component with get_parameters/set_parameters methods.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize the optimizer.
        
        Args:
            name: Optimizer name
        """
        self.name = name or self.__class__.__name__
        self.best_result = None
    
    @abstractmethod
    def optimize(self, component, param_space: Dict[str, List[Any]], 
                evaluation_function: Callable, **kwargs) -> Dict[str, Any]:
        """
        Optimize component parameters.
        
        Args:
            component: Component to optimize
            param_space: Dictionary mapping parameter names to possible values
            evaluation_function: Function to evaluate a parameter set
            **kwargs: Additional optimizer-specific parameters
            
        Returns:
            dict: Optimization results

        
        """
     
        pass
    
    def get_best_result(self) -> Optional[Dict[str, Any]]:
        """
        Get the best result found during optimization.
        
        Returns:
            dict: Best result or None if no optimization performed
        """
        return self.best_result


class GridSearchOptimizer(ComponentOptimizer):
    """
    Optimizer that performs exhaustive grid search over parameter space.
    
    This optimizer evaluates all possible combinations of parameters
    to find the optimal configuration.
    """
    
    def __init__(self, name: str = "grid_search_optimizer"):
        """
        Initialize the grid search optimizer.
        
        Args:
            name: Optimizer name
        """
        super().__init__(name)
    
    def optimize(self, component, param_space: Dict[str, List[Any]], 
                evaluation_function: Callable, **kwargs) -> Dict[str, Any]:
        """
        Perform grid search optimization.
        
        Args:
            component: Component to optimize
            param_space: Dictionary mapping parameter names to possible values
            evaluation_function: Function to evaluate a parameter set
            **kwargs: Additional parameters including:
                - constraints: Optional list of constraint functions
                - verbose: Whether to print progress
                
        Returns:
            dict: Optimization results
        """
        # Extract parameters
        constraints = kwargs.get('constraints', [])
        verbose = kwargs.get('verbose', True)
        
        # Generate all parameter combinations
        param_names = list(param_space.keys())
        param_values = [param_space[name] for name in param_names]
        param_values = [
            val if isinstance(val, (list, tuple)) else [val]
            for val in param_values
        ]
        combinations = list(itertools.product(*param_values))
        combinations = list(itertools.product(*param_values))
        total_evaluations = len(combinations)
        
        if verbose:
            logger.info(f"Grid search: evaluating {total_evaluations} parameter combinations")
        
        # Track best results
        best_params = None
        best_score = float('-inf')
        all_results = []
        
        # Save original parameters to restore later
        original_params = component.get_parameters()
        
        try:
            # Evaluate each combination
            for i, values in enumerate(combinations):
                # Create parameter dict
                params = {name: value for name, value in zip(param_names, values)}
                
                # Skip invalid parameter combinations
                if constraints and not all(constraint(params) for constraint in constraints):
                    continue
                
                # Apply parameters to component
                component.set_parameters(params)
                
                # Evaluate fitness
                score = evaluation_function(component, **kwargs)
                
                # Store result
                result = {'params': params, 'score': score}
                all_results.append(result)
                
                # Update best if improved
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    
                # Progress update
                if verbose and ((i + 1) % 10 == 0 or (i + 1) == total_evaluations):
                    logger.info(f"Evaluated {i + 1}/{total_evaluations} combinations")
        finally:
            # Restore original parameters
            component.set_parameters(original_params)
        
        # Store and return results
        self.best_result = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results,
            'n_evaluations': len(all_results)
        }
        
        return self.best_result


class RandomSearchOptimizer(ComponentOptimizer):
    """
    Optimizer that performs random sampling of parameter space.
    
    This optimizer evaluates random combinations of parameters,
    which can be more efficient than grid search for high-dimensional spaces.
    """
    
    def __init__(self, n_iterations: int = 100, name: str = "random_search_optimizer"):
        """
        Initialize the random search optimizer.
        
        Args:
            n_iterations: Number of random parameter sets to evaluate
            name: Optimizer name
        """
        super().__init__(name)
        self.n_iterations = n_iterations
    
    def optimize(self, component, param_space: Dict[str, List[Any]], 
                evaluation_function: Callable, **kwargs) -> Dict[str, Any]:
        """
        Perform random search optimization.
        
        Args:
            component: Component to optimize
            param_space: Dictionary mapping parameter names to possible values
            evaluation_function: Function to evaluate a parameter set
            **kwargs: Additional parameters including:
                - constraints: Optional list of constraint functions
                - verbose: Whether to print progress
                - seed: Random seed for reproducibility
                
        Returns:
            dict: Optimization results
        """
        # Extract parameters
        constraints = kwargs.get('constraints', [])
        verbose = kwargs.get('verbose', True)
        seed = kwargs.get('seed')
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        if verbose:
            logger.info(f"Random search: evaluating {self.n_iterations} random parameter sets")
        
        # Track best results
        best_params = None
        best_score = float('-inf')
        all_results = []
        
        # Save original parameters to restore later
        original_params = component.get_parameters()
        
        try:
            # Evaluate random combinations
            for i in range(self.n_iterations):
                # Generate random parameter set
                params = {}
                for name, values in param_space.items():
                    params[name] = random.choice(values)
                
                # Skip invalid parameter combinations
                if constraints and not all(constraint(params) for constraint in constraints):
                    continue
                
                # Apply parameters to component
                component.set_parameters(params)
                
                # Evaluate fitness
                score = evaluation_function(component, **kwargs)
                
                # Store result
                result = {'params': params, 'score': score}
                all_results.append(result)
                
                # Update best if improved
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    
                # Progress update
                if verbose and ((i + 1) % 10 == 0 or (i + 1) == self.n_iterations):
                    logger.info(f"Evaluated {i + 1}/{self.n_iterations} combinations")
        finally:
            # Restore original parameters
            component.set_parameters(original_params)
        
        # Store and return results
        self.best_result = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results,
            'n_evaluations': len(all_results)
        }
        
        return self.best_result


class GeneticOptimizer(ComponentOptimizer):
    """
    Optimizer that uses genetic algorithm for parameter search.
    
    This optimizer evolves a population of parameter sets using
    principles of natural selection (selection, crossover, mutation).
    """
    
    def __init__(self, population_size: int = 50, generations: int = 20,
               mutation_rate: float = 0.2, crossover_rate: float = 0.7,
               name: str = "genetic_optimizer"):
        """
        Initialize the genetic optimizer.
        
        Args:
            population_size: Number of individuals in population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            name: Optimizer name
        """
        super().__init__(name)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
    
    def optimize(self, component, param_space: Dict[str, List[Any]], 
                evaluation_function: Callable, **kwargs) -> Dict[str, Any]:
        """
        Perform genetic algorithm optimization.
        
        Args:
            component: Component to optimize
            param_space: Dictionary mapping parameter names to possible values
            evaluation_function: Function to evaluate a parameter set
            **kwargs: Additional parameters including:
                - constraints: Optional list of constraint functions
                - verbose: Whether to print progress
                - seed: Random seed for reproducibility
                
        Returns:
            dict: Optimization results
        """
        # Extract parameters
        constraints = kwargs.get('constraints', [])
        verbose = kwargs.get('verbose', True)
        seed = kwargs.get('seed')
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        if verbose:
            logger.info(f"Genetic optimization: {self.population_size} individuals, {self.generations} generations")
            
        # Initialize population with random parameter combinations
        population = self._initialize_population(param_space, constraints)
        
        # Save original parameters to restore later
        original_params = component.get_parameters()
        
        # Track best individual and generation statistics
        best_individual = None
        best_score = float('-inf')
        generation_stats = []
        all_individuals = []
        
        try:
            # Evolve population over generations
            for generation in range(self.generations):
                if verbose:
                    logger.info(f"Generation {generation+1}/{self.generations}")
                
                # Evaluate fitness for each individual
                fitnesses = []
                for individual in population:
                    # Apply parameters to component
                    component.set_parameters(individual)
                    
                    # Evaluate fitness
                    fitness = evaluation_function(component, **kwargs)
                    fitnesses.append(fitness)
                    
                    # Track best individual
                    if fitness > best_score:
                        best_score = fitness
                        best_individual = individual.copy()
                        
                    # Store individual results
                    all_individuals.append({
                        'params': individual.copy(),
                        'score': fitness,
                        'generation': generation + 1
                    })
                
                # Record generation statistics
                generation_stats.append({
                    'generation': generation + 1,
                    'max_fitness': max(fitnesses),
                    'min_fitness': min(fitnesses),
                    'avg_fitness': sum(fitnesses) / len(fitnesses)
                })
                
                # Early stopping if we're at the last generation
                if generation == self.generations - 1:
                    break
                
                # Selection and reproduction for next generation
                population = self._evolve_population(population, fitnesses, param_space, constraints)
        finally:
            # Restore original parameters
            component.set_parameters(original_params)
        
        # Store and return results
        self.best_result = {
            'best_params': best_individual,
            'best_score': best_score,
            'generation_stats': generation_stats,
            'all_results': all_individuals,
            'n_evaluations': len(all_individuals)
        }
        
        return self.best_result
    
    def _initialize_population(self, param_space: Dict[str, List[Any]], 
                             constraints: List[Callable]) -> List[Dict[str, Any]]:
        """
        Initialize population with random parameter sets.
        
        Args:
            param_space: Dictionary mapping parameter names to possible values
            constraints: List of constraint functions
            
        Returns:
            list: Population of parameter dictionaries
        """
        population = []
        
        # Create random individuals until we have enough
        while len(population) < self.population_size:
            # Generate random parameter set
            individual = {}
            for name, values in param_space.items():
                individual[name] = random.choice(values)
            
            # Add to population if valid
            if not constraints or all(constraint(individual) for constraint in constraints):
                population.append(individual)
        
        return population
    
    def _evolve_population(self, population: List[Dict[str, Any]], 
                         fitnesses: List[float], param_space: Dict[str, List[Any]], 
                         constraints: List[Callable]) -> List[Dict[str, Any]]:
        """
        Create next generation through selection, crossover and mutation.
        
        Args:
            population: Current population
            fitnesses: Fitness scores for current population
            param_space: Dictionary mapping parameter names to possible values
            constraints: List of constraint functions
            
        Returns:
            list: New population
        """
        # Calculate selection probabilities
        total_fitness = sum(max(0, f) for f in fitnesses)
        if total_fitness <= 0:
            # If all negative fitnesses, use rank selection
            ranked_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
            selection_probs = [1.0/i for i in range(1, len(ranked_indices)+1)]
        else:
            # Use fitness proportionate selection
            selection_probs = [max(0, f)/total_fitness for f in fitnesses]
        
        # Create new population
        new_population = []
        
        while len(new_population) < self.population_size:
            # Selection - choose two parents
            parent_indices = np.random.choice(
                range(len(population)), 
                size=2, 
                replace=False, 
                p=selection_probs
            )
            parent1 = population[parent_indices[0]]
            parent2 = population[parent_indices[1]]
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2, param_space)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1 = self._mutate(child1, param_space)
            child2 = self._mutate(child2, param_space)
            
            # Add to new population if valid
            if not constraints or all(constraint(child1) for constraint in constraints):
                new_population.append(child1)
            if len(new_population) < self.population_size:
                if not constraints or all(constraint(child2) for constraint in constraints):
                    new_population.append(child2)
        
        return new_population[:self.population_size]  # Ensure exact population size
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any], 
                 param_space: Dict[str, List[Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent parameter set
            parent2: Second parent parameter set
            param_space: Dictionary mapping parameter names to possible values
            
        Returns:
            tuple: Two child parameter sets
        """
        child1 = {}
        child2 = {}
        
        # Uniform crossover - for each parameter, randomly choose from either parent
        for param in param_space:
            if random.random() < 0.5:
                child1[param] = parent1[param]
                child2[param] = parent2[param]
            else:
                child1[param] = parent2[param]
                child2[param] = parent1[param]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, Any], 
              param_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Perform mutation on an individual.
        
        Args:
            individual: Parameter set to mutate
            param_space: Dictionary mapping parameter names to possible values
            
        Returns:
            dict: Mutated parameter set
        """
        mutated = individual.copy()
        
        # For each parameter, randomly decide whether to mutate
        for param, values in param_space.items():
            if random.random() < self.mutation_rate:
                # Choose a different value from the parameter space
                current_value = mutated[param]
                other_values = [v for v in values if v != current_value]
                if other_values:  # Only mutate if there are other values to choose from
                    mutated[param] = random.choice(other_values)
        
        return mutated


class WalkForwardOptimizer(ComponentOptimizer):
    """
    Optimizer that uses walk-forward analysis for parameter optimization.
    
    This optimizer divides data into multiple windows with training and testing
    periods to prevent overfitting and account for time-series data characteristics.
    """
    
    def __init__(self, train_size: float = 0.7, test_size: float = 0.3, 
               windows: int = 3, base_optimizer: Optional[ComponentOptimizer] = None,
               name: str = "walk_forward_optimizer"):
        """
        Initialize the walk-forward optimizer.
        
        Args:
            train_size: Proportion of window for training
            test_size: Proportion of window for testing
            windows: Number of train/test windows
            base_optimizer: Optimizer to use for each window (defaults to GridSearchOptimizer)
            name: Optimizer name
        """
        super().__init__(name)
        self.train_size = train_size
        self.test_size = test_size
        self.windows = windows
        self.base_optimizer = base_optimizer or GridSearchOptimizer()
    
    def optimize(self, component, param_space: Dict[str, List[Any]], 
                evaluation_function: Callable, **kwargs) -> Dict[str, Any]:
        """
        Perform walk-forward optimization.
        
        Args:
            component: Component to optimize
            param_space: Dictionary mapping parameter names to possible values
            evaluation_function: Function to evaluate a parameter set
            **kwargs: Additional parameters including:
                - data_handler: Required data handler with loaded data
                - start_date: Optional start date for optimization
                - end_date: Optional end date for optimization
                - verbose: Whether to print progress
                
        Returns:
            dict: Optimization results
        """
        # Extract required parameters
        data_handler = kwargs.get('data_handler')
        if data_handler is None:
            raise ValueError("data_handler is required for walk-forward optimization")
            
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        verbose = kwargs.get('verbose', True)
        
        # Calculate window boundaries
        window_boundaries = self._calculate_window_boundaries(data_handler, start_date, end_date)
        
        if verbose:
            logger.info(f"Walk-forward optimization: {self.windows} windows")
            for i, (train_start, train_end, test_start, test_end) in enumerate(window_boundaries):
                logger.info(f"  Window {i+1}: Train {train_start} to {train_end}, Test {test_start} to {test_end}")
        
        # Track window results
        window_results = []
        all_results = []
        
        # Save original parameters to restore later
        original_params = component.get_parameters()
        
        try:
            # Process each window
            for i, (train_start, train_end, test_start, test_end) in enumerate(window_boundaries):
                if verbose:
                    logger.info(f"\nProcessing window {i+1}/{len(window_boundaries)}")
                
                # Modify evaluation kwargs for this window's training period
                train_kwargs = dict(kwargs)
                train_kwargs['start_date'] = train_start
                train_kwargs['end_date'] = train_end
                
                # Create training evaluation function
                def train_eval_func(comp, **eval_kwargs):
                    return evaluation_function(comp, **train_kwargs)
                
                # Optimize on training window
                window_opt_result = self.base_optimizer.optimize(
                    component=component,
                    param_space=param_space,
                    evaluation_function=train_eval_func,
                    **train_kwargs
                )
                
                # Get best parameters from training
                best_params = window_opt_result.get('best_params')
                train_score = window_opt_result.get('best_score')
                
                # Prepare test kwargs
                test_kwargs = dict(kwargs)
                test_kwargs['start_date'] = test_start
                test_kwargs['end_date'] = test_end
                
                # Apply parameters and evaluate on test window
                component.set_parameters(best_params)
                test_score = evaluation_function(component, **test_kwargs)
                
                # Store window result
                window_result = {
                    'window': i + 1,
                    'train_period': (train_start, train_end),
                    'test_period': (test_start, test_end),
                    'params': best_params,
                    'train_score': train_score,
                    'test_score': test_score
                }
                window_results.append(window_result)
                
                # Add window's results to all results
                all_results.extend(window_opt_result.get('all_results', []))
                
                if verbose:
                    logger.info(f"  Window {i+1} results:")
                    logger.info(f"    Best parameters: {best_params}")
                    logger.info(f"    Train score: {train_score:.4f}, Test score: {test_score:.4f}")
        finally:
            # Restore original parameters
            component.set_parameters(original_params)
        
        # Analyze results to find most robust parameters
        robust_params = self._analyze_window_results(window_results)
        
        # Store and return results
        self.best_result = {
            'best_params': robust_params,
            'window_results': window_results,
            'all_results': all_results,
            'n_evaluations': len(all_results)
        }
        
        return self.best_result
    
    def _calculate_window_boundaries(self, data_handler, start_date=None, end_date=None):
        """
        Calculate window boundaries for walk-forward analysis.
        
        Args:
            data_handler: Data handler with loaded data
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            list: List of (train_start, train_end, test_start, test_end) tuples
        """
        symbol = data_handler.get_symbols()[0]  # Use first symbol
        
        # Get all dates from data handler
        data_handler.reset()
        dates = []
        
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
                
            date = bar.get_timestamp()
            
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                break
                
            dates.append(date)
            
        if not dates:
            raise ValueError("No data found in specified date range")
            
        # Sort dates
        dates.sort()
        
        # Calculate window size in number of bars
        total_bars = len(dates)
        window_size = total_bars // self.windows
        
        # Create windows
        window_boundaries = []
        
        for i in range(self.windows):
            # Calculate window start and end
            window_start = i * window_size
            window_end = (i + 1) * window_size if i < self.windows - 1 else total_bars
            
            # Calculate train/test split
            split_idx = window_start + int((window_end - window_start) * self.train_size)
            
            train_start = dates[window_start]
            train_end = dates[split_idx - 1]
            test_start = dates[split_idx]
            test_end = dates[window_end - 1]
            
            window_boundaries.append((train_start, train_end, test_start, test_end))
            
        return window_boundaries
    
    def _analyze_window_results(self, window_results):
        """
        Analyze window results to find most robust parameters.
        
        Args:
            window_results: List of window optimization results
            
        Returns:
            dict: Most robust parameter set
        """
        # Group results by parameter set
        param_results = {}
        
        for result in window_results:
            params_key = str(sorted(result['params'].items()))
            
            if params_key not in param_results:
                param_results[params_key] = {
                    'params': result['params'],
                    'train_scores': [],
                    'test_scores': [],
                    'windows': []
                }
                
            param_results[params_key]['train_scores'].append(result['train_score'])
            param_results[params_key]['test_scores'].append(result['test_score'])
            param_results[params_key]['windows'].append(result['window'])
            
        # Calculate robustness metrics for each parameter set
        for params_key, results in param_results.items():
            # Calculate statistics
            train_mean = np.mean(results['train_scores'])
            train_std = np.std(results['train_scores'])
            test_mean = np.mean(results['test_scores'])
            test_std = np.std(results['test_scores'])
            
            # Calculate robustness score (higher is better)
            # We prefer higher test scores with lower standard deviation
            if test_mean > 0:
                robustness = test_mean * (1.0 - 0.5 * (test_std / abs(test_mean)))
            else:
                robustness = test_mean / (1.0 + test_std)
                
            # Store metrics
            results['train_mean'] = train_mean
            results['train_std'] = train_std
            results['test_mean'] = test_mean
            results['test_std'] = test_std
            results['robustness'] = robustness
            
        # Find most robust parameter set
        best_key = max(param_results.keys(), key=lambda k: param_results[k]['robustness'])
        best_params = param_results[best_key]['params']
        
        return best_params


class RegimeBasedOptimizer(ComponentOptimizer):
    """
    Optimizer that tunes parameters differently for different market regimes.
    
    This optimizer uses a regime detector to segment data by market regime,
    then optimizes parameters separately for each regime.
    """
    
    def __init__(self, regime_detector: RegimeDetectorBase, 
               base_optimizer: Optional[ComponentOptimizer] = None,
               name: str = "regime_based_optimizer"):
        """
        Initialize the regime-based optimizer.
        
        Args:
            regime_detector: Regime detector instance
            base_optimizer: Optimizer to use for each regime (defaults to GridSearchOptimizer)
            name: Optimizer name
        """
        super().__init__(name)
        self.regime_detector = regime_detector
        self.base_optimizer = base_optimizer or GridSearchOptimizer()
    
    def optimize(self, component, param_space: Dict[str, List[Any]], 
                evaluation_function: Callable, **kwargs) -> Dict[str, Any]:
        """
        Perform regime-specific optimization.
        
        Args:
            component: Component to optimize
            param_space: Dictionary mapping parameter names to possible values
            evaluation_function: Function to evaluate a parameter set
            **kwargs: Additional parameters including:
                - data_handler: Required data handler with loaded data
                - start_date: Optional start date for optimization
                - end_date: Optional end date for optimization
                - min_regime_bars: Minimum bars required for a regime
                - min_improvement: Minimum improvement to use regime parameters
                - verbose: Whether to print progress
                
        Returns:
            dict: Optimization results
        """
        # Extract required parameters
        data_handler = kwargs.get('data_handler')
        if data_handler is None:
            raise ValueError("data_handler is required for regime-based optimization")
            
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        min_regime_bars = kwargs.get('min_regime_bars', 50)
        min_improvement = kwargs.get('min_improvement', 0.1)
        verbose = kwargs.get('verbose', True)
        
        # Reset regime detector
        if hasattr(self.regime_detector, 'reset'):
            self.regime_detector.reset()
        
        # 1. First detect regimes in the data
        if verbose:
            logger.info("Detecting market regimes in historical data")
        
        self._detect_regimes(data_handler, start_date, end_date)
        
        # Get symbol from data handler
        symbol = data_handler.get_symbols()[0]  # Use first symbol
        
        # 2. Get periods for each regime
        regime_periods = self.regime_detector.get_regime_periods(symbol, start_date, end_date)
        
        if verbose:
            logger.info("Regime periods detected:")
            for regime, periods in regime_periods.items():
                total_days = sum((end - start).total_seconds() / 86400 for start, end in periods) if periods else 0
                logger.info(f"  {regime.value}: {len(periods)} periods, {total_days:.1f} days")
        
        # 3. First optimize on all data as baseline
        if verbose:
            logger.info("\n--- Optimizing baseline parameters (all regimes) ---")
        
        baseline_result = self.base_optimizer.optimize(
            component=component,
            param_space=param_space,
            evaluation_function=evaluation_function,
            **kwargs
        )
        
        baseline_params = baseline_result.get('best_params', {})
        baseline_score = baseline_result.get('best_score', float('-inf'))
        
        if verbose:
            logger.info(f"Baseline parameters: {baseline_params}")
            logger.info(f"Baseline score: {baseline_score:.4f}")
        
        # 4. Optimize separately for each regime
        regime_params = {}
        regime_scores = {}
        all_results = []
        
        # Save original parameters to restore later
        original_params = component.get_parameters()
        
        try:
            for regime, periods in regime_periods.items():
                if not periods:
                    if verbose:
                        logger.info(f"\nSkipping {regime.value} regime - no data periods found")
                    continue
                    
                # Count total bars in this regime
                bar_count = self._count_bars_in_regime(data_handler, symbol, periods)
                
                if bar_count < min_regime_bars:
                    if verbose:
                        logger.info(f"\nSkipping {regime.value} regime - insufficient data ({bar_count} bars)")
                    continue
                    
                if verbose:
                    logger.info(f"\n--- Optimizing for {regime.value} regime ({bar_count} bars) ---")
                
                # Create regime-specific evaluation function
                def regime_eval_func(comp, **eval_kwargs):
                    return self._evaluate_in_regime(
                        comp, evaluation_function, data_handler, periods, **eval_kwargs
                    )
                
                # Optimize for this regime
                regime_result = self.base_optimizer.optimize(
                    component=component,
                    param_space=param_space,
                    evaluation_function=regime_eval_func,
                    **kwargs
                )
                
                # Extract results
                regime_best_params = regime_result.get('best_params', {})
                regime_score = regime_result.get('best_score', float('-inf'))
                
                # Add to all results
                all_results.extend(regime_result.get('all_results', []))
                
                # Check if regime-specific parameters are better than baseline
                # First evaluate baseline with regime-specific data
                component.set_parameters(baseline_params)
                baseline_regime_score = regime_eval_func(component)
                
                # Calculate improvement
                if baseline_regime_score != 0:
                    improvement = (regime_score - baseline_regime_score) / abs(baseline_regime_score)
                else:
                    improvement = float('inf') if regime_score > 0 else float('-inf')
                
                if verbose:
                    logger.info(f"Regime score: {regime_score:.4f}, Baseline on same data: {baseline_regime_score:.4f}")
                    logger.info(f"Improvement: {improvement:.2%}")
                
                # Store results if improvement is sufficient
                if improvement >= min_improvement:
                    if verbose:
                        logger.info(f"Using regime-specific parameters for {regime.value}")
                    
                    regime_params[regime] = regime_best_params
                    regime_scores[regime] = regime_score
                else:
                    if verbose:
                        logger.info(f"Improvement insufficient, using baseline parameters for {regime.value}")
                    
                    regime_params[regime] = baseline_params
                    regime_scores[regime] = baseline_regime_score
        finally:
            # Restore original parameters
            component.set_parameters(original_params)
        
        # 5. Include the unknown regime
        regime_params[MarketRegime.UNKNOWN] = baseline_params
        regime_scores[MarketRegime.UNKNOWN] = baseline_score
        
        # Store and return results
        self.best_result = {
            'baseline_parameters': baseline_params,
            'baseline_score': baseline_score,
            'regime_parameters': regime_params,
            'regime_scores': regime_scores,
            'all_results': all_results
        }
        
        return self.best_result
    
    def _detect_regimes(self, data_handler, start_date=None, end_date=None):
        """
        Detect regimes in historical data.
        
        Args:
            data_handler: Data handler with loaded data
            start_date: Optional start date
            end_date: Optional end date
        """
        symbol = data_handler.get_symbols()[0]
        
        # Reset data handler
        data_handler.reset()
        
        # Process bars to detect regimes
        bar_count = 0
        
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
                
            # Skip bars outside date range
            date = bar.get_timestamp()
            
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                break
                
            # Update detector with bar
            self.regime_detector.update(bar)
            bar_count += 1
            
        # Reset data handler
        data_handler.reset()
    
    def _count_bars_in_regime(self, data_handler, symbol, periods):
        """
        Count total bars across multiple time periods for a regime.
        
        Args:
            data_handler: Data handler with loaded data
            symbol: Symbol to count bars for
            periods: List of (start, end) tuples defining time periods
            
        Returns:
            int: Total bar count
        """
        # Reset data handler
        data_handler.reset()
        
        total_bars = 0
        
        # Process each bar
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
                
            # Check if bar falls in any of the periods
            date = bar.get_timestamp()
            
            for start, end in periods:
                # Make sure timestamps can be compared (remove timezone info if needed)
                start_comp = start.replace(tzinfo=None) if hasattr(start, 'tzinfo') and start.tzinfo else start
                end_comp = end.replace(tzinfo=None) if hasattr(end, 'tzinfo') and end.tzinfo else end
                date_comp = date.replace(tzinfo=None) if hasattr(date, 'tzinfo') and date.tzinfo else date
                
                if start_comp <= date_comp <= end_comp:
                    total_bars += 1
                    break
        
        # Reset data handler
        data_handler.reset()
        
        return total_bars
    
    def _evaluate_in_regime(self, component, evaluation_function, data_handler, periods, **kwargs):
        """
        Evaluate a component only on data from a specific regime.
        
        Args:
            component: Component to evaluate
            evaluation_function: Evaluation function
            data_handler: Data handler with loaded data
            periods: List of (start, end) tuples defining regime periods
            **kwargs: Additional parameters for evaluation
            
        Returns:
            float: Evaluation score
        """
        if not periods:
            return float('-inf')
            
        # Get time range from periods
        first_period_start = min(start for start, _ in periods)
        last_period_end = max(end for _, end in periods)
        
        # Modify data_handler to only process bars in the regime
        original_get_next_bar = data_handler.get_next_bar
        
        def regime_filtered_get_next_bar(symbol):
            bar = original_get_next_bar(symbol)
            
            if bar is None:
                return None
                
            # Check if bar is in regime periods
            date = bar.get_timestamp()
            date_comp = date.replace(tzinfo=None) if hasattr(date, 'tzinfo') and date.tzinfo else date
            
            for start, end in periods:
                start_comp = start.replace(tzinfo=None) if hasattr(start, 'tzinfo') and start.tzinfo else start
                end_comp = end.replace(tzinfo=None) if hasattr(end, 'tzinfo') and end.tzinfo else end
                
                if start_comp <= date_comp <= end_comp:
                    return bar
                    
            # If not in regime, recursively get next bar
            return regime_filtered_get_next_bar(symbol)
        
        try:
            # Replace get_next_bar with filtered version
            data_handler.get_next_bar = regime_filtered_get_next_bar
            
            # Create evaluation kwargs with full date range
            eval_kwargs = dict(kwargs)
            eval_kwargs['start_date'] = first_period_start
            eval_kwargs['end_date'] = last_period_end
            eval_kwargs['data_handler'] = data_handler
            
            # Evaluate component
            score = evaluation_function(component, **eval_kwargs)
            
            return score
        finally:
            # Restore original method
            data_handler.get_next_bar = original_get_next_bar
            
            # Reset data handler
            data_handler.reset()


# Export components
__all__ = [
    'ComponentOptimizer',
    'GridSearchOptimizer',
    'RandomSearchOptimizer',
    'GeneticOptimizer',
    'WalkForwardOptimizer',
    'RegimeBasedOptimizer'
]
