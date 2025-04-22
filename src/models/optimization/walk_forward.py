"""
Walk-Forward Optimization Module

This module implements walk-forward optimization for strategy parameter tuning,
which is a method of testing a trading strategy by dividing historical data into
in-sample (IS) and out-of-sample (OOS) segments to prevent curve-fitting.
"""
import logging
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import itertools

from .optimizer_base import OptimizerBase
from .result import OptimizationResult

logger = logging.getLogger(__name__)

class WalkForwardOptimizer(OptimizerBase):
    """
    Walk-forward optimization for strategy parameters.
    Uses time-based train/test splits to prevent look-ahead bias.
    
    Walk-forward optimization divides the data into multiple windows,
    each with a training period and a testing period. For each window,
    parameters are optimized on the training data and evaluated on the
    testing data. This approach respects time-series chronological ordering.
    """
    
    def __init__(self, train_size=0.6, test_size=0.4, windows=3, name="walk_forward_optimizer"):
        """
        Initialize the walk-forward optimizer.
        
        Args:
            train_size: Proportion of data for training (in each window)
            test_size: Proportion of data for testing (in each window)
            windows: Number of train/test windows to use
            name: Optimizer name
        """
        super().__init__(name)
        self.train_size = train_size
        self.test_size = test_size
        self.windows = windows
        self.results = []
        
        # Validate inputs
        if train_size + test_size != 1.0:
            logger.warning(f"Train size ({train_size}) + test size ({test_size}) != 1.0. "
                          f"Results may not cover all data.")
        
        if train_size <= 0 or test_size <= 0:
            raise ValueError("Both train_size and test_size must be positive")
    
    def optimize(self, param_space: Dict[str, List[Any]], 
                fitness_function: Callable[[Dict[str, Any]], float],
                constraints: Optional[List[Callable[[Dict[str, Any]], bool]]] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Perform walk-forward optimization.

        Args:
            param_space: Dictionary mapping parameter names to possible values
            fitness_function: Function that evaluates parameter combinations
            constraints: Optional list of constraint functions
            **kwargs: Additional parameters including:
                - data_handler: Required data handler with loaded data
                - start_date: Optional start date for optimization
                - end_date: Optional end date for optimization
                - base_optimizer: Optional optimizer to use for each window

        Returns:
            Dictionary with optimization results
        """
        # Extract required arguments
        data_handler = kwargs.get('data_handler')
        if data_handler is None:
            raise ValueError("data_handler is required for walk-forward optimization")
        
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        base_optimizer = kwargs.get('base_optimizer')
        
        # Get symbols from data handler
        symbols = data_handler.get_symbols()
        if not symbols:
            raise ValueError("No symbols available in data handler")
        
        symbol = symbols[0]  # Use first symbol
        
        logger.info(f"Starting walk-forward optimization with {self.windows} windows")
        logger.info(f"Train size: {self.train_size:.2f}, Test size: {self.test_size:.2f}")
        
        # Reset data handler and collect date range information
        data_handler.reset()
        
        # Find actual date range from the data
        first_date, last_date = self._get_date_range(data_handler, symbol, start_date, end_date)
        
        if first_date is None or last_date is None:
            raise ValueError("Could not determine date range from data")
        
        logger.info(f"Optimization period: {first_date} to {last_date}")
        
        # Create time-based windows
        window_results = []
        
        # Calculate total date range in seconds
        date_range = (last_date - first_date).total_seconds()
        window_size = date_range / self.windows
        
        for i in range(self.windows):
            # Calculate window dates
            window_start = first_date + datetime.timedelta(seconds=i * window_size)
            window_end = first_date + datetime.timedelta(seconds=(i + 1) * window_size)
            
            # Calculate train/test split within window
            split_point = window_start + datetime.timedelta(seconds=window_size * self.train_size)
            
            train_start = window_start
            train_end = split_point
            test_start = split_point
            test_end = window_end
            
            logger.info(f"\nWindow {i+1}/{self.windows}:")
            logger.info(f"  Train: {train_start} to {train_end}")
            logger.info(f"  Test:  {test_start} to {test_end}")
            
            try:
                # Optimize on training data
                if base_optimizer:
                    # Use provided optimizer
                    train_kwargs = dict(kwargs)
                    train_kwargs.update({
                        'start_date': train_start,
                        'end_date': train_end
                    })
                    
                    # Run optimization on training window
                    train_result = base_optimizer.optimize(
                        param_space, 
                        fitness_function, 
                        constraints,
                        **train_kwargs
                    )
                    
                    best_params = train_result.get('best_params')
                    best_score = train_result.get('best_score')
                else:
                    # Use simple grid search if no optimizer provided
                    best_params, best_score = self._grid_search(
                        param_space, 
                        fitness_function,
                        constraints,
                        train_start, 
                        train_end,
                        **kwargs
                    )
                
                # Test on out-of-sample data
                test_score = self._evaluate_parameters(
                    best_params, 
                    fitness_function, 
                    test_start, 
                    test_end,
                    **kwargs
                )
                
                logger.info(f"  Best parameters: {best_params}")
                logger.info(f"  Train score: {best_score:.4f}, Test score: {test_score:.4f}")
                
                # Store window results
                window_results.append({
                    'window': i + 1,
                    'params': best_params,
                    'train_score': best_score,
                    'test_score': test_score,
                    'train_period': (train_start, train_end),
                    'test_period': (test_start, test_end)
                })
                
            except Exception as e:
                logger.error(f"Error in optimization window {i+1}: {e}")
                import traceback
                traceback.print_exc()
        
        # If no windows completed successfully, return empty result
        if not window_results:
            logger.error("No optimization windows completed successfully")
            return {
                'best_params': None,
                'best_score': float('-inf'),
                'window_results': []
            }
        
        # Aggregate results across windows
        param_performances = self._analyze_window_results(window_results)
        
        # Find best overall parameters
        best_key = max(param_performances.keys(), 
                     key=lambda k: param_performances[k]['avg_score'])
        best_overall = param_performances[best_key]
        
        # Format final results
        robust_score = (
            best_overall['avg_score'] * 
            (1 - 0.5 * (best_overall['std_score'] / max(abs(best_overall['avg_score']), 1e-10)))
        )
        
        optimization_result = {
            'best_params': best_overall['params'],
            'best_avg_score': best_overall['avg_score'],
            'best_min_score': best_overall['min_score'],
            'best_max_score': best_overall['max_score'],
            'best_std_score': best_overall['std_score'],
            'robust_score': robust_score,
            'window_results': window_results,
            'all_params': [param_performances[k] for k in param_performances]
        }
        
        # Sort all parameter results by average score (descending)
        optimization_result['all_params'].sort(
            key=lambda x: x['avg_score'], 
            reverse=True
        )
        
        # Store result for the optimizer's get_best_result method
        self.best_result = {
            'params': best_overall['params'],
            'score': best_overall['avg_score']
        }
        
        return optimization_result
    
    def _get_date_range(self, data_handler, symbol, start_date=None, end_date=None):
        """
        Determine the actual date range of data to optimize over.
        
        Args:
            data_handler: Data handler with loaded data
            symbol: Symbol to analyze
            start_date: Optional start date constraint
            end_date: Optional end date constraint
            
        Returns:
            tuple: (first_date, last_date)
        """
        # Reset data handler
        data_handler.reset()
        
        first_date = None
        last_date = None
        
        # Collect all bar dates
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
                
            bar_date = bar.get_timestamp()
            
            # Ignore bars outside requested date range
            if start_date and bar_date < start_date:
                continue
            if end_date and bar_date > end_date:
                break
                
            # Track date range
            if first_date is None:
                first_date = bar_date
            last_date = bar_date
        
        # Reset data handler for subsequent operations
        data_handler.reset()
        
        return first_date, last_date
    
    def _grid_search(self, param_space, fitness_function, constraints, 
                   start_date, end_date, **kwargs):
        """
        Perform grid search on a specific time period.
        
        Args:
            param_space: Dictionary of parameters to test
            fitness_function: Function to evaluate parameters
            constraints: List of constraint functions or None
            start_date: Start date for training
            end_date: End date for training
            **kwargs: Additional parameters for fitness function
            
        Returns:
            tuple: (best_params, best_score)
        """
        # Generate all parameter combinations
        param_names = sorted(param_space.keys())
        param_values = [param_space[name] for name in param_names]
        combinations = list(itertools.product(*param_values))
        
        logger.info(f"Grid search: evaluating {len(combinations)} parameter combinations")
        
        # Track best parameters
        best_params = None
        best_score = float('-inf')
        
        # Evaluate each combination
        for i, values in enumerate(combinations):
            # Create parameter dict
            params = {name: value for name, value in zip(param_names, values)}
            
            # Skip invalid parameter combinations
            if constraints:
                if not all(constraint(params) for constraint in constraints):
                    continue
            
            # Update kwargs with date range
            eval_kwargs = dict(kwargs)
            eval_kwargs.update({
                'start_date': start_date,
                'end_date': end_date
            })
            
            # Evaluate on training data
            score = self._evaluate_parameters(params, fitness_function, 
                                            start_date, end_date, **kwargs)
            
            # Update best if improved
            if score > best_score:
                best_score = score
                best_params = params
                
            # Progress update
            if (i + 1) % 5 == 0 or (i + 1) == len(combinations):
                logger.info(f"Evaluated {i + 1}/{len(combinations)} combinations")
        
        return best_params, best_score
    
    def _evaluate_parameters(self, params, fitness_function, 
                           start_date, end_date, **kwargs):
        """
        Evaluate parameters on a specific time period.
        
        Args:
            params: Parameters to evaluate
            fitness_function: Function to evaluate parameters
            start_date: Start date for evaluation
            end_date: End date for evaluation
            **kwargs: Additional parameters for fitness function
            
        Returns:
            float: Evaluation score
        """
        # Update kwargs with date range
        eval_kwargs = dict(kwargs)
        eval_kwargs.update({
            'start_date': start_date,
            'end_date': end_date
        })
        
        # Call fitness function
        try:
            result = fitness_function(params, **eval_kwargs)
            
            # Handle different return types
            if isinstance(result, dict):
                # If dictionary returned, extract score
                score_key = kwargs.get('score_key', 'score')
                if score_key in result:
                    score = result[score_key]
                else:
                    # Try common metric names
                    for key in ['sharpe_ratio', 'return', 'profit', 'pnl']:
                        if key in result:
                            score = result[key]
                            break
                    else:
                        # Use first value if no recognized keys
                        score = next(iter(result.values()))
            else:
                # Assume result is the score
                score = result
                
            return score
        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")
            return float('-inf')
    
    def _analyze_window_results(self, window_results):
        """
        Analyze results across windows to find robust parameters.
        
        Args:
            window_results: List of results from each window
            
        Returns:
            dict: Analysis of parameter performance
        """
        # Track test performance for each parameter set
        param_performances = {}
        
        # Collect test scores for each parameter set
        for result in window_results:
            params_key = self._params_to_key(result['params'])
            
            if params_key not in param_performances:
                param_performances[params_key] = {
                    'params': result['params'],
                    'scores': [],
                    'windows': []
                }
                
            param_performances[params_key]['scores'].append(result['test_score'])
            param_performances[params_key]['windows'].append(result['window'])
        
        # Calculate statistics for each parameter set
        for params_key, performance in param_performances.items():
            scores = performance['scores']
            
            # Calculate statistics
            performance['avg_score'] = np.mean(scores) if scores else float('-inf')
            performance['min_score'] = min(scores) if scores else float('-inf')
            performance['max_score'] = max(scores) if scores else float('-inf')
            performance['std_score'] = np.std(scores) if len(scores) > 1 else 0.0
            performance['score_count'] = len(scores)
            
        return param_performances
    
    def _params_to_key(self, params):
        """Convert parameters dict to string key for tracking."""
        # Sort items to ensure consistent keys for same parameters
        sorted_items = sorted(params.items())
        return str(sorted_items)
    
    def _key_to_params(self, key):
        """Convert string key back to parameters dict."""
        try:
            # Parse string representation of sorted items back to dict
            items = eval(key)
            return dict(items)
        except:
            logger.error(f"Error parsing parameter key: {key}")
            return {}
