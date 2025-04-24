"""
Optimization validator to ensure optimization is working correctly.
"""
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class OptimizationValidator:
    """
    Tool to validate optimization results and compare strategies.
    
    This helps verify that the optimization is functioning correctly by:
    1. Testing multiple parameter combinations independently 
    2. Comparing results to make sure the best parameters are being found
    3. Providing visualizations of the parameter landscape
    """
    
    def __init__(self, optimizer_manager, data_handler, start_date=None, end_date=None):
        """
        Initialize the optimization validator.
        
        Args:
            optimizer_manager: Optimization manager to test
            data_handler: Data handler with test data
            start_date: Start date for validation tests
            end_date: End date for validation tests
        """
        self.optimizer_manager = optimizer_manager
        self.data_handler = data_handler
        self.start_date = start_date
        self.end_date = end_date
        self.results = {}
        
        # Storage for parameter sweep results
        self.parameter_sweep_results = []
        
    def validate_component(self, component_name, param_space, evaluator_name="sharpe_ratio", **kwargs):
        """
        Validate optimization of a component with brute force verification.
        
        Args:
            component_name: Name of component to validate
            param_space: Parameter space to test
            evaluator_name: Name of evaluator to use
            **kwargs: Additional parameters for optimization
            
        Returns:
            dict: Validation results
        """
        logger.info(f"Validating optimization for {component_name}")
        
        # Get the component
        component = self.optimizer_manager.targets.get(component_name)
        if not component:
            raise ValueError(f"Component {component_name} not found")
            
        # Get the evaluator
        evaluator = self.optimizer_manager.evaluators.get(evaluator_name)
        if not evaluator:
            raise ValueError(f"Evaluator {evaluator_name} not found")
            
        # 1. Run optimization normally
        logger.info("Running optimization...")
        opt_result = self.optimizer_manager.optimize_component(
            target_name=component_name,
            optimizer_name="grid",  # Use grid search for validation
            evaluator_name=evaluator_name,
            param_space=param_space,
            data_handler=self.data_handler,
            start_date=self.start_date,
            end_date=self.end_date,
            **kwargs
        )
        
        # Store result
        self.results['optimization'] = opt_result
        
        # 2. Run manual parameter sweep to verify
        logger.info("Running manual parameter sweep for verification...")
        self._run_parameter_sweep(component, param_space, evaluator)
        
        # 3. Compare results
        validation_passed = self._compare_results(opt_result)
        
        # Return validation results
        return {
            'optimization_result': opt_result,
            'parameter_sweep': self.parameter_sweep_results,
            'validation_passed': validation_passed
        }
    
    def _run_parameter_sweep(self, component, param_space, evaluator):
        """
        Run a manual parameter sweep to verify optimization results.
        
        Args:
            component: Component to test
            param_space: Parameter space to test
            evaluator: Evaluation function
        """
        # Generate all parameter combinations
        import itertools
        
        # Get parameter names and values
        param_names = list(param_space.keys())
        param_values = [param_space[name] for name in param_names]
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        total_combinations = len(combinations)
        
        logger.info(f"Testing {total_combinations} parameter combinations...")
        
        # Save original parameters
        original_params = component.get_parameters()
        
        # Test each combination
        sweep_results = []
        
        try:
            for i, values in enumerate(combinations):
                # Create parameter dict
                params = {name: value for name, value in zip(param_names, values)}
                
                # Apply parameters
                component.set_parameters(params)
                
                # Evaluate
                score = evaluator(
                    component, 
                    data_handler=self.data_handler,
                    start_date=self.start_date,
                    end_date=self.end_date
                )
                
                # Store result
                sweep_results.append({
                    'params': params.copy(),
                    'score': score
                })
                
                # Log progress
                if (i + 1) % 5 == 0 or (i + 1) == total_combinations:
                    logger.info(f"Tested {i + 1}/{total_combinations} combinations")
        finally:
            # Restore original parameters
            component.set_parameters(original_params)
        
        # Sort by score (descending)
        sweep_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Store results
        self.parameter_sweep_results = sweep_results
    
    def _compare_results(self, optimization_result):
        """
        Compare optimization results with manual parameter sweep.
        
        Args:
            optimization_result: Result from the optimizer
            
        Returns:
            bool: True if validation passed
        """
        # Check if we have both results
        if not optimization_result or not self.parameter_sweep_results:
            logger.error("Missing results for comparison")
            return False
            
        # Get best parameters from both methods
        opt_best_params = optimization_result.get('best_params')
        opt_best_score = optimization_result.get('best_score')
        
        sweep_best = self.parameter_sweep_results[0]
        sweep_best_params = sweep_best['params']
        sweep_best_score = sweep_best['score']
        
        # Log the comparison
        logger.info("\nComparing results:")
        logger.info(f"Optimization best: {opt_best_params} with score {opt_best_score}")
        logger.info(f"Parameter sweep best: {sweep_best_params} with score {sweep_best_score}")
        
        # Check if the best parameters match
        params_match = opt_best_params == sweep_best_params
        
        # Check if scores are within tolerance
        score_tolerance = 1e-6
        scores_match = abs(opt_best_score - sweep_best_score) < score_tolerance
        
        # Log the comparison
        if params_match:
            logger.info("✓ Parameters match!")
        else:
            logger.warning("✗ Parameters don't match")
            
        if scores_match:
            logger.info("✓ Scores match!")
        else:
            logger.warning(f"✗ Scores don't match: diff = {abs(opt_best_score - sweep_best_score)}")
            
        # Check optimization vs random parameters
        if len(self.parameter_sweep_results) > 1:
            # Get a random result (not the best)
            import random
            random_idx = random.randint(1, min(len(self.parameter_sweep_results) - 1, 10))
            random_result = self.parameter_sweep_results[random_idx]
            
            logger.info(f"Random parameters: {random_result['params']} with score {random_result['score']}")
            
            # Check if optimization is better than random
            optimization_better = opt_best_score > random_result['score']
            
            if optimization_better:
                logger.info("✓ Optimization outperforms random parameters")
            else:
                logger.warning("✗ Optimization doesn't outperform random parameters")
                
            return params_match and scores_match and optimization_better
        
        return params_match and scores_match
    
    def plot_parameter_landscape(self, param1, param2, metric='score'):
        """
        Plot the parameter landscape for two parameters.
        
        Args:
            param1: First parameter name
            param2: Second parameter name
            metric: Metric to plot (default: 'score')
            
        Returns:
            matplotlib.Figure: Plot figure
        """
        if not self.parameter_sweep_results:
            logger.error("No parameter sweep results to plot")
            return None
        
        # Extract unique parameter values
        unique_param1 = sorted(set(result['params'][param1] for result in self.parameter_sweep_results))
        unique_param2 = sorted(set(result['params'][param2] for result in self.parameter_sweep_results))
        
        # Create grid for heatmap
        grid = np.zeros((len(unique_param1), len(unique_param2)))
        
        # Fill grid with scores
        for result in self.parameter_sweep_results:
            i = unique_param1.index(result['params'][param1])
            j = unique_param2.index(result['params'][param2])
            grid[i, j] = result['score']
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(grid, interpolation='nearest', cmap='viridis')
        
        # Set labels
        ax.set_xticks(np.arange(len(unique_param2)))
        ax.set_yticks(np.arange(len(unique_param1)))
        ax.set_xticklabels(unique_param2)
        ax.set_yticklabels(unique_param1)
        
        ax.set_xlabel(param2)
        ax.set_ylabel(param1)
        ax.set_title(f'Parameter Landscape: {metric}')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(metric, rotation=-90, va="bottom")
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        for i in range(len(unique_param1)):
            for j in range(len(unique_param2)):
                text = ax.text(j, i, f"{grid[i, j]:.2f}",
                              ha="center", va="center", color="w")
        
        fig.tight_layout()
        return fig
    
    def generate_report(self):
        """
        Generate a comprehensive report of the validation results.
        
        Returns:
            str: Formatted report
        """
        if not self.results or not self.parameter_sweep_results:
            return "No validation results available. Run validate_component() first."
            
        # Generate report string
        report = ["# Optimization Validation Report\n"]
        
        # Add optimization results
        opt_result = self.results.get('optimization', {})
        opt_best_params = opt_result.get('best_params', {})
        opt_best_score = opt_result.get('best_score', 0)
        
        report.append("## Optimization Results\n")
        report.append(f"Best parameters: {opt_best_params}")
        report.append(f"Best score: {opt_best_score}\n")
        
        # Add parameter sweep results
        report.append("## Parameter Sweep Results\n")
        report.append("Top 5 parameter combinations:\n")
        
        for i, result in enumerate(self.parameter_sweep_results[:5], 1):
            params = result['params']
            score = result['score']
            report.append(f"{i}. Parameters: {params}, Score: {score}")
        
        report.append("\n## Validation Summary\n")
        
        # Compare best results
        if self.parameter_sweep_results:
            sweep_best = self.parameter_sweep_results[0]
            sweep_best_params = sweep_best['params']
            sweep_best_score = sweep_best['score']
            
            params_match = opt_best_params == sweep_best_params
            score_tolerance = 1e-6
            scores_match = abs(opt_best_score - sweep_best_score) < score_tolerance
            
            report.append(f"Parameters match: {'Yes' if params_match else 'No'}")
            report.append(f"Scores match: {'Yes' if scores_match else 'No'}")
            
            if not params_match:
                report.append("\nParameter differences:")
                for param in opt_best_params:
                    if param in sweep_best_params and opt_best_params[param] != sweep_best_params[param]:
                        report.append(f"- {param}: {opt_best_params[param]} vs {sweep_best_params[param]}")
        
        return "\n".join(report)
