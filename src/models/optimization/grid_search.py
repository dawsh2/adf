# src/models/optimization/grid_search.py
from .optimizer_base import OptimizerBase
from itertools import product

class GridSearchOptimizer(OptimizerBase):
    """Exhaustive search through parameter space."""
    
    def optimize(self, param_space, fitness_function, constraints=None, **kwargs):
        """Perform grid search optimization."""
        # Get parameter names and values
        param_names = list(param_space.keys())
        param_values = [param_space[name] for name in param_names]
        
        # Generate all combinations
        combinations = list(product(*param_values))
        total_evaluations = len(combinations)
        
        print(f"Grid search: evaluating {total_evaluations} parameter combinations")
        
        # Track best results
        best_params = None
        best_score = float('-inf')
        all_results = []
        
        # Evaluate each combination
        for i, values in enumerate(combinations):
            # Create parameter dict
            params = {name: value for name, value in zip(param_names, values)}
            
            # Check constraints
            if constraints and not all(constraint(params) for constraint in constraints):
                continue
            
            # Evaluate fitness
            score = fitness_function(params)
            
            # Store result
            result = {'params': params, 'score': score}
            all_results.append(result)
            
            # Update best if improved
            if score > best_score:
                best_score = score
                best_params = params
                
            # Progress update
            if (i + 1) % 10 == 0 or (i + 1) == total_evaluations:
                print(f"Evaluated {i + 1}/{total_evaluations} combinations")
        
        # Store and return results
        self.best_result = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results,
            'n_evaluations': len(all_results)
        }
        
        return self.best_result
