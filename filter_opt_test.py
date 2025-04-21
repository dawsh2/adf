# Example test/demo for optimization
from src.models.optimization.grid_search import GridSearchOptimizer
from src.models.optimization.manager import OptimizationManager

# Create simple MA crossover strategy for testing
from src.models.components.base import StrategyBase
from src.models.components.mixins import OptimizableMixin

class MACrossoverStrategy(StrategyBase, OptimizableMixin):
    def __init__(self, name, symbols, fast_window=10, slow_window=30):
        super().__init__(name, symbols)
        self.params = {
            'fast_window': fast_window,
            'slow_window': slow_window
        }
    
    def validate_parameters(self, params):
        """Validate parameters."""
        if 'fast_window' in params and 'slow_window' in params:
            return params['fast_window'] < params['slow_window']
        return True

# Set up optimization
optimizer = GridSearchOptimizer()
manager = OptimizationManager()

# Register components
manager.register_optimizer('grid', optimizer)
manager.register_target('ma_strategy', MACrossoverStrategy('test', ['AAPL']))

# Define evaluation function
def evaluate_strategy(params):
    """Simple evaluation function for testing."""
    # In a real system, this would run a backtest
    # For this example, just use a simple metric
    fast = params.get('fast_window', 10)
    slow = params.get('slow_window', 30)
    # Arbitrary scoring function for testing
    return -(slow - fast - 20)**2 + 100

manager.register_evaluator('test_eval', evaluate_strategy)

# Create parameter space
param_space = {
    'fast_window': range(5, 30, 5),
    'slow_window': range(20, 100, 10)
}

# Define simple sequential optimization
def sequential_optimization(manager, optimizer, targets, evaluator, **kwargs):
    """Simple sequential optimization."""
    param_spaces = kwargs.get('param_spaces', {})
    results = {}
    
    for name, target in targets.items():
        space = param_spaces.get(name, {})
        
        def evaluate(params):
            # Save original parameters
            original = target.get_parameters()
            # Apply new parameters
            target.set_parameters(params)
            # Evaluate
            score = evaluator()
            # Restore original parameters
            target.set_parameters(original)
            return score
        
        # Run optimization
        result = optimizer.optimize(space, evaluate)
        
        # Apply best parameters
        target.set_parameters(result['best_params'])
        
        # Store result
        results[name] = result
    
    return results

# Register sequence
manager.register_sequence('sequential', sequential_optimization)

# Run optimization
results = manager.run_optimization(
    sequence_name='sequential',
    optimizer_name='grid', 
    target_names=['ma_strategy'],
    evaluator_name='test_eval',
    param_spaces={'ma_strategy': param_space}
)

# Print results
print("Optimization results:")
print(f"Best parameters: {results['ma_strategy']['best_params']}")
print(f"Best score: {results['ma_strategy']['best_score']}")
