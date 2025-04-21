# src/models/components/mixins.py

from src.models.optimization.interfaces import OptimizationTarget
from src.models.filters.interfaces import FilterTarget

class OptimizableMixin(OptimizationTarget):
    """
    Mixin class to make components optimizable.
    """
    
    def get_parameters(self):
        """Get current parameter dict for optimization."""
        # Use component's params attribute
        return self.params.copy() if hasattr(self, 'params') else {}
    
    def set_parameters(self, params):
        """Set parameters from optimization result."""
        if hasattr(self, 'params'):
            # Update only existing parameters
            for key, value in params.items():
                if key in self.params:
                    self.params[key] = value
            
            # Call any initialization that depends on parameters
            if hasattr(self, '_initialize'):
                self._initialize()
    
    def validate_parameters(self, params):
        """Validate if parameters are valid for this target."""
        # Default implementation accepts all parameters
        return True



# src/models/components/mixins.py (continued)

class FilterableMixin(FilterTarget):
    """
    Mixin class to make components filterable.
    """
    
    def get_filter_context(self, symbol=None, timestamp=None):
        """Get context data needed for filter evaluation."""
        context = {
            'component': self,
            'symbol': symbol,
            'timestamp': timestamp
        }
        
        # Add parameters to context
        if hasattr(self, 'params'):
            context['params'] = self.params
            
        # Add component-specific data
        if hasattr(self, 'get_data') and symbol:
            context['data'] = self.get_data(symbol)
            
        return context
    
    def apply_filter_result(self, result, symbol=None, timestamp=None):
        """Apply filter result to this component."""
        # Store filter result
        if not hasattr(self, 'filter_results'):
            self.filter_results = {}
            
        if symbol:
            if symbol not in self.filter_results:
                self.filter_results[symbol] = {}
                
            self.filter_results[symbol][timestamp or 'latest'] = result    
