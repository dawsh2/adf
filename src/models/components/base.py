from .mixins import OptimizableMixin




class ComponentBase:
    component_type = "components"
    
    def __init__(self, name, config=None, container=None):
        self.name = name
        self.config = config
        self.container = container
        self.event_bus = None
        
        # Load parameters
        self.params = self._load_parameters()
        self._validate_params()
    
    def set_event_bus(self, event_bus):
        """Set the event bus for this component."""
        self.event_bus = event_bus
        return self  # For method chaining
        
    def _load_parameters(self):
        """Load parameters from configuration."""
        if self.config and hasattr(self.config, 'get_section'):
            section = self.config.get_section(self.component_type)
            # Try to get component-specific config
            if hasattr(section, 'get_section'):
                component_section = section.get_section(self.name)
                return component_section.as_dict()
        
        # Return default parameters if no config available
        return self.default_params()
    
    @classmethod
    def default_params(cls):
        """Get default parameters for this component."""
        return {}
        
    def _validate_params(self):
        """Validate parameters."""
        # Default implementation does nothing
        pass
        
    def reset(self):
        """Reset component state."""
        # Default implementation does nothing
        pass    


# src/models/components/base.py

class IndicatorBase(ComponentBase, OptimizableMixin):
    """Base class for technical indicators."""
    
    component_type = "indicators"
    
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.values = {}  # symbol -> indicator values
    
    def calculate(self, data):
        """Calculate indicator value from data."""
        raise NotImplementedError("Subclasses must implement calculate()")
    
    def get_value(self, symbol):
        """Get current indicator value for a symbol."""
        return self.values.get(symbol)
    
    # Additional methods for OptimizableMixin
    def validate_parameters(self, params):
        """Validate indicator parameters."""
        # Default implementation accepts all parameters
        return True    

# src/models/components/base.py

class RuleBase(ComponentBase, OptimizableMixin):
    """Base class for trading rules."""
    
    component_type = "rules"
    
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.state = {}  # For storing rule state
    
    def on_bar(self, event):
        """Process a bar event to generate signals."""
        raise NotImplementedError("Subclasses must implement on_bar()")
    
    # Additional methods for OptimizableMixin
    def validate_parameters(self, params):
        """Validate rule parameters."""
        # Default implementation accepts all parameters
        return True    

# src/models/components/base.py

class StrategyBase(ComponentBase, OptimizableMixin, FilterableMixin):
    """Base class for all trading strategies."""
    
    component_type = "strategies"
    
    def __init__(self, name, symbols, **kwargs):
        super().__init__(name, **kwargs)
        
        # Convert single symbol to list
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        
        # Component collections
        self.indicators = {}
        self.features = {}
        self.rules = {}
        
        # Strategy state
        self.state = {}
        
        # Filter state
        self.filter_results = {}
        
        # Create components based on configuration
        self._setup_components()
    
    def _setup_components(self):
        """Setup strategy components."""
        # Implementation depends on specific strategy
        pass
    
    def on_bar(self, event):
        """Process a bar event."""
        # Check if filtered
        symbol = event.get_symbol()
        
        # Skip if filtered out
        if hasattr(self, 'filter_results') and symbol in self.filter_results:
            latest_result = self.filter_results[symbol].get('latest')
            if latest_result and not latest_result.passed:
                return None
        
        # Regular strategy logic
        pass
    
    # Additional methods for OptimizableMixin
    def validate_parameters(self, params):
        """Validate strategy parameters."""
        # Example validation logic
        if 'fast_window' in params and 'slow_window' in params:
            return params['fast_window'] < params['slow_window']
        return True        
