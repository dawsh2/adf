# Component base with component_type as class attribute
class ComponentBase:
    component_type = "components"  # Base value, overridden in subclasses
    
    def __init__(self, name, config=None, container=None):
        self.name = name
        self.config = config
        self.container = container
        
        # Get configuration section using class attribute
        self.params = self._load_parameters()
        self._validate_params()
    
    def _load_parameters(self):
        """Load parameters from configuration."""
        if not self.config:
            return self.default_params()
            
        # Use class attribute for component type
        return self.config.get_section(self.component_type).get_dict(self.name, self.default_params())

# Indicators with component_type as class attribute
class IndicatorBase(ComponentBase):
    component_type = "indicators"
    
    def __init__(self, name, config=None, container=None):
        super().__init__(name, config, container)
        self.values = {}  # symbol -> indicator values

class FeatureBase(ComponentBase):
    component_type = "features"
    
    def __init__(self, name, config=None, container=None, indicators=None):
        """
        Initialize the feature.
        
        Args:
            name: Feature name
            config: Configuration object
            container: DI container
            indicators: Dictionary of indicators to use (injected)
        """
        super().__init__(name, config, container)
        self.values = {}  # symbol -> feature value
        self.indicators = indicators or {}
    
    def calculate(self, data):
        """
        Calculate the feature value from data.
        
        Args:
            data: Dictionary or DataFrame with price and indicator data
            
        Returns:
            Feature value(s)
        """
        raise NotImplementedError("Subclasses must implement calculate()")
    
    def on_bar(self, event):
        """Process a bar event to update feature values."""
        # Default implementation - subclasses may override
        pass


class RuleBase(ComponentBase):
    component_type = "rules"
    
    def __init__(self, name, config=None, container=None, signal_emitter=None):
        """
        Initialize the rule.
        
        Args:
            name: Rule name
            config: Configuration object
            container: DI container
            signal_emitter: Signal emitter for generating signals (injected)
        """
        super().__init__(name, config, container)
        self.signal_emitter = signal_emitter
        self.state = {}  # For storing rule state
    
    def on_bar(self, event):
        """
        Process a bar event to generate signals.
        
        Args:
            event: BarEvent to process
        """
        raise NotImplementedError("Subclasses must implement on_bar()")
    
    def reset(self):
        """Reset the rule state."""
        self.state = {}    
        
