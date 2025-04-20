

# Component base with component_type as class attribute
class ComponentBase:
    component_type = "components"
    
    def __init__(self, name, config=None, container=None):
        self.name = name
        self.config = config
        self.container = container
        self.event_bus = None  # Initialize to None
        self.emitter = None    # Initialize to None
        
        # Load parameters
        self.params = self._load_parameters()
        self._validate_params()
    
    def set_event_bus(self, event_bus):
        """Set the event bus for this component."""
        self.event_bus = event_bus
        return self  # For method chaining
        
    def set_emitter(self, emitter):
        """Set the emitter for this component."""
        self.emitter = emitter
        return self  # For method chaining


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
        


class StrategyBase(ComponentBase):
    component_type = "strategies"
    
    def __init__(self, name, config=None, container=None, signal_emitter=None, order_emitter=None):
        super().__init__(name, config, container)
        self.signal_emitter = signal_emitter
        self.order_emitter = order_emitter
        
        # Component collections
        self.indicators = {}
        self.features = {}
        self.rules = {}
        
        # Track the symbols this strategy is monitoring
        self.symbols = set()
        
        # Strategy state
        self.state = {}
        
        # Create components based on configuration
        self._setup_components()
    
    # Event handlers
    def on_bar(self, event):
        pass
        
    def on_signal(self, event):
        pass        
