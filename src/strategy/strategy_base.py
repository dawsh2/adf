# Strategy without setters, all dependencies injected in constructor
class StrategyBase:
    def __init__(self, name, config=None, container=None, signal_emitter=None, order_emitter=None):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            config: Configuration object
            container: DI container
            signal_emitter: Signal emitter for generating signals
            order_emitter: Order emitter for executing trades
        """
        self.name = name
        self.config = config
        self.container = container
        self.signal_emitter = signal_emitter
        self.order_emitter = order_emitter
        
        # Load configuration
        self.params = self._load_parameters()
        
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
        
    # Methods for event handling directly in on_bar and on_signal without delegating
    def on_bar(self, event):
        """Process a bar event."""
        # Implementation directly in this method, not delegated to _process_bar
        pass
        
    def on_signal(self, event):
        """Process a signal event."""
        # Implementation directly in this method, not delegated to _process_signal
        pass
