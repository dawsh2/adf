class FilterBase(ComponentBase):
    """Base class for market condition filters."""
    
    component_type = "filters"
    
    def __init__(self, name, config=None, container=None):
        """
        Initialize the filter.
        
        Args:
            name: Filter name
            config: Configuration object
            container: DI container
        """
        super().__init__(name, config, container)
        self.state = {}  # For storing filter state
    
    @classmethod
    def default_params(cls):
        """Get default parameters for this filter."""
        return {}
    
    def evaluate(self, data):
        """
        Evaluate market conditions and determine if filter passes.
        
        Args:
            data: Dictionary or DataFrame with market data
            
        Returns:
            bool: True if filter condition passes, False otherwise
        """
        raise NotImplementedError("Subclasses must implement evaluate()")
    
    def on_bar(self, event):
        """
        Process a bar event to update filter state.
        
        Args:
            event: BarEvent to process
            
        Returns:
            bool: Current filter state after processing the event
        """
        # Default implementation
        symbol = event.get_symbol()
        
        # Extract data needed for evaluation
        data = {
            'symbol': symbol,
            'timestamp': event.get_timestamp(),
            'open': event.get_open(),
            'high': event.get_high(),
            'low': event.get_low(),
            'close': event.get_close(),
            'volume': event.get_volume()
        }
        
        # Store additional data in state if needed
        if symbol not in self.state:
            self.state[symbol] = {'history': []}
            
        # Update state
        self.state[symbol]['history'].append(data)
        
        # Keep state history manageable
        max_history = self.params.get('max_history', 100)
        if len(self.state[symbol]['history']) > max_history:
            self.state[symbol]['history'] = self.state[symbol]['history'][-max_history:]
            
        # Evaluate filter condition
        result = self.evaluate(self.state[symbol])
        
        # Update filter state
        self.state[symbol]['active'] = result
        
        return result
    
    def is_active(self, symbol):
        """
        Check if filter is active for a symbol.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            bool: True if filter is active, False otherwise
        """
        if symbol not in self.state:
            return True  # Default to active if no state
            
        return self.state[symbol].get('active', True)
    
    def reset(self):
        """Reset the filter state."""
        self.state = {}


class RegimeFilterBase(FilterBase):
    """Abstract base class for regime-based filters."""
    
    component_type = "filters.regime"  # More specific configuration section
    
    @classmethod
    def default_params(cls):
        """Get default parameters for regime filters."""
        return {
            'lookback': 63,  # ~3 months of trading days
            'warmup_period': 126,  # Period needed before regime detection is reliable
        }
    
    @abstractmethod
    def detect_regime(self, data):
        """
        Detect the current market regime.
        
        Args:
            data: Dictionary or DataFrame with market data
            
        Returns:
            str: Identified regime (e.g., 'bullish', 'bearish', 'sideways', 'volatile')
        """
        pass
    
    def evaluate(self, data):
        """Determine if current regime passes filter criteria."""
        # Check if we have enough data
        history = data.get('history', [])
        if len(history) < self.params.get('warmup_period', 126):
            return True  # Default to active during warmup period
        
        # Detect regime
        regime = self.detect_regime(data)
        
        # Store detected regime in state
        data['current_regime'] = regime
        
        # Check if this regime is in allowed regimes
        allowed_regimes = self.params.get('allowed_regimes', [])
        if allowed_regimes and regime not in allowed_regimes:
            return False
            
        return True
        
