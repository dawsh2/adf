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
        

