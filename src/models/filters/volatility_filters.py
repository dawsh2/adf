# src/models/filters/volatility_filter.py
import numpy as np
from .filter_base import FilterBase
from .result import FilterResult

class VolatilityFilter(FilterBase):
    """Filter based on price volatility."""
    
    def __init__(self, name=None, lookback=21, threshold=0.2, comparison='less'):
        super().__init__(name or "volatility_filter")
        self.lookback = lookback  # Window for volatility calculation
        self.threshold = threshold  # Volatility threshold
        self.comparison = comparison  # 'less', 'greater'
        self.data = {}  # symbol -> historical close prices
    
    def evaluate(self, context, params=None):
        """Evaluate volatility condition."""
        # Use provided params or instance params
        p = params or {'lookback': self.lookback, 'threshold': self.threshold}
        
        # Extract data from context
        symbol = context.get('symbol')
        close_price = context.get('close')
        
        if not symbol or close_price is None:
            return FilterResult(False, "Missing required context data")
        
        # Update price history
        if symbol not in self.data:
            self.data[symbol] = []
        
        self.data[symbol].append(close_price)
        
        # Keep history manageable
        if len(self.data[symbol]) > max(p['lookback'] * 2, 100):
            self.data[symbol] = self.data[symbol][-max(p['lookback'] * 2, 100):]
        
        # Check if we have enough data
        if len(self.data[symbol]) < p['lookback']:
            return FilterResult(True, f"Insufficient data ({len(self.data[symbol])}/{p['lookback']})")
        
        # Calculate volatility (standard deviation of returns)
        prices = self.data[symbol][-p['lookback']:]
        returns = [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Evaluate against threshold
        if self.comparison == 'less':
            passed = volatility < p['threshold']
            reason = f"Volatility {volatility:.2%} {'<' if passed else '≥'} threshold {p['threshold']:.2%}"
        else:  # 'greater'
            passed = volatility > p['threshold']
            reason = f"Volatility {volatility:.2%} {'>' if passed else '≤'} threshold {p['threshold']:.2%}"
        
        return FilterResult(
            passed=passed,
            reason=reason,
            metadata={'volatility': volatility, 'threshold': p['threshold']}
        )
    
    def get_parameters(self):
        """Get filter parameters."""
        return {
            'lookback': self.lookback,
            'threshold': self.threshold,
            'comparison': self.comparison
        }
    
    def set_parameters(self, params):
        """Set filter parameters."""
        if 'lookback' in params:
            self.lookback = params['lookback']
        if 'threshold' in params:
            self.threshold = params['threshold']
        if 'comparison' in params:
            self.comparison = params['comparison']
