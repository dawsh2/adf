from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class PositionSizerBase(ABC):
    """Abstract base class for position sizers."""
    
    @abstractmethod
    def calculate_position_size(self, symbol, direction, price, portfolio, signal=None):
        """Calculate position size based on parameters and portfolio state."""
        pass


class PositionSizer(PositionSizerBase):
    """
    Calculates position sizes based on a specified method.
    Supports fixed size, percent of equity, percent risk, volatility, etc.
    """
    
    def __init__(self, method='fixed', params=None):
        """
        Initialize the position sizer.
        
        Args:
            method: Sizing method ('fixed', 'percent_equity', 'percent_risk', 'volatility')
            params: Parameters for the selected method
        """
        self.method = method
        self.params = params or {}
    
    def calculate_position_size(self, symbol, direction, price, portfolio, signal=None):
        """
        Calculate position size based on the selected method.
        
        Args:
            symbol: Instrument symbol
            direction: Trade direction ('BUY' or 'SELL')
            price: Current price
            portfolio: Portfolio object
            signal: Original signal event (for confidence-based sizing)
            
        Returns:
            float: Calculated position size
        """
        if price <= 0:
            return 0
            
        if self.method == 'fixed':
            return self._fixed_size()
        
        elif self.method == 'percent_equity':
            return self._percent_equity(price, portfolio)
        
        elif self.method == 'percent_risk':
            return self._percent_risk(price, portfolio, symbol)
        
        elif self.method == 'volatility':
            return self._volatility_based(price, symbol, portfolio)
        
        elif self.method == 'confidence':
            return self._confidence_based(price, portfolio, signal)
        
        return 0  # Default
    
    def _fixed_size(self):
        """Fixed contract/share size."""
        return self.params.get('shares', 1)
    
    def _percent_equity(self, price, portfolio):
        """Percentage of equity."""
        percent = self.params.get('percent', 0.02)  # Default 2%
        equity = portfolio.get_equity()
        
        return int((equity * percent) / price)
    
    def _percent_risk(self, price, portfolio, symbol):
        """Percentage of equity risked."""
        risk_percent = self.params.get('risk_percent', 0.01)  # Default 1%
        stop_percent = self.params.get('stop_percent', 0.05)  # Default 5%
        
        equity = portfolio.get_equity()
        risk_amount = equity * risk_percent
        
        # Calculate stop distance
        stop_distance = price * stop_percent
        
        # Position size = risk amount / stop distance
        size = int(risk_amount / stop_distance) if stop_distance > 0 else 0
        return size
    
    def _volatility_based(self, price, symbol, portfolio):
        """Position sizing based on volatility."""
        # This would typically use historical volatility data
        # Simplified implementation for now
        atr_multiple = self.params.get('atr_multiple', 2.0)
        fixed_atr = self.params.get('fixed_atr', price * 0.02)  # Default to 2% of price
        risk_amount = portfolio.get_equity() * self.params.get('risk_percent', 0.01)
        
        # Size = risk amount / (ATR * multiple)
        size = int(risk_amount / (fixed_atr * atr_multiple))
        return size
    
    def _confidence_based(self, price, portfolio, signal):
        """Size position based on signal confidence."""
        # Base size calculation using percent equity method
        base_size = self._percent_equity(price, portfolio)
        
        # Adjust by confidence if signal provides it
        confidence = 1.0
        if signal and hasattr(signal, 'get_confidence'):
            confidence = signal.get_confidence()
        
        # Apply confidence multiplier
        adjusted_size = int(base_size * confidence)
        return adjusted_size
