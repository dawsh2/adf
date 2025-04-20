class Position:
    """
    Represents a position in a single instrument.
    """
    
    def __init__(self, symbol):
        self.symbol = symbol
        self.quantity = 0
        self.cost_basis = 0.0
        self.realized_pnl = 0.0
    
    def add_quantity(self, quantity, price):
        """Add to the position."""
        if quantity <= 0:
            return
            
        # Calculate new cost basis
        old_value = self.quantity * self.cost_basis
        new_value = quantity * price
        new_quantity = self.quantity + quantity
        
        if new_quantity > 0:
            self.cost_basis = (old_value + new_value) / new_quantity
        
        self.quantity = new_quantity
    
    def reduce_quantity(self, quantity, price):
        """Reduce the position."""
        if quantity <= 0 or self.quantity <= 0:
            return
            
        # Cannot reduce by more than current quantity
        actual_reduction = min(quantity, self.quantity)
        
        # Calculate realized profit/loss
        transaction_pnl = actual_reduction * (price - self.cost_basis)
        self.realized_pnl += transaction_pnl
        
        # Reduce quantity
        self.quantity -= actual_reduction
        
        # If position is now zero, reset cost basis
        if self.quantity == 0:
            self.cost_basis = 0.0
    
    def market_value(self, market_price=None):
        """Calculate current market value of position."""
        if market_price is None or self.quantity == 0:
            return 0.0
            
        return self.quantity * market_price
    
    def unrealized_pnl(self, market_price=None):
        """Calculate unrealized profit/loss."""
        if market_price is None or self.quantity == 0:
            return 0.0
            
        return self.quantity * (market_price - self.cost_basis)


class PositionSizerBase(ABC):
    @abstractmethod
    def calculate_position_size(self, symbol, direction, price, portfolio, signal=None):
        pass
    
class PositionSizer:
    """
    Calculates position sizes based on a specified method.
    """
    
    def __init__(self, method='fixed', params=None):
        """
        Initialize the position sizer.
        
        Args:
            method: Sizing method ('fixed', 'percent_risk', 'percent_equity', 'volatility')
            params: Parameters for the selected method
        """
        self.method = method
        self.params = params or {}
    
    def calculate_position_size(self, symbol, direction, price, portfolio, signal_event=None):
        """
        Calculate position size based on the selected method.
        
        Args:
            symbol: Instrument symbol
            direction: Trade direction
            price: Current price
            portfolio: Portfolio object
            signal_event: Original signal event (for confidence-based sizing)
            
        Returns:
            float: Calculated position size
        """
        if self.method == 'fixed':
            return self._fixed_size()
        
        elif self.method == 'percent_equity':
            return self._percent_equity(price, portfolio)
        
        elif self.method == 'percent_risk':
            return self._percent_risk(price, portfolio, symbol)
        
        elif self.method == 'volatility':
            return self._volatility_based(price, symbol, portfolio)
        
        elif self.method == 'confidence':
            return self._confidence_based(price, portfolio, signal_event)
        
        return 0  # Default
    
    def _fixed_size(self):
        """Fixed contract/share size."""
        return self.params.get('shares', 1)
    
    def _percent_equity(self, price, portfolio):
        """Percentage of equity."""
        percent = self.params.get('percent', 0.02)  # Default 2%
        equity = portfolio.get_equity()
        
        return (equity * percent) / price
    
    def _percent_risk(self, price, portfolio, symbol):
        """Percentage of equity risked."""
        risk_percent = self.params.get('risk_percent', 0.01)  # Default 1%
        stop_percent = self.params.get('stop_percent', 0.05)  # Default 5%
        
        equity = portfolio.get_equity()
        risk_amount = equity * risk_percent
        
        # Calculate stop distance
        stop_distance = price * stop_percent
        
        # Position size = risk amount / stop distance
        return risk_amount / stop_distance if stop_distance > 0 else 0    
