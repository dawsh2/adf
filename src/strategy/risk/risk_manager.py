class RiskManagerBase(ABC):
    @abstractmethod
    def evaluate_trade(self, symbol, direction, quantity, price):
        pass

class RiskManager:
    """
    Evaluates trade requests against risk rules.
    Calculates position sizes based on risk parameters.
    """
    
    def __init__(self, portfolio, position_sizer, risk_limits=None):
        self.portfolio = portfolio
        self.position_sizer = position_sizer
        self.risk_limits = risk_limits or {}
    
    def calculate_order_details(self, symbol, direction, price, signal_event=None):
        """
        Calculate order details including quantity based on risk rules.
        
        Returns:
            dict: Order details including quantity, or None if trade rejected
        """
        # Get current portfolio state
        current_position = self.portfolio.positions.get(symbol, None)
        position_qty = current_position.quantity if current_position else 0
        
        # Check if trade would exceed maximum position size
        max_position_size = self.risk_limits.get('max_position_size', float('inf'))
        
        # Calculate desired position size from position sizer
        desired_quantity = self.position_sizer.calculate_position_size(
            symbol, direction, price, self.portfolio, signal_event
        )
        
        # Apply risk limits
        if direction == OrderEvent.BUY:
            # Check if buying would exceed max position
            new_position_size = position_qty + desired_quantity
            if new_position_size > max_position_size:
                desired_quantity = max(0, max_position_size - position_qty)
                
            # Check if buying would exceed max portfolio exposure
            max_exposure = self.risk_limits.get('max_exposure', 1.0)
            portfolio_value = self.portfolio.get_equity()
            trade_value = desired_quantity * price
            
            if trade_value > portfolio_value * max_exposure:
                scaled_quantity = (portfolio_value * max_exposure) / price
                desired_quantity = min(desired_quantity, scaled_quantity)
        
        # Final check for minimum trade size
        min_trade_size = self.risk_limits.get('min_trade_size', 0)
        if desired_quantity < min_trade_size:
            return None  # Reject trade if too small
        
        return {
            'quantity': desired_quantity,
            'order_type': OrderEvent.MARKET,
            'price': price
        }
