import unittest
from unittest.mock import MagicMock, patch
import datetime

from src.core.events.event_types import EventType, SignalEvent
from src.strategy.risk.risk_manager import RiskManager
from src.strategy.risk.position_sizer import PositionSizer


class TestRiskManager(unittest.TestCase):
    """Test cases for the RiskManager class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock portfolio
        self.portfolio = MagicMock()
        self.portfolio.get_equity.return_value = 100000.0
        
        # Mock position
        self.position = MagicMock()
        self.position.quantity = 100
        
        # Configure portfolio mock to return position
        self.portfolio.get_position.return_value = self.position
        
        # Create position sizer
        self.position_sizer = PositionSizer(method='fixed', params={'shares': 100})
        
        # Create risk manager with limits
        self.risk_limits = {
            'max_position_size': 500,
            'max_exposure': 0.1,
            'min_trade_size': 10
        }
        self.risk_manager = RiskManager(
            portfolio=self.portfolio,
            position_sizer=self.position_sizer,
            risk_limits=self.risk_limits
        )
        
        # Create mock event bus
        self.event_bus = MagicMock()
        self.risk_manager.set_event_bus(self.event_bus)
    
    def test_on_signal_buy(self):
        """Test processing buy signals."""
        # Create signal event
        signal = SignalEvent(
            signal_value=SignalEvent.BUY,
            price=100.0,
            symbol="AAPL",
            rule_id="test_rule",
            confidence=1.0,
            timestamp=datetime.datetime.now()
        )
        
        # Process signal
        self.risk_manager.on_signal(signal)
        
        # Verify order was created and emitted
        self.event_bus.emit.assert_called_once()
        order = self.event_bus.emit.call_args[0][0]
        
        # Check order details
        self.assertEqual(order.get_symbol(), "AAPL")
        self.assertEqual(order.get_direction(), "BUY")
        self.assertEqual(order.get_quantity(), 100)  # Based on fixed sizer
        self.assertEqual(order.get_price(), 100.0)
    
    def test_on_signal_sell(self):
        """Test processing sell signals."""
        # Create signal event
        signal = SignalEvent(
            signal_value=SignalEvent.SELL,
            price=100.0,
            symbol="AAPL",
            rule_id="test_rule",
            confidence=1.0,
            timestamp=datetime.datetime.now()
        )
        
        # Process signal
        self.risk_manager.on_signal(signal)
        
        # Verify order was created and emitted
        self.event_bus.emit.assert_called_once()
        order = self.event_bus.emit.call_args[0][0]
        
        # Check order details
        self.assertEqual(order.get_symbol(), "AAPL")
        self.assertEqual(order.get_direction(), "SELL")
        self.assertEqual(order.get_quantity(), 100)  # Based on fixed sizer
        self.assertEqual(order.get_price(), 100.0)
    
    def test_apply_risk_limits_max_position(self):
        """Test applying maximum position size limit."""
        # Set up current position
        self.position.quantity = 450  # Already close to max of 500
        
        # Apply risk limits to a buy quantity
        quantity = self.risk_manager._apply_risk_limits("AAPL", "BUY", 100, 100.0)
        
        # Should be limited to 50 more shares to reach max of 500
        self.assertEqual(quantity, 50)
    
    def test_apply_risk_limits_max_exposure(self):
        """Test applying maximum exposure limit."""
        # Apply risk limits to a quantity that would exceed exposure
        # 100,000 * 0.1 = 10,000 max exposure, at $200 that's 50 shares
        quantity = self.risk_manager._apply_risk_limits("AAPL", "BUY", 100, 200.0)
        
        # Should be limited to 50 shares based on exposure limit
        self.assertEqual(quantity, 50)
    
    def test_apply_risk_limits_min_trade(self):
        """Test applying minimum trade size."""
        # Set up position near max to force small remaining quantity
        self.position.quantity = 495  # Only 5 more to reach max of 500
        
        # Apply risk limits
        quantity = self.risk_manager._apply_risk_limits("AAPL", "BUY", 100, 100.0)
        
        # Should be 0 because 5 is below min trade size of 10
        self.assertEqual(quantity, 0)


if __name__ == '__main__':
    unittest.main()
