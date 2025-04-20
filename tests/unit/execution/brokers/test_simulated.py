import unittest
from unittest.mock import MagicMock, patch
import datetime

from src.core.events.event_types import OrderEvent, FillEvent
from src.execution.brokers.broker_base import BrokerBase
from src.execution.brokers.simulated import SimulatedBroker, DefaultSlippageModel


class TestSimulatedBroker(unittest.TestCase):
    """Test cases for the SimulatedBroker class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create fill emitter mock
        self.fill_emitter = MagicMock()
        
        # Create simulated broker
        self.broker = SimulatedBroker(fill_emitter=self.fill_emitter)
        
        # Add market data
        self.broker.update_market_data("AAPL", {"price": 150.0})
        self.broker.update_market_data("MSFT", {"price": 250.0})
    
    def test_place_order_buy(self):
        """Test placing a buy order."""
        # Create order
        order = OrderEvent(
            symbol="AAPL",
            order_type="MARKET",
            direction="BUY",
            quantity=100,
            price=None,  # Market order
            timestamp=datetime.datetime.now()
        )
        
        # Place order
        self.broker.place_order(order)
        
        # Verify order was stored
        self.assertIn(order.get_id(), self.broker.orders)
        
        # Verify fill event was emitted
        self.fill_emitter.emit.assert_called_once()
        fill = self.fill_emitter.emit.call_args[0][0]
        
        # Check fill details
        self.assertEqual(fill.get_symbol(), "AAPL")
        self.assertEqual(fill.get_direction(), "BUY")
        self.assertEqual(fill.get_quantity(), 100)
        
        # Price should include slippage (0.1% higher for buys)
        expected_price = 150.0 * 1.001
        self.assertAlmostEqual(fill.get_price(), expected_price, places=6)
        
        # Commission should be calculated
        self.assertGreater(fill.get_commission(), 0)
    
    def test_place_order_sell(self):
        """Test placing a sell order."""
        # Create order
        order = OrderEvent(
            symbol="MSFT",
            order_type="MARKET",
            direction="SELL",
            quantity=50,
            price=None,  # Market order
            timestamp=datetime.datetime.now()
        )
        
        # Place order
        self.broker.place_order(order)
        
        # Verify fill event was emitted
        self.fill_emitter.emit.assert_called_once()
        fill = self.fill_emitter.emit.call_args[0][0]
        
        # Check fill details
        self.assertEqual(fill.get_symbol(), "MSFT")
        self.assertEqual(fill.get_direction(), "SELL")
        self.assertEqual(fill.get_quantity(), 50)
        
        # Price should include slippage (0.1% lower for sells)
        expected_price = 250.0 * 0.999
        self.assertAlmostEqual(fill.get_price(), expected_price, places=6)
    
    def test_cancel_order(self):
        """Test cancelling an order."""
        # Create and place order
        order = OrderEvent(
            symbol="AAPL",
            order_type="LIMIT",
            direction="BUY",
            quantity=100,
            price=145.0,
            timestamp=datetime.datetime.now()
        )
        self.broker.place_order(order)
        order_id = order.get_id()
        
        # Cancel order
        result = self.broker.cancel_order(order_id)
        
        # Verify order was cancelled
        self.assertTrue(result)
        self.assertNotIn(order_id, self.broker.orders)
    
    def test_custom_slippage_model(self):
        """Test broker with custom slippage model."""
        # Create custom slippage model
        class CustomSlippageModel:
            def apply_slippage(self, price, direction, quantity, symbol=None):
                # 0.5% slippage
                if direction == "BUY":
                    return price * 1.005
                elif direction == "SELL":
                    return price * 0.995
                return price
        
        # Create broker with custom model
        broker = SimulatedBroker(
            slippage_model=CustomSlippageModel(),
            fill_emitter=self.fill_emitter
        )
        broker.update_market_data("AAPL", {"price": 150.0})
        
        # Create order
        order = OrderEvent(
            symbol="AAPL",
            order_type="MARKET",
            direction="BUY",
            quantity=100,
            price=None,
            timestamp=datetime.datetime.now()
        )
        
        # Place order
        broker.place_order(order)
        
        # Verify fill price with custom slippage
        fill = self.fill_emitter.emit.call_args[0][0]
        expected_price = 150.0 * 1.005
        self.assertAlmostEqual(fill.get_price(), expected_price, places=6)
    
    def test_market_data_missing(self):
        """Test behavior when market data is missing."""
        # Reset fill emitter
        self.fill_emitter.reset_mock()
        
        # Create order for symbol with no market data
        order = OrderEvent(
            symbol="GOOG",  # No market data for this
            order_type="MARKET",
            direction="BUY",
            quantity=10,
            price=None,
            timestamp=datetime.datetime.now()
        )
        
        # Place order
        self.broker.place_order(order)
        
        # Should default to price of 100.0
        fill = self.fill_emitter.emit.call_args[0][0]
        expected_price = 100.0 * 1.001  # Default with slippage
        self.assertAlmostEqual(fill.get_price(), expected_price, places=6)


class TestDefaultSlippageModel(unittest.TestCase):
    """Test cases for the DefaultSlippageModel."""
    
    def setUp(self):
        """Set up test environment."""
        self.slippage_model = DefaultSlippageModel()
    
    def test_buy_slippage(self):
        """Test slippage for buy orders."""
        price = 100.0
        slipped_price = self.slippage_model.apply_slippage(price, "BUY", 100)
        
        # Expected: 100 * 1.001 = 100.1
        self.assertAlmostEqual(slipped_price, 100.1, places=6)
    
    def test_sell_slippage(self):
        """Test slippage for sell orders."""
        price = 100.0
        slipped_price = self.slippage_model.apply_slippage(price, "SELL", 100)
        
        # Expected: 100 * 0.999 = 99.9
        self.assertAlmostEqual(slipped_price, 99.9, places=6)


if __name__ == '__main__':
    unittest.main()
