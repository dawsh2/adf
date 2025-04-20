import unittest
from unittest.mock import MagicMock, patch
import datetime

from src.core.events.event_types import EventType, OrderEvent, FillEvent
from src.execution.execution_base import ExecutionEngine


class TestExecutionEngine(unittest.TestCase):
    """Test cases for the ExecutionEngine class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock broker interface
        self.broker_interface = MagicMock()
        
        # Create execution engine
        self.execution_engine = ExecutionEngine(self.broker_interface)
        
        # Create mock event bus
        self.event_bus = MagicMock()
        self.execution_engine.set_event_bus(self.event_bus)
    
    def test_on_order(self):
        """Test processing order events."""
        # Create order event
        order = OrderEvent(
            symbol="AAPL",
            order_type="MARKET",
            direction="BUY",
            quantity=100,
            price=150.0,
            timestamp=datetime.datetime.now()
        )
        
        # Process order
        self.execution_engine.on_order(order)
        
        # Check that order was stored and status updated
        order_id = order.get_id()
        self.assertIn(order_id, self.execution_engine.orders)
        self.assertEqual(self.execution_engine.order_status[order_id], "PLACED")
        
        # Verify broker interface was called
        self.broker_interface.place_order.assert_called_once_with(order)
    
    def test_place_order(self):
        """Test placing orders directly."""
        # Create order
        order = OrderEvent(
            symbol="MSFT",
            order_type="LIMIT",
            direction="SELL",
            quantity=50,
            price=250.0,
            timestamp=datetime.datetime.now()
        )
        
        # Place order
        self.execution_engine.place_order(order)
        
        # Verify broker interface was called
        self.broker_interface.place_order.assert_called_once_with(order)
        
        # Check order status
        order_id = order.get_id()
        self.assertEqual(self.execution_engine.order_status[order_id], "PLACED")


    def test_cancel_order(self):
        """Test cancelling orders."""
        # Create and place order
        order = OrderEvent(
            symbol="GOOG",
            order_type="MARKET",
            direction="BUY",
            quantity=10,
            price=1500.0,
            timestamp=datetime.datetime.now()
        )

        # Store the order in the execution engine's state
        self.execution_engine.orders[order.get_id()] = order
        self.execution_engine.order_status[order.get_id()] = "PLACED"

        # Configure broker to return success
        self.broker_interface.cancel_order.return_value = True

        # Cancel order
        result = self.execution_engine.cancel_order(order.get_id())

        # Verify result and broker call
        self.assertTrue(result)
        self.broker_interface.cancel_order.assert_called_once_with(order.get_id())
        self.assertEqual(self.execution_engine.order_status[order.get_id()], "CANCELLING")
        

    def test_get_open_orders(self):
        """Test getting open orders."""
        # Create and place multiple orders
        symbols = ["AAPL", "MSFT", "GOOG"]
        for symbol in symbols:
            order = OrderEvent(
                symbol=symbol,
                order_type="MARKET",
                direction="BUY",
                quantity=100,
                price=100.0,
                timestamp=datetime.datetime.now()
            )
            self.execution_engine.orders[order.get_id()] = order
            self.execution_engine.order_status[order.get_id()] = "PLACED"

        # Set one order to filled status
        filled_order_id = list(self.execution_engine.orders.keys())[0]
        self.execution_engine.order_status[filled_order_id] = "FILLED"

        # Get all open orders
        open_orders = self.execution_engine.get_open_orders()

        # Should have 2 open orders, 1 filled
        self.assertEqual(len(open_orders), 2)

if __name__ == '__main__':
    unittest.main()
