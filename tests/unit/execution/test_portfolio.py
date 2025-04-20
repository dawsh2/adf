import unittest
from unittest.mock import MagicMock, patch
import datetime

from src.core.events.event_types import FillEvent
from src.execution.portfolio import PortfolioManager
from src.execution.position import Position


class TestPosition(unittest.TestCase):
    """Test cases for the Position class."""
    
    def setUp(self):
        """Set up test environment."""
        self.position = Position("AAPL")
    
    def test_add_quantity(self):
        """Test adding to a position."""
        # Initial position
        self.position.add_quantity(100, 150.0)
        
        # Check position state
        self.assertEqual(self.position.quantity, 100)
        self.assertEqual(self.position.cost_basis, 150.0)
        
        # Add more at different price
        self.position.add_quantity(50, 160.0)
        
        # Check updated state
        self.assertEqual(self.position.quantity, 150)
        # New cost basis: (100*150 + 50*160) / 150 = 153.33
        self.assertAlmostEqual(self.position.cost_basis, 153.33, places=2)
    
    def test_reduce_quantity(self):
        """Test reducing a position."""
        # Initial position
        self.position.add_quantity(100, 150.0)
        
        # Reduce position
        self.position.reduce_quantity(40, 160.0)
        
        # Check position state
        self.assertEqual(self.position.quantity, 60)
        self.assertEqual(self.position.cost_basis, 150.0)  # Cost basis unchanged
        
        # Check realized P&L: 40 * (160 - 150) = 400
        self.assertEqual(self.position.realized_pnl, 400.0)
    
    def test_reduce_to_zero(self):
        """Test reducing a position to zero."""
        # Initial position
        self.position.add_quantity(100, 150.0)
        
        # Reduce position to zero
        self.position.reduce_quantity(100, 160.0)
        
        # Check position state
        self.assertEqual(self.position.quantity, 0)
        self.assertEqual(self.position.cost_basis, 0.0)  # Reset when position is zero
        
        # Check realized P&L: 100 * (160 - 150) = 1000
        self.assertEqual(self.position.realized_pnl, 1000.0)
    
    def test_market_value(self):
        """Test market value calculation."""
        # Create position
        self.position.add_quantity(100, 150.0)
        
        # Calculate market value
        value = self.position.market_value(155.0)
        
        # Check value: 100 * 155 = 15500
        self.assertEqual(value, 15500.0)
    
    def test_unrealized_pnl(self):
        """Test unrealized P&L calculation."""
        # Create position
        self.position.add_quantity(100, 150.0)
        
        # Calculate unrealized P&L
        pnl = self.position.unrealized_pnl(155.0)
        
        # Check P&L: 100 * (155 - 150) = 500
        self.assertEqual(pnl, 500.0)
    
    def test_total_pnl(self):
        """Test total P&L calculation."""
        # Create position
        self.position.add_quantity(100, 150.0)
        
        # Reduce partially with profit
        self.position.reduce_quantity(40, 160.0)
        
        # Calculate total P&L with current market price
        pnl = self.position.total_pnl(155.0)
        
        # Expected: Realized 400 + Unrealized 60 * (155 - 150) = 700
        self.assertEqual(pnl, 700.0)


class TestPortfolioManager(unittest.TestCase):
    """Test cases for the PortfolioManager class."""
    
    def setUp(self):
        """Set up test environment."""
        self.initial_cash = 100000.0
        self.portfolio = PortfolioManager(initial_cash=self.initial_cash)
    
    def test_on_fill_buy(self):
        """Test processing buy fill events."""
        # Create fill event
        fill = FillEvent(
            symbol="AAPL",
            direction="BUY",
            quantity=100,
            price=150.0,
            commission=7.5,
            timestamp=datetime.datetime.now()
        )
        
        # Process fill
        self.portfolio.on_fill(fill)
        
        # Check position
        position = self.portfolio.get_position("AAPL")
        self.assertIsNotNone(position)
        self.assertEqual(position.quantity, 100)
        self.assertEqual(position.cost_basis, 150.0)
        
        # Check cash: 100000 - (100 * 150) - 7.5 = 84992.5
        self.assertEqual(self.portfolio.cash, 84992.5)
    
    def test_on_fill_sell(self):
        """Test processing sell fill events."""
        # First buy to establish position
        buy_fill = FillEvent(
            symbol="AAPL",
            direction="BUY",
            quantity=100,
            price=150.0,
            commission=7.5,
            timestamp=datetime.datetime.now()
        )
        self.portfolio.on_fill(buy_fill)
        
        # Then sell part of position
        sell_fill = FillEvent(
            symbol="AAPL",
            direction="SELL",
            quantity=40,
            price=160.0,
            commission=5.0,
            timestamp=datetime.datetime.now()
        )
        self.portfolio.on_fill(sell_fill)
        
        # Check position
        position = self.portfolio.get_position("AAPL")
        self.assertEqual(position.quantity, 60)
        
        # Check cash: 84992.5 + (40 * 160) - 5 = 91387.5
        self.assertEqual(self.portfolio.cash, 91387.5)
    
    def test_get_equity(self):
        """Test equity calculation."""
        # Add a position
        fill = FillEvent(
            symbol="AAPL",
            direction="BUY",
            quantity=100,
            price=150.0,
            commission=7.5,
            timestamp=datetime.datetime.now()
        )
        self.portfolio.on_fill(fill)
        
        # Calculate equity with current market prices
        market_prices = {"AAPL": 155.0}
        equity = self.portfolio.get_equity(market_prices)
        
        # Expected: Cash 84992.5 + Position value (100 * 155) = 100492.5
        self.assertEqual(equity, 100492.5)
    
    def test_multiple_positions(self):
        """Test handling multiple positions."""
        # Create fills for different symbols
        fills = [
            FillEvent(
                symbol="AAPL",
                direction="BUY",
                quantity=100,
                price=150.0,
                commission=7.5,
                timestamp=datetime.datetime.now()
            ),
            FillEvent(
                symbol="MSFT",
                direction="BUY",
                quantity=50,
                price=250.0,
                commission=7.5,
                timestamp=datetime.datetime.now()
            )
        ]
        
        # Process fills
        for fill in fills:
            self.portfolio.on_fill(fill)
        
        # Check positions
        positions = self.portfolio.get_all_positions()
        self.assertEqual(len(positions), 2)
        self.assertIn("AAPL", positions)
        self.assertIn("MSFT", positions)
        
        # Check cash: 100000 - (100 * 150) - 7.5 - (50 * 250) - 7.5 = 62485
        self.assertEqual(self.portfolio.cash, 72485.0)
    
    def test_reset(self):
        """Test portfolio reset."""
        # Add a position
        fill = FillEvent(
            symbol="AAPL",
            direction="BUY",
            quantity=100,
            price=150.0,
            commission=7.5,
            timestamp=datetime.datetime.now()
        )
        self.portfolio.on_fill(fill)
        
        # Reset portfolio
        self.portfolio.reset()
        
        # Check that cash is restored and positions cleared
        self.assertEqual(self.portfolio.cash, self.initial_cash)
        self.assertEqual(len(self.portfolio.positions), 0)
        self.assertEqual(len(self.portfolio.fill_history), 0)


if __name__ == '__main__':
    unittest.main()
