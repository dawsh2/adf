import unittest
import datetime
from unittest.mock import MagicMock

from src.core.events.event_types import EventType, Event, SignalEvent
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.strategy.risk.risk_manager import RiskManager
from src.strategy.risk.position_sizer import PositionSizer
from src.execution.execution_base import ExecutionEngine
from src.execution.brokers.simulated import SimulatedBroker
from src.execution.portfolio import PortfolioManager


class TestExecutionFlow(unittest.TestCase):
    """Integration tests for the execution flow."""
    
    def setUp(self):
        """Set up test environment."""
        # Create event system
        self.event_bus = EventBus(use_weak_refs=False)  # Use strong refs for testing
        self.event_manager = EventManager(self.event_bus)
        
        # Create portfolio
        self.portfolio = PortfolioManager(initial_cash=100000.0)
        
        # Create position sizer
        self.position_sizer = PositionSizer(method='fixed', params={'shares': 100})
        
        # Create risk manager
        self.risk_manager = RiskManager(
            portfolio=self.portfolio,
            position_sizer=self.position_sizer,
            risk_limits={
                'max_position_size': 500,
                'max_exposure': 0.2,
                'min_trade_size': 10
            }
        )
        
        # Create broker
        self.broker = SimulatedBroker()
        
        # Create execution engine
        self.execution_engine = ExecutionEngine(broker_interface=self.broker)
        
        # Register components with event manager
        self.event_manager.register_component('risk', self.risk_manager, [EventType.SIGNAL])
        self.event_manager.register_component('execution', self.execution_engine, [EventType.ORDER])
        self.event_manager.register_component('portfolio', self.portfolio, [EventType.FILL])
        
        # Set up market data for broker
        self.broker.update_market_data("AAPL", {"price": 150.0})
    
    def test_full_execution_flow(self):
        """Test the full execution flow from signal to portfolio update."""
        # Create signal event
        signal = SignalEvent(
            signal_value=SignalEvent.BUY,
            price=150.0,
            symbol="AAPL",
            rule_id="test_rule",
            confidence=1.0,
            timestamp=datetime.datetime.now()
        )
        
        # Initial portfolio state
        initial_cash = self.portfolio.cash
        
        # Emit signal event
        self.event_bus.emit(signal)
        
        # Verify portfolio was updated
        position = self.portfolio.get_position("AAPL")
        self.assertIsNotNone(position)
        self.assertEqual(position.quantity, 100)  # Based on fixed position sizer
        
        # Verify cash was reduced
        # Cash reduction calculation:
        # 100 shares * ~150.15 (price with slippage) + commission
        self.assertLess(self.portfolio.cash, initial_cash)
        
        # Approximate check (since exact commission may vary)
        expected_cash_reduction = 100 * 150.0 * 1.001 + 10  # Approx commission
        self.assertAlmostEqual(
            initial_cash - self.portfolio.cash,
            expected_cash_reduction,
            delta=20  # Allow for rounding differences
        )


if __name__ == '__main__':
    unittest.main()
