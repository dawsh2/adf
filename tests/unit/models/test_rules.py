import unittest
import datetime
from unittest.mock import MagicMock, patch

from src.models.components.base import RuleBase
from src.core.events.event_types import BarEvent, SignalEvent


class TestRuleBase(unittest.TestCase):
    """Test cases for RuleBase class."""
    
    def test_initialization(self):
        """Test initialization of rule base class."""
        # Create mock config and emitter
        mock_config = MagicMock()
        mock_emitter = MagicMock()
        
        # Create mock rule
        rule = MockRule("test_rule", config=mock_config, signal_emitter=mock_emitter)
        
        # Verify properties
        self.assertEqual(rule.name, "test_rule")
        self.assertEqual(rule.config, mock_config)
        self.assertEqual(rule.signal_emitter, mock_emitter)
        self.assertEqual(rule.state, {})
        self.assertEqual(rule.component_type, "rules")
    
    def test_set_event_bus(self):
        """Test setting event bus."""
        # Create rule
        rule = MockRule("test_rule")
        
        # Create mock event bus
        mock_event_bus = MagicMock()
        
        # Set event bus
        rule.set_event_bus(mock_event_bus)
        
        # Verify event bus is set
        self.assertEqual(rule.event_bus, mock_event_bus)
    
    def test_reset(self):
        """Test resetting rule state."""
        # Create rule
        rule = MockRule("test_rule")
        
        # Set some state
        rule.state = {"symbol": "test_data"}
        
        # Reset rule
        rule.reset()
        
        # Verify state is cleared
        self.assertEqual(rule.state, {})


class TestMockRule(unittest.TestCase):
    """Test cases for MockRule implementation."""
    
    def setUp(self):
        """Set up test environment."""
        self.signal_emitter = MagicMock()
        self.rule = MockRule("test_rule", signal_emitter=self.signal_emitter)
    
    def test_on_bar_with_insufficient_data(self):
        """Test rule with insufficient data."""
        # Create bar event
        bar = BarEvent(
            symbol="AAPL",
            timestamp=datetime.datetime.now(),
            open_price=100.0,
            high_price=105.0,
            low_price=95.0,
            close_price=102.0,
            volume=1000
        )
        
        # Process bar
        result = self.rule.on_bar(bar)
        
        # Verify no signal is generated
        self.assertIsNone(result)
        self.signal_emitter.emit.assert_not_called()
    
    def test_on_bar_with_threshold_crossing(self):
        """Test rule with price crossing above threshold."""
        # Create rule with threshold
        rule = MockRule("test_rule", params={"threshold": 100.0}, signal_emitter=self.signal_emitter)
        
        # Add state data
        rule.state = {
            "AAPL": {
                "last_price": 95.0,
                "crossed_above": False,
                "crossed_below": False
            }
        }
        
        # Create bar event with price above threshold
        bar = BarEvent(
            symbol="AAPL",
            timestamp=datetime.datetime.now(),
            open_price=98.0,
            high_price=105.0,
            low_price=97.0,
            close_price=102.0,
            volume=1000
        )
        
        # Process bar
        result = rule.on_bar(bar)
        
        # Verify signal is generated
        self.assertIsNotNone(result)
        self.assertEqual(result.get_signal_value(), SignalEvent.BUY)
        self.assertEqual(result.get_symbol(), "AAPL")
        self.assertEqual(result.get_price(), 102.0)
        
        # Verify emitter is called
        self.signal_emitter.emit.assert_called_once()
    
    def test_on_bar_with_threshold_crossing_below(self):
        """Test rule with price crossing below threshold."""
        # Create rule with threshold
        rule = MockRule("test_rule", params={"threshold": 100.0}, signal_emitter=self.signal_emitter)
        
        # Add state data
        rule.state = {
            "AAPL": {
                "last_price": 105.0,
                "crossed_above": False,
                "crossed_below": False
            }
        }
        
        # Create bar event with price below threshold
        bar = BarEvent(
            symbol="AAPL",
            timestamp=datetime.datetime.now(),
            open_price=102.0,
            high_price=103.0,
            low_price=95.0,
            close_price=98.0,
            volume=1000
        )
        
        # Process bar
        result = rule.on_bar(bar)
        
        # Verify signal is generated
        self.assertIsNotNone(result)
        self.assertEqual(result.get_signal_value(), SignalEvent.SELL)
        self.assertEqual(result.get_symbol(), "AAPL")
        self.assertEqual(result.get_price(), 98.0)
        
        # Verify emitter is called
        self.signal_emitter.emit.assert_called_once()
    
    def test_on_bar_no_crossing(self):
        """Test rule with no threshold crossing."""
        # Create rule with threshold
        rule = MockRule("test_rule", params={"threshold": 100.0}, signal_emitter=self.signal_emitter)
        
        # Add state data (already above threshold)
        rule.state = {
            "AAPL": {
                "last_price": 105.0,
                "crossed_above": True,
                "crossed_below": False
            }
        }
        
        # Create bar event with price still above threshold
        bar = BarEvent(
            symbol="AAPL",
            timestamp=datetime.datetime.now(),
            open_price=104.0,
            high_price=106.0,
            low_price=102.0,
            close_price=103.0,
            volume=1000
        )
        
        # Process bar
        result = rule.on_bar(bar)
        
        # Verify no signal is generated
        self.assertIsNone(result)
        self.signal_emitter.emit.assert_not_called()
    
    def test_multiple_symbols(self):
        """Test rule with multiple symbols."""
        # Create rule with threshold
        rule = MockRule("test_rule", params={"threshold": 100.0}, signal_emitter=self.signal_emitter)
        
        # Add state data for multiple symbols
        rule.state = {
            "AAPL": {
                "last_price": 95.0,
                "crossed_above": False,
                "crossed_below": False
            },
            "MSFT": {
                "last_price": 105.0,
                "crossed_above": False,
                "crossed_below": False
            }
        }
        
        # Process AAPL bar (crosses above)
        aapl_bar = BarEvent(
            symbol="AAPL",
            timestamp=datetime.datetime.now(),
            open_price=98.0,
            high_price=105.0,
            low_price=97.0,
            close_price=102.0,
            volume=1000
        )
        rule.on_bar(aapl_bar)
        
        # Process MSFT bar (crosses below)
        msft_bar = BarEvent(
            symbol="MSFT",
            timestamp=datetime.datetime.now(),
            open_price=102.0,
            high_price=103.0,
            low_price=95.0,
            close_price=98.0,
            volume=1000
        )
        rule.on_bar(msft_bar)
        
        # Verify both signals were emitted
        self.assertEqual(self.signal_emitter.emit.call_count, 2)
        
        # Verify state is updated correctly
        self.assertTrue(rule.state["AAPL"]["crossed_above"])
        self.assertFalse(rule.state["AAPL"]["crossed_below"])
        self.assertEqual(rule.state["AAPL"]["last_price"], 102.0)
        
        self.assertFalse(rule.state["MSFT"]["crossed_above"])
        self.assertTrue(rule.state["MSFT"]["crossed_below"])
        self.assertEqual(rule.state["MSFT"]["last_price"], 98.0)


# Mock rule implementation for testing
class MockRule(RuleBase):
    """Mock rule implementation for testing."""
    
    def __init__(self, name, config=None, container=None, signal_emitter=None, params=None):
        """Initialize the mock rule."""
        # Set default params if not provided
        params = params or {"threshold": 100.0}
        
        # Call parent constructor
        super().__init__(name, config, container, signal_emitter)
        
        # Override params if provided
        if params:
            self.params = params
    
    def on_bar(self, event):
        """
        Process a bar event to generate signals.
        
        This mock rule generates:
        - BUY signal when price crosses above threshold
        - SELL signal when price crosses below threshold
        """
        if not isinstance(event, BarEvent):
            return None
            
        symbol = event.get_symbol()
        price = event.get_close()
        threshold = self.params.get("threshold", 100.0)
        
        # Initialize state for symbol if needed
        if symbol not in self.state:
            self.state[symbol] = {
                "last_price": None,
                "crossed_above": False,
                "crossed_below": False
            }
            
        # Get previous state
        last_price = self.state[symbol]["last_price"]
        crossed_above = self.state[symbol]["crossed_above"]
        crossed_below = self.state[symbol]["crossed_below"]
        
        # Update state with current price
        self.state[symbol]["last_price"] = price
        
        # Skip if no previous price
        if last_price is None:
            return None
        
        # Generate signals on threshold crossing
        signal = None
        
        # Crossing above threshold
        if price > threshold and last_price <= threshold and not crossed_above:
            self.state[symbol]["crossed_above"] = True
            self.state[symbol]["crossed_below"] = False
            
            # Create buy signal
            signal = SignalEvent(
                signal_value=SignalEvent.BUY,
                price=price,
                symbol=symbol,
                rule_id=self.name,
                confidence=1.0,
                metadata={"threshold": threshold},
                timestamp=event.get_timestamp()
            )
            
        # Crossing below threshold
        elif price < threshold and last_price >= threshold and not crossed_below:
            self.state[symbol]["crossed_above"] = False
            self.state[symbol]["crossed_below"] = True
            
            # Create sell signal
            signal = SignalEvent(
                signal_value=SignalEvent.SELL,
                price=price,
                symbol=symbol,
                rule_id=self.name,
                confidence=1.0,
                metadata={"threshold": threshold},
                timestamp=event.get_timestamp()
            )
        
        # Emit signal if generated
        if signal and self.signal_emitter:
            self.signal_emitter.emit(signal)
            
        return signal


if __name__ == '__main__':
    unittest.main()
