## 1. Overview

The Algorithmic Trading System uses an event-driven architecture to enable loose coupling between components. This document outlines the Event Emitter pattern implementation across all system modules to standardize event handling and emission.

## 2. Core Architecture Principles

### 2.1 Separation of Concerns

- **Event Generation**: Components determine *when* to create events
- **Event Emission**: Emitters handle *how* events are distributed
- **Event Routing**: The Event Bus manages delivery to registered handlers
- **Event Handling**: Components process events via standardized `on_X` methods

### 2.2 Key Benefits

- Improved testability through abstraction
- Cleaner component interfaces
- Enhanced flexibility for event filtering and transformation
- Consistent event handling patterns across the system
- Simplified dependency injection

## 3. Event Emitter Hierarchy

```
EventEmitter (Base Class)
├── SignalEmitter
├── OrderEmitter
├── BarEmitter
├── FillEmitter
├── PositionEmitter
├── PortfolioEmitter
├── StrategyEmitter
└── MetricEmitter
```

## 4. Standard Implementation

### 4.1 Base Event Emitter

```python
class EventEmitter:
    """Base class for all event emitters."""
    
    def __init__(self, name, event_bus=None):
        self.name = name
        self.event_bus = event_bus
        self.stats = {'emitted': 0, 'errors': 0}
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
    
    def emit(self, event):
        """Emit an event to the event bus."""
        if not self.event_bus:
            logger.warning(f"No event bus set for emitter {self.name}")
            self.stats['errors'] += 1
            return False
            
        try:
            self.event_bus.emit(event)
            self.stats['emitted'] += 1
            return True
        except Exception as e:
            logger.error(f"Error emitting event: {e}", exc_info=True)
            self.stats['errors'] += 1
            return False
```

### 4.2 Specialized Emitter Example

```python
class SignalEmitter(EventEmitter):
    """Emitter for trading signals."""
    
    def emit_signal(self, signal_value, price, symbol, rule_id=None, 
                    confidence=1.0, metadata=None, timestamp=None):
        """Emit a signal event."""
        signal = SignalEvent(
            signal_value=signal_value,
            price=price,
            symbol=symbol,
            rule_id=rule_id,
            confidence=confidence,
            metadata=metadata,
            timestamp=timestamp
        )
        return self.emit(signal)
        
    def emit_buy_signal(self, price, symbol, rule_id=None, 
                       confidence=1.0, metadata=None, timestamp=None):
        """Emit a buy signal."""
        return self.emit_signal(
            SignalEvent.BUY, price, symbol, rule_id, 
            confidence, metadata, timestamp
        )
        
    def emit_sell_signal(self, price, symbol, rule_id=None, 
                        confidence=1.0, metadata=None, timestamp=None):
        """Emit a sell signal."""
        return self.emit_signal(
            SignalEvent.SELL, price, symbol, rule_id, 
            confidence, metadata, timestamp
        )
```

## 5. Component Integration

### 5.1 Component Base Classes

All components that need to emit events should receive appropriate emitters:

```python
class RuleBase(ComponentBase):
    """Base class for trading rules."""
    
    def __init__(self, name, params=None, signal_emitter=None):
        super().__init__(name, params)
        self.signal_emitter = signal_emitter
        self.state = {}
    
    def on_bar(self, event):
        """Process a bar event."""
        # Implementation in subclasses
        pass
```

### 5.2 Standardized Event Handler Methods

All components should use consistent `on_X` methods for event handling:

- `on_bar(event)` - Handle bar data events
- `on_signal(event)` - Handle trading signals
- `on_order(event)` - Handle order events
- `on_fill(event)` - Handle order fill events
- `on_position(event)` - Handle position updates
- `on_portfolio(event)` - Handle portfolio state changes
- `on_metric(event)` - Handle performance metrics

## 6. Event Manager Integration

The EventManager should automatically wire up components based on their `on_X` methods:

```python
def register_component(self, name, component, event_types=None):
    """Register a component with the event manager."""
    self.components[name] = component
    
    # Set event bus on component if it has the attribute
    if hasattr(component, 'set_event_bus'):
        component.set_event_bus(self.event_bus)
    
    # Register component handlers
    if event_types:
        for event_type in event_types:
            # Check for dedicated handler method
            handler_name = f"on_{event_type.name.lower()}"
            if hasattr(component, handler_name):
                handler = getattr(component, handler_name)
                self.event_bus.register(event_type, handler)
```

## 7. Module-Specific Implementation

### 7.1 Data Module

- `BarEmitter` for emitting market data events
- Data handlers should have `BarEmitter` injected
- Transformers receive raw data and emit transformed bars

### 7.2 Models/Components Module

- Rules, indicators, and features should receive appropriate emitters
- Rules use `SignalEmitter` to emit trading signals
- Components should implement `on_bar` for processing market data

### 7.3 Strategy Module

- Strategies receive various emitters based on their responsibilities
- Strategies primarily implement `on_bar` and `on_signal` methods
- Strategy managers use `StrategyEmitter` to emit strategy state changes

### 7.4 Execution Module

- Execution components use `OrderEmitter` and `FillEmitter`
- Portfolio uses `PositionEmitter` and `PortfolioEmitter`
- Execution engine implements handlers for signals and orders

### 7.5 Analytics Module

- Performance tracking components use `MetricEmitter`
- Analytics components primarily implement event handlers

## 8. Testing Approach

### 8.1 Mock Emitters

Create mock emitters for testing components in isolation:

```python
class MockSignalEmitter:
    """Mock signal emitter for testing."""
    
    def __init__(self):
        self.signals = []
    
    def emit_signal(self, signal_value, price, symbol, rule_id=None, 
                   confidence=1.0, metadata=None, timestamp=None):
        """Record emitted signal."""
        self.signals.append({
            'signal_value': signal_value,
            'price': price,
            'symbol': symbol,
            'rule_id': rule_id,
            'confidence': confidence,
            'metadata': metadata,
            'timestamp': timestamp
        })
        return True
        
    def get_signals(self):
        """Get emitted signals."""
        return self.signals
        
    def reset(self):
        """Reset recorded signals."""
        self.signals = []
```

### 8.2 Testing Handler Methods

Test component handlers directly:

```python
def test_rule_on_bar():
    # Create mock emitter
    mock_emitter = MockSignalEmitter()
    
    # Create rule with mock emitter
    rule = MovingAverageCrossoverRule('test_rule', {'fast_window': 10, 'slow_window': 20}, mock_emitter)
    
    # Create bar event
    bar = BarEvent('AAPL', datetime.now(), 100, 101, 99, 100.5, 1000)
    
    # Call handler directly
    rule.on_bar(bar)
    
    # Verify emitted signals
    signals = mock_emitter.get_signals()
    # Test assertions here
```

## 9. Advanced Emitter Features

### 9.1 Filtered Emitters

Emitters can implement filtering logic:

```python
class FilteredSignalEmitter(SignalEmitter):
    """Signal emitter with filtering."""
    
    def __init__(self, name, event_bus=None, min_confidence=0.5):
        super().__init__(name, event_bus)
        self.min_confidence = min_confidence
    
    def emit_signal(self, signal_value, price, symbol, rule_id=None, 
                   confidence=1.0, metadata=None, timestamp=None):
        """Emit signal only if confidence exceeds threshold."""
        if confidence < self.min_confidence:
            return False
            
        return super().emit_signal(
            signal_value, price, symbol, rule_id, 
            confidence, metadata, timestamp
        )
```

### 9.2 Throttled Emitters

Emitters can implement rate limiting:

```python
class ThrottledEmitter(EventEmitter):
    """Emitter with rate limiting."""
    
    def __init__(self, name, event_bus=None, min_interval=1.0):
        super().__init__(name, event_bus)
        self.min_interval = min_interval
        self.last_emit_time = {}  # event_type -> timestamp
    
    def emit(self, event):
        """Emit event if minimum interval has elapsed."""
        now = time.time()
        event_type = event.get_type()
        
        if event_type in self.last_emit_time:
            elapsed = now - self.last_emit_time[event_type]
            if elapsed < self.min_interval:
                return False
                
        result = super().emit(event)
        if result:
            self.last_emit_time[event_type] = now
            
        return result
```

## 10. Implementation Roadmap

1. Create base `EventEmitter` class in `core/events/event_emitters.py`
2. Implement specialized emitters for each event type
3. Update component base classes to use emitters
4. Refactor existing components to use standardized `on_X` methods
5. Update EventManager to auto-detect and register `on_X` methods
6. Create mock emitters for testing
7. Implement advanced emitter features as needed

## 11. Conclusion

The Event Emitter pattern creates a cleaner, more testable event architecture that explicitly separates the concerns of event generation and distribution. By consistently applying this pattern across all system modules, we create a more maintainable and extensible trading system that can better accommodate complex event flows and advanced features as the system evolves.