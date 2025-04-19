# Configuration & Dependency Injection Guide

## Introduction

This guide introduces two powerful architectural patterns that will help maintain our algorithmic trading system as it grows in complexity:

1. **Centralized Configuration Management**
2. **Dependency Injection (DI)**

These patterns are widely used in enterprise software development to manage complexity, improve testability, and enhance maintainability. This guide explains what these patterns are, why they're valuable, and how to implement them in our codebase.

## 1. Centralized Configuration Management

### What Is It?

A configuration system provides a centralized way to manage all settings and parameters across the application. Instead of hard-coding values or scattering them throughout the codebase, we define them in a central location with a clear hierarchy.

### Why Is It Valuable?

- **Consistency**: All components access configuration in a standard way
- **Flexibility**: Easily change settings without modifying code
- **Environment Support**: Different settings for development, testing, production
- **Transparency**: Clear view of all system parameters
- **Validation**: Centralized validation and type conversion

### How To Use It

#### Basic Usage

```python
from core.config import default_config

# Get a simple configuration value
data_dir = default_config.get_section('data').get('data_dir')

# Get a typed value
max_bars = default_config.get_section('data').get_int('max_bars_history', 100)

# Check a boolean flag
debug_mode = default_config.get_section('core').get_bool('debug', False)
```

#### Component Configuration

For components that need multiple configuration values:

```python
class DataHandler:
    def __init__(self, config):
        # Get all settings from the relevant section
        data_config = config.get_section('data')
        self.data_dir = data_config.get('data_dir')
        self.date_format = data_config.get('date_format', '%Y-%m-%d')
        self.max_bars_history = data_config.get_int('max_bars_history', 100)
```

#### Defining Default Configurations

Register default configurations in `core/bootstrap.py`:

```python
def register_default_configs(config):
    # Core system defaults
    config.register_defaults('core', {
        'log_level': 'INFO',
        'debug': False,
    })
    
    # Data handling defaults
    config.register_defaults('data', {
        'data_dir': './data',
        'date_format': '%Y-%m-%d',
        'max_bars_history': 100,
    })
    
    # Strategy defaults
    config.register_defaults('strategy', {
        'default_fast_ma': 10,
        'default_slow_ma': 30,
    })
```

#### Configuration Files

Create YAML or JSON configuration files:

```yaml
# config/default.yaml
core:
  log_level: INFO
  
data:
  data_dir: ./data
  date_format: '%Y-%m-%d'
  
strategy:
  default_fast_ma: 10
  default_slow_ma: 30
```

Load configuration files in your bootstrap process:

```python
# Load default configurations
config.load_file('config/default.yaml')

# Load environment-specific configurations (overrides defaults)
env = os.getenv('ENV', 'development')
config.load_file(f'config/{env}.yaml')

# Load local configurations (overrides everything)
if os.path.exists('config/local.yaml'):
    config.load_file('config/local.yaml')
```

## 2. Dependency Injection

### What Is It?

Dependency Injection (DI) is a design pattern where objects receive their dependencies from external sources rather than creating them internally. A DI container manages object creation and wiring.

### Why Is It Valuable?

- **Decoupling**: Components don't create their dependencies
- **Testability**: Easy to substitute mock implementations
- **Lifecycle Management**: Container handles object creation and lifecycle
- **Centralized Wiring**: Dependencies defined in one place
- **Consistency**: Standard approach for component creation

### How To Use It

#### Basic Container Usage

```python
from core.di import default_container

# Get a component (container resolves all dependencies)
event_bus = default_container.get('event_bus')
data_handler = default_container.get('data_handler')
strategy = default_container.get('strategy')
```

#### Registering Components

Components should be registered during the bootstrap process:

```python
def register_core_components(container, config):
    # Register event system
    container.register('event_bus', EventBus)
    container.register('event_manager', EventManager, {
        'event_bus': 'event_bus'
    })
    
    # Register logging components
    container.register('logging_handler', LoggingHandler, {
        'name': 'main_logger',
        'log_level': config.get_section('core').get('log_level')
    })
```

#### Component Implementation

Components should accept their dependencies through constructors:

```python
class HistoricalDataHandler:
    def __init__(self, data_source, event_bus, max_bars_history=100):
        self.data_source = data_source
        self.event_bus = event_bus
        self.max_bars_history = max_bars_history
        self.data_frames = {}
```

For optional dependencies, use setter methods:

```python
class Strategy:
    def __init__(self, name):
        self.name = name
        self.event_bus = None
    
    def set_event_bus(self, event_bus):
        """Set the event bus (optional dependency)."""
        self.event_bus = event_bus
```

#### Factory Methods

For complex object creation, use factory methods:

```python
def create_strategy(container):
    """Factory function for creating strategy based on configuration."""
    config = container.get('config')
    strategy_config = config.get_section('strategy')
    
    strategy_type = strategy_config.get('type', 'ma_crossover')
    
    if strategy_type == 'ma_crossover':
        return MovingAverageCrossoverStrategy(
            fast_window=strategy_config.get_int('fast_ma'),
            slow_window=strategy_config.get_int('slow_ma'),
            event_bus=container.get('event_bus')
        )
    elif strategy_type == 'mean_reversion':
        return MeanReversionStrategy(
            lookback=strategy_config.get_int('lookback'),
            threshold=strategy_config.get_float('threshold'),
            event_bus=container.get('event_bus')
        )
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

# Register with a factory
container.register_factory('strategy', create_strategy)
```

## Integration with Existing Codebase

### Bootstrap Process

Create a bootstrap module to initialize the configuration and DI container:

```python
# src/core/bootstrap.py

def bootstrap(config_files=None):
    """Bootstrap the application."""
    # Initialize configuration
    config = Config()
    
    # Register defaults
    register_default_configs(config)
    
    # Load configuration files
    if config_files:
        for file in config_files:
            if os.path.exists(file):
                config.load_file(file)
    
    # Load environment variables
    config.load_env(prefix='TRADING_')
    
    # Initialize container
    container = Container()
    
    # Register components
    container.register_instance('config', config)
    register_core_components(container, config)
    register_data_components(container, config)
    register_strategy_components(container, config)
    register_execution_components(container, config)
    
    return container, config
```

### Entry Point

Use the bootstrap function in your main entry point:

```python
from core.bootstrap import bootstrap

def main():
    # Bootstrap the application
    container, config = bootstrap([
        'config/default.yaml',
        f'config/{os.getenv("ENV", "development")}.yaml',
        'config/local.yaml'
    ])
    
    # Get required components
    event_manager = container.get('event_manager')
    data_handler = container.get('data_handler')
    strategy = container.get('strategy')
    execution_engine = container.get('execution_engine')
    
    # Initialize and run the system
    # ...
```

## Best Practices for Complex Systems

In addition to Configuration and DI, here are other architectural patterns that can help manage complexity in large systems:

### 1. Command Query Responsibility Segregation (CQRS)

Separate operations that change state (commands) from operations that read state (queries). This simplifies complex domain models and improves performance.

```python
# Command handler for changing state
class OrderCommandHandler:
    def __init__(self, order_repository):
        self.order_repository = order_repository
    
    def create_order(self, order_command):
        # Validate command
        # Create order entity
        # Save to repository
        pass
    
    def cancel_order(self, cancel_command):
        # Validate command
        # Get order
        # Change state
        # Save
        pass

# Query handler for reading state
class OrderQueryHandler:
    def __init__(self, order_repository):
        self.order_repository = order_repository
    
    def get_order(self, order_id):
        return self.order_repository.get_by_id(order_id)
    
    def get_orders_by_status(self, status):
        return self.order_repository.get_by_status(status)
```

### 2. Repository Pattern

Abstract away data access with repositories, providing a collection-like interface to domain entities.

```python
class StrategyRepository:
    def __init__(self, db_connection):
        self.db_connection = db_connection
    
    def get_by_id(self, strategy_id):
        # Get strategy from database
        pass
    
    def get_all(self):
        # Get all strategies
        pass
    
    def save(self, strategy):
        # Save strategy to database
        pass
    
    def delete(self, strategy_id):
        # Delete strategy
        pass
```

### 3. Unit of Work Pattern

Maintain a list of objects affected by a business transaction and coordinate the persistence of changes.

```python
class UnitOfWork:
    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.session = None
    
    def __enter__(self):
        self.session = self.session_factory()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.rollback()
        else:
            self.commit()
        self.session.close()
    
    def commit(self):
        self.session.commit()
    
    def rollback(self):
        self.session.rollback()
```

### 4. Circuit Breaker Pattern

Prevents cascading failures in distributed systems by detecting failures and encapsulating the logic of handling them.

```python
class CircuitBreaker:
    def __init__(self, failure_threshold, recovery_timeout):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.state = "CLOSED"
        self.last_failure_time = None
    
    def execute(self, func, *args, **kwargs):
        if self.state == "OPEN":
            # Check if recovery timeout has elapsed
            if (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = "HALF-OPEN"
            else:
                raise CircuitBreakerOpenException("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            
            # Reset on success if in half-open state
            if self.state == "HALF-OPEN":
                self.reset()
                
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
                
            raise e
    
    def reset(self):
        self.failures = 0
        self.state = "CLOSED"
        self.last_failure_time = None
```

### 5. Mediator Pattern

Define an object that encapsulates how objects interact, promoting loose coupling by keeping objects from referring to each other explicitly.

```python
class Mediator:
    def __init__(self):
        self.components = {}
    
    def register(self, name, component):
        self.components[name] = component
        component.mediator = self
    
    def notify(self, sender, event, data=None):
        for name, component in self.components.items():
            if component != sender:
                component.handle(event, data)
```

### 6. Event Sourcing

Store all changes to application state as a sequence of events rather than just the current state.

```python
class EventStore:
    def __init__(self, db_connection):
        self.db_connection = db_connection
    
    def append_events(self, aggregate_id, events, expected_version):
        # Check optimistic concurrency
        current_version = self.get_aggregate_version(aggregate_id)
        if current_version != expected_version:
            raise ConcurrencyException()
        
        # Append events
        for event in events:
            self.db_connection.execute(
                "INSERT INTO events (aggregate_id, type, data, version) VALUES (?, ?, ?, ?)",
                (aggregate_id, event.type, event.serialize(), current_version + 1)
            )
            current_version += 1
    
    def get_events(self, aggregate_id):
        # Get all events for aggregate
        cursor = self.db_connection.execute(
            "SELECT type, data FROM events WHERE aggregate_id = ? ORDER BY version",
            (aggregate_id,)
        )
        
        events = []
        for row in cursor:
            event_type, event_data = row
            events.append(self.deserialize_event(event_type, event_data))
            
        return events
```

## Conclusion

Implementing centralized Configuration Management and Dependency Injection provides a solid foundation for maintaining a complex algorithmic trading system. As the system grows, consider adopting additional architectural patterns to manage specific aspects of complexity.

Remember these core principles:

1. **Single Responsibility**: Each component should have one reason to change
2. **Loose Coupling**: Components should depend on abstractions, not concrete implementations
3. **Explicit Dependencies**: Dependencies should be obvious from constructors or setters
4. **Testability**: All components should be easily testable in isolation
5. **Configuration Over Code**: Use configuration for values that might change

By following these principles and leveraging the architectural patterns described in this guide, we can build a system that remains maintainable and extensible as it grows in complexity.
