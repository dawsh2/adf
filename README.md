# Algorithmic Trading Framework

A modern, event-driven framework for developing, testing, and deploying algorithmic trading strategies.

## Overview

This framework provides a robust foundation for algorithmic trading with a focus on:

- **Modularity**: Components can be combined and reused in different strategies
- **Testability**: Clean interfaces and dependency injection enable comprehensive testing
- **Extensibility**: New data sources, indicators, strategies, and execution methods can be added easily
- **Performance**: Efficient event processing for real-time and backtesting scenarios

## Architecture

The system is built on several core architectural patterns:

### Event-Driven Architecture

The central nervous system of the framework is an event-driven architecture where:

- Components communicate through strongly-typed events
- An event bus provides loose coupling between components
- Specialized event emitters handle specific event types
- Event handlers process events based on their type

### Dependency Injection

All components receive their dependencies rather than creating them:

- A central container manages component creation and lifecycle
- Components declare their dependencies in constructors
- Optional dependencies use setter methods
- Factory methods handle complex initialization

### Component Model

The system is organized around a hierarchical component model:

- **ComponentBase**: The foundation for all components with configuration loading
- **Indicators**: Calculate technical indicators from market data
- **Features**: Derive features from indicators and market data
- **Rules**: Generate trading signals based on conditions
- **Filters**: Control strategy activation based on market conditions
- **Strategies**: Coordinate rules and manage trading decisions

### Configuration Management

A centralized configuration system provides:

- Hierarchical organization of settings
- Type conversion and validation
- Environment-specific configurations
- Sensible defaults with overrides

## Core Modules

### Core

The `core` module provides fundamental infrastructure:

- **Events**: Event types, bus, emitters, and handlers
- **Config**: Configuration management
- **DI**: Dependency injection container
- **Logging**: Centralized logging facilities

### Data

The `data` module handles market data:

- **Sources**: Interfaces to various data providers
- **Handlers**: Historical and real-time data processing
- **Transformers**: Data resampling and normalization

### Models

The `models` module contains analytical components:

- **Components**: Building blocks for strategies (indicators, features, rules)
- **Filters**: Market condition filters including regime detection
- **Optimization**: Parameter optimization tools (grid search, genetic algorithms)
- **ML**: Machine learning model integration
- **RL**: Reinforcement learning model integration
- **Factor**: Factor modeling and analysis

### Strategy

The `strategy` module manages trading decisions:

- **Strategies**: Implementations of trading strategies
- **Risk**: Position sizing and risk management
- **Portfolio**: Asset allocation and portfolio management
- **Ensemble**: Weighted combination of multiple strategies

### Execution

The `execution` module handles order execution:

- **Engine**: Unified execution engine (supports both historical and live data)
- **Live**: Real-time execution with brokers
- **Position**: Position management
- **Portfolio**: Portfolio tracking and management

## Design Principles

### 1. Strong Typing

- All components use explicit type annotations
- Events are passed as strongly-typed objects, never dictionaries
- No unwrapping/wrapping of event data

### 2. Single Responsibility

- Each component has one clear purpose
- Components are focused and cohesive
- Complex functionality is composed from simple components

### 3. Interface Segregation

- Components depend only on the interfaces they use
- Base classes define minimal required methods
- Optional functionality is provided through extension

### 4. Open/Closed Principle

- Components are open for extension, closed for modification
- New functionality is added through inheritance or composition
- Core abstractions remain stable

### 5. Explicit Dependencies

- All dependencies are clearly declared
- No hidden dependencies or global state
- Dependencies can be easily substituted for testing

## Advanced Capabilities

### Optimization

The framework includes advanced optimization capabilities:

- **Grid Search**: Exhaustive search across parameter combinations
- **Genetic Algorithms**: Evolutionary optimization of strategy parameters
- **Walk-Forward Testing**: Time-based validation to prevent overfitting
- **Cross-Validation**: K-fold validation techniques adapted for time series

### Machine Learning Integration

The system supports integration with various machine learning approaches:

- **Supervised Learning**: Classification and regression models for prediction
- **Unsupervised Learning**: Clustering and dimensionality reduction
- **Feature Engineering**: Automated feature generation and selection
- **Model Validation**: Specialized validation for financial time series

### Reinforcement Learning

The framework provides infrastructure for reinforcement learning:

- **Custom Environments**: Financial market environments for RL agents
- **Reward Functions**: Configurable reward functions based on financial metrics
- **Policy Gradient Methods**: Implementation of policy optimization algorithms
- **Deep Q-Networks**: Value-based reinforcement learning

## Usage Examples

### Creating a Simple Strategy

```python
from core.di import default_container
from strategy.strategies import MovingAverageCrossoverStrategy

# Get strategy from container (all dependencies resolved automatically)
strategy = default_container.get('strategy')

# Or create with explicit dependencies
strategy = MovingAverageCrossoverStrategy(
    name='ma_crossover',
    config=config,
    container=container,
    signal_emitter=signal_emitter
)

# Configure strategy parameters
strategy.parameters = {
    'fast_window': 10,
    'slow_window': 30,
    'symbol': 'AAPL'
}

# Initialize and start
strategy.initialize()
```

### Running Historical Data Tests

```python
from core.bootstrap import bootstrap

# Bootstrap the application with historical test configuration
container, config = bootstrap(['config/historical_test.yaml'])

# Get required components
event_manager = container.get('event_manager')
execution_engine = container.get('execution_engine')
data_handler = container.get('data_handler')
strategy = container.get('strategy')

# Set up historical data range
start_date = '2020-01-01'
end_date = '2021-01-01'
symbols = ['AAPL', 'MSFT', 'GOOG']

# Load data and run system
data_handler.load_data(symbols, start_date, end_date)
execution_engine.initialize()

# Process data through event system
for symbol in symbols:
    while True:
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break

# Analyze results
results = execution_engine.get_results()
print(f"Final equity: ${results.get_final_equity():.2f}")
print(f"Sharpe ratio: {results.get_metric('sharpe_ratio'):.2f}")
```

### Creating a Custom Indicator

```python
from models.components import IndicatorBase

class RelativeStrengthIndex(IndicatorBase):
    component_type = "indicators"
    
    @classmethod
    def default_params(cls):
        return {
            'window': 14,
            'price_key': 'close'
        }
    
    def calculate(self, data):
        """Calculate RSI from price data."""
        # Implementation
        pass
    
    def on_bar(self, event):
        """Update indicator value on new bar data."""
        # Implementation
        pass
```

## Development

### Requirements

- Python 3.8+
- Required packages listed in requirements.txt

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/trading-framework.git
cd trading-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Testing

```bash
# Run all tests
python -m pytest

# Run specific test module
python -m pytest tests/test_events.py

# Run with coverage
python -m pytest --cov=src
```

## Project Structure

```
src/
├── core/                       # Core infrastructure
│   ├── events/                 # Event system
│   ├── config/                 # Configuration
│   ├── di/                     # Dependency injection
│   └── logging/                # Logging system
│
├── data/                       # Data handling
│   ├── sources/                # Data sources
│   └── transformers/           # Data transformers
│
├── models/                     # Models & components
│   ├── components/             # Strategy components
│   ├── filters/                # Market condition filters
│   └── optimization/           # Optimization tools
│
├── strategy/                   # Trading strategies
│   ├── strategies/             # Strategy implementations
│   └── risk/                   # Risk management
│
└── execution/                  # Order execution
    ├── backtest/               # Backtesting
    └── live/                   # Live trading
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
