# Data Module for Algorithmic Trading System

## Overview

The Data module serves as the foundation for market data acquisition, processing, and management in our algorithmic trading system. It provides a clean, event-driven pipeline for delivering market data to strategy components through the event system.

## Architecture

The Data module follows these design principles:

1. **Clean Abstractions**: Base classes define clear interfaces for data sources and handlers
2. **Event-Driven Design**: Seamless integration with the event system
3. **Extensibility**: Easy to add new data sources and transformers
4. **Testability**: Components designed for unit and integration testing

### Directory Structure

```
src/data/
├── __init__.py
├── data_handler_base.py         # Abstract base class for data handlers
├── data_source_base.py          # Abstract base class for data sources
├── factory.py                   # Factory for creating data components
├── historical_data_handler.py   # Handler for historical data
├── registry.py                  # Registry of available components
├── sources/                     # Data source implementations
│   ├── __init__.py
│   ├── csv_handler.py           # CSV file data source
│   └── ...
└── transformers/                # Data transformation components
    ├── __init__.py
    ├── normalizer.py            # Data normalization
    ├── resampler.py             # Time-based resampling
    └── ...
```

## Core Components

### DataSourceBase

Abstract base class for all data sources. Implements methods to:
- Get data for a specific symbol and date range
- Check if data is available

```python
# Example implementation
class CustomDataSource(DataSourceBase):
    def get_data(self, symbol, start_date=None, end_date=None, timeframe='1d'):
        # Implementation to get data
        pass
        
    def is_available(self, symbol, start_date=None, end_date=None, timeframe='1d'):
        # Implementation to check availability
        return True
```

### DataHandlerBase

Abstract base class for data handlers that manage how data flows into the system and integrate with the event system.

```python
# Example implementation
class CustomDataHandler(DataHandlerBase):
    def __init__(self, data_source, event_bus=None):
        super().__init__(event_bus)
        self.data_source = data_source
        
    def load_data(self, symbols, start_date=None, end_date=None, timeframe='1d'):
        # Implementation to load data
        pass
        
    def get_next_bar(self, symbol):
        # Implementation to get next bar and emit event
        bar = create_bar_event(...)
        if self.event_bus:
            self.event_bus.emit(bar)
        return bar
        
    def reset(self):
        # Reset handler state
        pass
```

### CSVDataSource

Implementation of `DataSourceBase` for reading data from CSV files.

```python
from src.data.sources.csv_handler import CSVDataSource

# Create a data source for CSV files
data_source = CSVDataSource(
    data_dir='./data',
    filename_pattern='{symbol}_{timeframe}.csv',
    date_column='date',
    date_format='%Y-%m-%d'
)

# Get data for a symbol
df = data_source.get_data('AAPL', start_date='2022-01-01', end_date='2022-12-31', timeframe='1d')
```

### HistoricalDataHandler

Implementation of `DataHandlerBase` for processing historical data in backtests.

```python
from src.data.historical_data_handler import HistoricalDataHandler
from src.core.events.event_bus import EventBus

# Create event bus
event_bus = EventBus()

# Create data handler
data_handler = HistoricalDataHandler(data_source, event_bus)

# Load data
data_handler.load_data(['AAPL', 'MSFT'], start_date='2022-01-01', end_date='2022-12-31')

# Process data bar by bar
while True:
    bar = data_handler.get_next_bar('AAPL')
    if bar is None:
        break
    # Bar events are automatically emitted to the event bus
```

### Data Transformers

Utility classes for transforming data:

#### Resampler

For time-based resampling of data (e.g., 1m → 5m, 1h → 1d)

```python
from src.data.transformers.resampler import Resampler

# Create resampler
resampler = Resampler()

# Resample data
daily_data = df_hourly.copy()
weekly_data = resampler.resample_to_timeframe(daily_data, '1w')
```

#### Normalizer

For data normalization and scaling

```python
from src.data.transformers.normalizer import Normalizer

# Create normalizer
normalizer = Normalizer(method='zscore', columns=['open', 'high', 'low', 'close'])

# Normalize data
normalized_data = normalizer.fit_transform(df)

# Restore to original scale
original_data = normalizer.inverse_transform(normalized_data)
```

## Integration with Event System

The Data module integrates with the event system to provide a seamless flow of market data to strategy components:

1. The `HistoricalDataHandler` emits `BarEvent` objects to the event bus
2. Strategy components registered for `EventType.BAR` receive these events
3. Strategies analyze the data and generate signals
4. Signal events are emitted back to the event bus
5. Execution components act on the signals

This event-driven approach allows for:
- Loose coupling between components
- Flexible strategy development
- Clean separation of data, strategy, and execution concerns

## Example Usage: Running a Backtest

Here's a complete example of using the Data module with the event system for a backtest:

```python
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler
from src.strategy.strategies.ma_crossover import MovingAverageCrossoverStrategy
from src.execution.portfolio import Portfolio
from src.execution.execution_engine import ExecutionEngine

# Create event system
event_bus = EventBus()
event_manager = EventManager(event_bus)

# Create data components
data_source = CSVDataSource('./data')
data_handler = HistoricalDataHandler(data_source, event_bus)

# Create strategy and execution components
strategy = MovingAverageCrossoverStrategy(fast_window=10, slow_window=30)
portfolio = Portfolio(initial_capital=100000.0)
executor = ExecutionEngine(portfolio)

# Register components with event manager
event_manager.register_component('strategy', strategy, [EventType.BAR])
event_manager.register_component('executor', executor, [EventType.SIGNAL])

# Load data
data_handler.load_data(['AAPL'], start_date='2022-01-01', end_date='2022-12-31')

# Run the backtest
for symbol in ['AAPL']:
    while True:
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break

# Get results
print(f"Final portfolio value: ${portfolio.get_summary()['equity']:.2f}")
print(f"Total trades: {len(executor.trades)}")
```

## Testing

The Data module includes a comprehensive test suite:

1. **Unit Tests**: Test individual components in isolation
   - `test_csv_handler.py`
   - `test_resampler.py`
   - `test_normalizer.py`

2. **Integration Tests**: Test component interaction
   - `test_data_event_integration.py`: Tests how data flows through the event system

Run the tests with:

```bash
# Run all tests
./run_tests.sh

# Run just data module tests
./run_tests.sh data

# Run integration tests
./run_tests.sh integration
```

## Extending the System

### Adding a New Data Source

1. Create a new class implementing `DataSourceBase`
2. Implement `get_data` and `is_available` methods
3. Register with the component registry

Example:

```python
from src.data.data_source_base import DataSourceBase

class APIDataSource(DataSourceBase):
    def __init__(self, api_key, api_url):
        self.api_key = api_key
        self.api_url = api_url
        self.session = None
        
    def connect(self):
        # Implementation to connect to API
        pass
        
    def get_data(self, symbol, start_date=None, end_date=None, timeframe='1d'):
        # Implementation to get data from API
        pass
        
    def is_available(self, symbol, start_date=None, end_date=None, timeframe='1d'):
        # Implementation to check if API has the requested data
        pass
```

### Adding a New Transformer

Create a new class in the `transformers` package:

```python
class TechnicalIndicators:
    def __init__(self):
        pass
        
    def add_moving_averages(self, df, windows=[10, 20, 50, 200]):
        """Add moving averages to the dataframe."""
        for window in windows:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
        return df
```

## Future Enhancements

Planned enhancements to the Data module include:

1. **Live Data Support**: Real-time data handlers for connecting to exchanges
2. **WebSocket Integration**: Direct streaming of market data events
3. **Database Support**: Storage and retrieval of historical data from databases
4. **Feature Engineering**: Built-in technical indicators and feature generation
5. **Adaptive Resampling**: Smart resampling based on market volatility

## Conclusion

The Data module provides a solid foundation for your algorithmic trading system by handling the critical task of market data acquisition and processing. Its clean integration with the event system ensures that data flows seamlessly to strategy components, while its extensible design makes it easy to add new data sources and transformers as your needs evolve.
