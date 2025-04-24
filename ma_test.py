"""
MA Crossover Strategy Validation Test

This script creates synthetic data with clear trends that should be profitable for 
moving average crossover strategies, then tests the strategy with known good parameters.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import strategy components
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler
from src.core.events.event_bus import EventBus
from src.core.events.event_emitters import BarEmitter
from src.strategy.strategies.ma_crossover import MovingAverageCrossoverStrategy
from src.execution.backtest.backtest import run_backtest
from src.analytics.performance import PerformanceAnalytics
from src.core.events.event_types import Event, EventType, SignalEvent, BarEvent
from src.core.events.event_utils import create_signal_event

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_synthetic_data():
    """Create synthetic data with clear trends for testing."""
    
    # Create directory for test data if it doesn't exist
    if not os.path.exists('test_data'):
        os.makedirs('test_data')
    
    # Parameters
    n_days = 200
    seed = 42
    
    np.random.seed(seed)
    
    # Start date
    start_date = datetime(2023, 1, 1)
    
    # Create dates
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Create a trending price series with some noise
    # Start with uptrend, then downtrend, then uptrend again
    base_price = 100.0
    prices = []
    
    # Up trend (first 30%)
    for i in range(int(n_days * 0.3)):
        trend = 0.5  # daily trend
        noise = np.random.normal(0, 0.3)  # Random noise
        change = trend + noise
        base_price *= (1 + change/100)
        prices.append(base_price)
    
    # Down trend (next 40%)
    for i in range(int(n_days * 0.3), int(n_days * 0.7)):
        trend = -0.4  # daily trend
        noise = np.random.normal(0, 0.3)  # Random noise
        change = trend + noise
        base_price *= (1 + change/100)
        prices.append(base_price)
    
    # Up trend again (remaining 30%)
    for i in range(int(n_days * 0.7), n_days):
        trend = 0.6  # daily trend
        noise = np.random.normal(0, 0.3)  # Random noise
        change = trend + noise
        base_price *= (1 + change/100)
        prices.append(base_price)
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
        'close': [p * (1 + np.random.uniform(-0.005, 0.005)) for p in prices],
        'volume': [int(np.random.uniform(1000, 10000)) for _ in prices]
    })
    
    # Add timestamp
    df['timestamp'] = dates
    
    # Save to CSV
    output_path = 'test_data/SYNTH_1d.csv'
    df.to_csv(output_path, index=False)
    
    # Plot the price chart to visualize
    plt.figure(figsize=(12, 6))
    plt.plot(dates, df['close'])
    plt.title('Synthetic Price Data with Clear Trends')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.savefig('test_data/synthetic_price_chart.png')
    plt.close()
    
    logger.info(f"Synthetic test data created and saved to {output_path}")
    logger.info(f"Price chart saved to test_data/synthetic_price_chart.png")
    
    return output_path

class BuyAndHoldStrategy:
    """Simple buy-and-hold strategy for benchmark comparison."""
    
    def __init__(self, name="buy_and_hold", symbols=None):
        self.name = name
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.bought = {symbol: False for symbol in self.symbols}
        self.event_bus = None
    
    def set_event_bus(self, event_bus):
        self.event_bus = event_bus
        return self
    
    def on_bar(self, event):
        if not isinstance(event, BarEvent):
            return None
            
        symbol = event.get_symbol()
        
        if symbol not in self.symbols:
            return None
            
        # Buy once at the beginning and hold
        if not self.bought[symbol]:
            # Create buy signal
            signal = create_signal_event(
                signal_value=SignalEvent.BUY,
                price=event.get_close(),
                symbol=symbol,
                rule_id=self.name,
                timestamp=event.get_timestamp()
            )
            
            # Mark as bought
            self.bought[symbol] = True
            
            # Emit if we have an event bus
            if self.event_bus:
                self.event_bus.emit(signal)
                
            return signal
            
        return None
    
    def reset(self):
        self.bought = {symbol: False for symbol in self.symbols}
    
    def get_parameters(self):
        return {}
    
    def set_parameters(self, params):
        pass

def track_event(event):
    """Track and log important events."""
    if event.get_type() == EventType.SIGNAL:
        signal_value = event.get_signal_value()
        price = event.get_price()
        timestamp = event.get_timestamp()
        direction = "BUY" if signal_value == SignalEvent.BUY else "SELL"
        logger.info(f"SIGNAL: {direction} at {price:.2f} on {timestamp}")
    
    elif event.get_type() == EventType.FILL:
        symbol = event.get_symbol()
        direction = event.get_direction()
        quantity = event.get_quantity()
        price = event.get_price()
        timestamp = event.get_timestamp()
        logger.info(f"FILL: {direction} {quantity} {symbol} at {price:.2f} on {timestamp}")

def setup_data_handler(data_path):
    """Set up data handler with the specified data file."""
    data_dir = os.path.dirname(data_path)
    filename = os.path.basename(data_path)
    symbol = 'SYNTH'  # Synthetic data symbol
    
    # Create data source
    data_source = CSVDataSource(
        data_dir=data_dir, 
        filename_pattern=filename
    )
    
    # Set up event bus and bar emitter
    event_bus = EventBus()
    bar_emitter = BarEmitter(name="bar_emitter", event_bus=event_bus)
    bar_emitter.start()
    
    # Create data handler
    data_handler = HistoricalDataHandler(
        data_source=data_source,
        bar_emitter=bar_emitter
    )
    
    # Load data
    data_handler.load_data(symbols=[symbol])
    
    # Verify data was loaded successfully
    if symbol not in data_handler.data_frames:
        raise ValueError(f"Failed to load data for {symbol}")
    
    # Print data information
    df = data_handler.data_frames[symbol]
    logger.info(f"Loaded {len(df)} bars for {symbol}")
    if hasattr(df.index, 'min') and hasattr(df.index, 'max'):
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Register event tracking
    event_bus.register(EventType.SIGNAL, track_event)
    event_bus.register(EventType.FILL, track_event)
    
    return data_handler, event_bus, symbol

def test_ma_crossover():
    """Test MA crossover strategy on synthetic data."""
    # Create synthetic data
    data_path = create_synthetic_data()
    
    # Set up data handler
    data_handler, event_bus, symbol = setup_data_handler(data_path)
    
    # Initial portfolio cash
    initial_cash = 10000.0
    
    # Test multiple parameter sets to find what works
    parameter_sets = [
        (5, 20),    # Short-term
        (10, 30),   # Medium-term 
        (20, 50),   # Long-term
        (5, 50)     # Wide gap
    ]
    
    results = []
    
    # Test each parameter set
    for fast_window, slow_window in parameter_sets:
        logger.info(f"\nTesting MA({fast_window}, {slow_window})")
        
        # Create strategy
        strategy = MovingAverageCrossoverStrategy(
            name=f"ma_{fast_window}_{slow_window}",
            symbols=[symbol],
            fast_window=fast_window,
            slow_window=slow_window
        )
        
        # Set the event bus
        strategy.set_event_bus(event_bus)
        
        # Reset data handler
        data_handler.reset()
        
        # Run backtest
        equity_curve, trades = run_backtest(
            component=strategy,
            data_handler=data_handler,
            initial_cash=initial_cash,
            debug=True
        )
        
        # Calculate metrics
        metrics = PerformanceAnalytics.calculate_metrics(equity_curve, trades)
        
        # Store results
        result = {
            'fast_window': fast_window,
            'slow_window': slow_window,
            'trade_count': len(trades),
            'return': metrics.get('total_return', 0),
            'sharpe': metrics.get('sharpe_ratio', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'win_rate': metrics.get('win_rate', 0),
            'profit_factor': metrics.get('profit_factor', 0),
            'equity_curve': equity_curve,
            'trades': trades
        }
        results.append(result)
        
        # Print results
        logger.info(f"Total Return: {result['return']:.2f}%")
        logger.info(f"Sharpe Ratio: {result['sharpe']:.2f}")
        logger.info(f"Max Drawdown: {result['max_drawdown']:.2f}%")
        logger.info(f"Total Trades: {result['trade_count']}")
        logger.info(f"Win Rate: {result['win_rate']:.2f}%")
        logger.info(f"Profit Factor: {result['profit_factor']:.2f}")
        
        # Plot equity curve for this parameter set
        if len(equity_curve) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(equity_curve['timestamp'], equity_curve['equity'])
            plt.title(f'Equity Curve: MA({fast_window}, {slow_window})')
            plt.xlabel('Date')
            plt.ylabel('Equity ($)')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'test_data/equity_curve_ma_{fast_window}_{slow_window}.png')
            plt.close()
    
    # Run buy-and-hold benchmark
    benchmark = BuyAndHoldStrategy(name="buy_and_hold", symbols=[symbol])
    benchmark.set_event_bus(event_bus)
    data_handler.reset()
    
    equity_curve_bh, trades_bh = run_backtest(
        component=benchmark,
        data_handler=data_handler,
        initial_cash=initial_cash
    )
    
    metrics_bh = PerformanceAnalytics.calculate_metrics(equity_curve_bh, trades_bh)
    
    logger.info("\nBuy and Hold Benchmark:")
    logger.info(f"Return: {metrics_bh.get('total_return', 0):.2f}%")
    logger.info(f"Sharpe: {metrics_bh.get('sharpe_ratio', 0):.2f}")
    logger.info(f"Max Drawdown: {metrics_bh.get('max_drawdown', 0):.2f}%")
    
    # Sort results by return
    results.sort(key=lambda x: x['return'], reverse=True)
    
    # Print summary 
    logger.info("\nResults Summary (sorted by Return):")
    for i, r in enumerate(results, 1):
        logger.info(f"{i}. MA({r['fast_window']}, {r['slow_window']}): "
                   f"Return: {r['return']:.2f}%, "
                   f"Sharpe: {r['sharpe']:.2f}, "
                   f"Trades: {r['trade_count']}, "
                   f"Win Rate: {r['win_rate']:.2f}%")
    
    # Compare best strategy with buy-and-hold
    if results:
        best = results[0]
        logger.info("\nBest Strategy vs Buy-and-Hold:")
        logger.info(f"Best Strategy: MA({best['fast_window']}, {best['slow_window']}): "
                   f"Return: {best['return']:.2f}%, Sharpe: {best['sharpe']:.2f}")
        logger.info(f"Buy and Hold: Return: {metrics_bh.get('total_return', 0):.2f}%, "
                   f"Sharpe: {metrics_bh.get('sharpe_ratio', 0):.2f}")
        
        # Compare equity curves
        plt.figure(figsize=(12, 6))
        plt.plot(best['equity_curve']['timestamp'], best['equity_curve']['equity'], 
                label=f"MA({best['fast_window']}, {best['slow_window']})")
        plt.plot(equity_curve_bh['timestamp'], equity_curve_bh['equity'], 
                label="Buy and Hold")
        plt.title('Strategy Comparison')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('test_data/strategy_comparison.png')
        plt.close()
        
        # Print first few trades from best strategy
        logger.info("\nSample trades from best strategy:")
        for i, trade in enumerate(best['trades'][:5]):
            logger.info(f"Trade {i+1}: {trade}")
        
    return results

if __name__ == "__main__":
    logger.info("Starting MA Crossover validation test")
    test_ma_crossover()
    logger.info("Test complete")
