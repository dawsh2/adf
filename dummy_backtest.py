#!/usr/bin/env python
"""
Dummy Backtest Script

This script runs a backtest using synthetic data and a dummy strategy
to verify the accuracy of the event system and execution logic.
"""
import datetime
import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pytz
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import core components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType, Event, SignalEvent, OrderEvent, FillEvent, BarEvent
from src.core.events.event_emitters import BarEmitter
from src.core.events.event_utils import create_bar_event, create_order_event, create_fill_event

# Import data components
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler

# Import execution components
from src.execution.brokers.simulated import SimulatedBroker
from src.execution.execution_base import ExecutionEngine
from src.execution.portfolio import PortfolioManager

# Import risk components
from src.strategy.risk.risk_manager import SimpleRiskManager


def create_synthetic_data(data_dir, symbol="SYNTH", days=10, bars_per_day=390):
    """
    Create synthetic price data with predictable patterns for testing.
    """
    # Create directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Create timestamps
    start_date = datetime.datetime(2023, 1, 1, 9, 30)  # 9:30 AM
    timestamps = []
    
    for day in range(days):
        day_start = start_date + datetime.timedelta(days=day)
        for bar in range(bars_per_day):
            timestamps.append(day_start + datetime.timedelta(minutes=bar))
    
    # Create price series with simple pattern:
    # Day 1-3: Strong Uptrend
    # Day 4-7: Strong Downtrend
    # Day 8-10: Strong Uptrend again
    
    base_price = 100.0
    prices = []
    
    for day in range(days):
        daily_bars = bars_per_day
        
        # Determine trend direction - make trends stronger for testing
        if day < 3:
            daily_trend = 0.03  # Strong Uptrend
        elif day < 7:
            daily_trend = -0.03  # Strong Downtrend
        else:
            daily_trend = 0.03   # Strong Uptrend
        
        # Create bars for the day
        for bar in range(daily_bars):
            # Add price with small random noise
            price = base_price + (bar * daily_trend)
            # Add very small noise
            price += np.random.normal(0, 0.01)
            prices.append(price)
        
        # Update base price for next day
        base_price = prices[-1]
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'Open': prices,
        'High': [p * 1.001 for p in prices],  # High slightly above price
        'Low': [p * 0.999 for p in prices],   # Low slightly below price
        'Close': prices,
        'Volume': [1000 for _ in prices]
    })
    
    # Save to CSV
    file_path = os.path.join(data_dir, f"{symbol}_1m.csv")
    df.to_csv(file_path, index=False)
    
    print(f"Synthetic data created and saved to {file_path}")
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df

def create_synthetic_data(data_dir, symbol="SYNTH", days=10, bars_per_day=390):
    """
    Create synthetic price data with predictable patterns for testing.
    
    Args:
        data_dir: Directory to save the data
        symbol: Symbol to use
        days: Number of days of data
        bars_per_day: Number of bars per day (390 = 1-minute bars for 6.5 hour trading day)
    """
    # Create directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Create timestamps
    start_date = datetime.datetime(2023, 1, 1, 9, 30)  # 9:30 AM
    timestamps = []
    
    for day in range(days):
        day_start = start_date + datetime.timedelta(days=day)
        for bar in range(bars_per_day):
            timestamps.append(day_start + datetime.timedelta(minutes=bar))
    
    # Create price series with simple pattern:
    # Day 1-3: Uptrend
    # Day 4-7: Downtrend
    # Day 8-10: Uptrend again
    
    base_price = 100.0
    prices = []
    trend = []
    
    for day in range(days):
        daily_bars = bars_per_day
        
        # Determine trend direction
        if day < 3:
            daily_trend = 0.01  # Uptrend
            trend_label = 1     # Bullish
        elif day < 7:
            daily_trend = -0.01  # Downtrend
            trend_label = -1     # Bearish
        else:
            daily_trend = 0.01   # Uptrend
            trend_label = 1      # Bullish
        
        # Create bars for the day
        for bar in range(daily_bars):
            # Add price with small random noise
            price = base_price + (day * daily_bars + bar) * daily_trend
            # Add very small noise
            price += np.random.normal(0, 0.01)
            prices.append(price)
            trend.append(trend_label)
        
        # Update base price for continuity
        base_price = prices[-1]
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'Open': prices,
        'High': [p * 1.001 for p in prices],  # High slightly above price
        'Low': [p * 0.999 for p in prices],   # Low slightly below price
        'Close': prices,
        'Volume': [1000 for _ in prices],
        'trend': trend  # Include true trend for validation
    })
    
    # Save to CSV
    file_path = os.path.join(data_dir, f"{symbol}_1m.csv")
    df.to_csv(file_path, index=False)
    
    print(f"Synthetic data created and saved to {file_path}")
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df

# class DummyStrategy:
#     """
#     Dummy strategy that generates signals based on price movements in synthetic data.
#     Used for validation purposes of the event system.
#     """
    
#     def __init__(self, name, symbols, fast_window=20, slow_window=50):
#         """
#         Initialize the dummy strategy.
        
#         Args:
#             name: Strategy name
#             symbols: List of symbols to trade
#             fast_window: Short-term price window for trend detection
#             slow_window: Long-term price window for trend detection
#         """
#         self.name = name
#         self.event_bus = None
#         self.symbols = symbols if isinstance(symbols, list) else [symbols]
#         self.signals = []
        
#         # Windows for trend detection
#         self.fast_window = fast_window
#         self.slow_window = slow_window
        
#         # Store price history and last signal direction
#         self.price_history = {symbol: [] for symbol in self.symbols}
#         self.last_signal = {symbol: None for symbol in self.symbols}
#         self.ma_values = {symbol: {'fast': None, 'slow': None} for symbol in self.symbols}
        
#         # Debug counter
#         self.bar_count = {symbol: 0 for symbol in self.symbols}
        
#         # Initialization phase - don't generate signals until we have enough data
#         self.initialized = {symbol: False for symbol in self.symbols}
        
#         print(f"Strategy initialized with fast_window={fast_window}, slow_window={slow_window}")
    
#     def set_event_bus(self, event_bus):
#         """Set the event bus."""
#         self.event_bus = event_bus
#         return self
    
#     def _calculate_moving_averages(self, prices):
#         """
#         Calculate moving averages for trend detection.
        
#         Args:
#             prices: List of closing prices
            
#         Returns:
#             dict: Dictionary with fast and slow MA values
#         """
#         if len(prices) < self.slow_window:
#             return {'fast': None, 'slow': None}
            
#         fast_ma = sum(prices[-self.fast_window:]) / self.fast_window
#         slow_ma = sum(prices[-self.slow_window:]) / self.slow_window
        
#         return {'fast': fast_ma, 'slow': slow_ma}
    
#     def on_bar(self, event):
#         """Process a bar event and generate signals based on price movements."""
#         if not isinstance(event, BarEvent):
#             return None
            
#         symbol = event.get_symbol()
#         if symbol not in self.symbols:
#             return None
        
#         # Update bar counter for debugging
#         self.bar_count[symbol] += 1
        
#         # Store price data
#         close_price = event.get_close()
#         self.price_history[symbol].append(close_price)
        
#         # Keep history manageable
#         max_history = self.slow_window + 10
#         if len(self.price_history[symbol]) > max_history:
#             self.price_history[symbol] = self.price_history[symbol][-max_history:]
        
#         # Check if we have enough data
#         if len(self.price_history[symbol]) < self.slow_window:
#             return None
        
#         # Calculate MAs
#         current_mas = self._calculate_moving_averages(self.price_history[symbol])
        
#         # Mark as initialized when we have enough data
#         if not self.initialized[symbol] and current_mas['fast'] is not None and current_mas['slow'] is not None:
#             self.initialized[symbol] = True
#             self.ma_values[symbol] = current_mas
#             print(f"Strategy initialized for {symbol} at bar {self.bar_count[symbol]}")
#             return None
            
#         # Get previous MA values
#         prev_fast = self.ma_values[symbol]['fast']
#         prev_slow = self.ma_values[symbol]['slow']
        
#         # Update MA values
#         self.ma_values[symbol] = current_mas
        
#         # Skip if no previous values or not initialized
#         if prev_fast is None or prev_slow is None or not self.initialized[symbol]:
#             return None
            
#         # Log MA values periodically for debugging
#         if self.bar_count[symbol] % 100 == 0:
#             print(f"DEBUG: Bar {self.bar_count[symbol]} - Symbol: {symbol}, Fast MA: {current_mas['fast']:.2f}, Slow MA: {current_mas['slow']:.2f}")
        
#         # Generate signal on MA crossover
#         signal = None
#         current_fast = current_mas['fast']
#         current_slow = current_mas['slow']
        
#         # Buy signal: fast MA crosses above slow MA
#         if current_fast > current_slow and prev_fast <= prev_slow:
#             signal = SignalEvent(
#                 signal_value=SignalEvent.BUY,
#                 price=close_price,
#                 symbol=symbol,
#                 rule_id=self.name,
#                 confidence=1.0,
#                 metadata={
#                     'fast_ma': current_fast,
#                     'slow_ma': current_slow
#                 },
#                 timestamp=event.get_timestamp()
#             )
#             self.last_signal[symbol] = SignalEvent.BUY
#             print(f"Generated BUY signal for {symbol} at {close_price:.2f}, bar {self.bar_count[symbol]}")
            
#         # Sell signal: fast MA crosses below slow MA
#         elif current_fast < current_slow and prev_fast >= prev_slow:
#             signal = SignalEvent(
#                 signal_value=SignalEvent.SELL,
#                 price=close_price,
#                 symbol=symbol,
#                 rule_id=self.name,
#                 confidence=1.0,
#                 metadata={
#                     'fast_ma': current_fast,
#                     'slow_ma': current_slow
#                 },
#                 timestamp=event.get_timestamp()
#             )
#             self.last_signal[symbol] = SignalEvent.SELL
#             print(f"Generated SELL signal for {symbol} at {close_price:.2f}, bar {self.bar_count[symbol]}")
        
#         # Emit signal if generated
#         if signal:
#             self.signals.append(signal)
#             if self.event_bus:
#                 self.event_bus.emit(signal)
#             return signal
                
#         return None
    
#     def reset(self):
#         """Reset the strategy state."""
#         self.signals = []
#         self.price_history = {symbol: [] for symbol in self.symbols}
#         self.last_signal = {symbol: None for symbol in self.symbols}
#         self.ma_values = {symbol: {'fast': None, 'slow': None} for symbol in self.symbols}
#         self.bar_count = {symbol: 0 for symbol in self.symbols}
#         self.initialized = {symbol: False for symbol in self.symbols}
        
class DummyStrategy:
    """
    Dummy strategy that generates signals based on price movements in synthetic data.
    Used for validation purposes of the event system.
    """
    
    def __init__(self, name, symbols, fast_window=5, slow_window=20):
        """
        Initialize the dummy strategy.
        
        Args:
            name: Strategy name
            symbols: List of symbols to trade
            fast_window: Short-term price window for trend detection
            slow_window: Long-term price window for trend detection
        """
        self.name = name
        self.event_bus = None
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.signals = []
        
        # Windows for trend detection
        self.fast_window = fast_window
        self.slow_window = slow_window
        
        # Store price history and last signal direction
        self.price_history = {symbol: [] for symbol in self.symbols}
        self.last_signal = {symbol: None for symbol in self.symbols}
        
        # Debug counter
        self.bar_count = {symbol: 0 for symbol in self.symbols}
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
        return self
    
    def _calculate_trend(self, prices):
        """
        Calculate trend based on price slope.
        
        Args:
            prices: List of closing prices
            
        Returns:
            int: 1 for uptrend, -1 for downtrend, 0 for no clear trend
        """
        if len(prices) < self.slow_window:
            return 0  # Not enough data
            
        # Calculate short-term and long-term average slopes
        fast_prices = prices[-self.fast_window:]
        fast_slope = (fast_prices[-1] - fast_prices[0]) / len(fast_prices)
        
        slow_prices = prices[-self.slow_window:]
        slow_slope = (slow_prices[-1] - slow_prices[0]) / len(slow_prices)
        
        # Determine trend based on slope relationship
        trend_threshold = 0.01  # Minimum slope to consider a trend
        
        if fast_slope > trend_threshold and fast_slope > slow_slope:
            return 1  # Uptrend
        elif fast_slope < -trend_threshold and fast_slope < slow_slope:
            return -1  # Downtrend
        else:
            return 0  # No clear trend
    
    def on_bar(self, event):
        """Process a bar event and generate signals based on price movements."""
        if not isinstance(event, BarEvent):
            return None
            
        symbol = event.get_symbol()
        if symbol not in self.symbols:
            return None
        
        # Update bar counter for debugging
        self.bar_count[symbol] += 1
        
        # Store price data
        close_price = event.get_close()
        self.price_history[symbol].append(close_price)
        
        # Keep history manageable
        max_history = max(self.fast_window, self.slow_window) + 10
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
        
        # Check if we have enough data
        if len(self.price_history[symbol]) < self.slow_window:
            return None
            
        # Calculate current trend
        current_trend = self._calculate_trend(self.price_history[symbol])
        
        # Log trend periodically for debugging
        if self.bar_count[symbol] % 100 == 0:
            print(f"DEBUG: Bar {self.bar_count[symbol]} - Symbol: {symbol}, Trend: {current_trend}")
        
        # Generate signal on trend change
        signal = None
        
        # Only generate signal if trend has changed or no previous signal
        last_sig = self.last_signal[symbol]
        
        if current_trend == 1 and (last_sig is None or last_sig != SignalEvent.BUY):
            # Uptrend detected - generate buy signal
            signal = SignalEvent(
                signal_value=SignalEvent.BUY,
                price=close_price,
                symbol=symbol,
                rule_id=self.name,
                confidence=1.0,
                metadata={'calculated_trend': current_trend},
                timestamp=event.get_timestamp()
            )
            self.last_signal[symbol] = SignalEvent.BUY
            print(f"Generated BUY signal for {symbol} at {close_price:.2f}, bar {self.bar_count[symbol]}")
            
        elif current_trend == -1 and (last_sig is None or last_sig != SignalEvent.SELL):
            # Downtrend detected - generate sell signal
            signal = SignalEvent(
                signal_value=SignalEvent.SELL,
                price=close_price,
                symbol=symbol,
                rule_id=self.name,
                confidence=1.0,
                metadata={'calculated_trend': current_trend},
                timestamp=event.get_timestamp()
            )
            self.last_signal[symbol] = SignalEvent.SELL
            print(f"Generated SELL signal for {symbol} at {close_price:.2f}, bar {self.bar_count[symbol]}")
        
        # Emit signal if generated
        if signal:
            self.signals.append(signal)
            if self.event_bus:
                self.event_bus.emit(signal)
            return signal
                
        return None
    
    def reset(self):
        """Reset the strategy state."""
        self.signals = []
        self.price_history = {symbol: [] for symbol in self.symbols}
        self.last_signal = {symbol: None for symbol in self.symbols}
        self.bar_count = {symbol: 0 for symbol in self.symbols}
        
class EventTracker:
    """Utility to track events passing through the system."""
    
    def __init__(self):
        self.events = defaultdict(list)
        self.event_counts = defaultdict(int)
    
    def track_event(self, event):
        """Track an event."""
        event_type = event.get_type()
        self.events[event_type].append(event)
        self.event_counts[event_type] += 1
        
        # Log the event
        if event_type == EventType.SIGNAL:
            symbol = event.get_symbol()
            signal_value = event.get_signal_value()
            direction = "BUY" if signal_value == SignalEvent.BUY else "SELL" if signal_value == SignalEvent.SELL else "NEUTRAL"
            logger.debug(f"Signal [{len(self.events[event_type])}]: {symbol} {direction}")
            
        elif event_type == EventType.ORDER:
            symbol = event.get_symbol()
            direction = event.get_direction()
            quantity = event.get_quantity()
            logger.debug(f"Order [{len(self.events[event_type])}]: {symbol} {direction} {quantity}")
            
        elif event_type == EventType.FILL:
            symbol = event.get_symbol()
            direction = event.get_direction()
            quantity = event.get_quantity()
            price = event.get_price()
            logger.debug(f"Fill [{len(self.events[event_type])}]: {symbol} {direction} {quantity} @ {price:.2f}")
    
    def get_summary(self):
        """Get a summary of tracked events."""
        return {
            event_type.name: len(events)
            for event_type, events in self.events.items()
        }


def run_backtest(data_dir, symbols=None, start_date=None, end_date=None,
               timeframe='1m', fixed_position_size=100, use_dummy_strategy=False):
    """
    Run a backtest with the given parameters.
    
    Args:
        data_dir: Directory containing data files
        symbols: Symbol or list of symbols to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        timeframe: Data timeframe
        fixed_position_size: Fixed position size for each trade
        use_dummy_strategy: Whether to use the dummy strategy instead of MA crossover
        
    Returns:
        tuple: (results, event_tracker, portfolio)
    """
    if symbols is None:
        symbols = ['SYNTH']
    
    if isinstance(symbols, str):
        symbols = [symbols]
    
    # Expand tilde in path if present
    if data_dir.startswith('~'):
        data_dir = os.path.expanduser(data_dir)
    
    logger.info(f"Using data directory: {data_dir}")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory '{data_dir}' does not exist!")
        print(f"\nERROR: Data directory '{data_dir}' does not exist!")
        print("Please check the path to your data files.")
        return None, None, None
    
    # List available files to help troubleshoot
    available_files = os.listdir(data_dir)
    logger.info(f"Files in {data_dir}: {available_files}")
    print(f"\nAvailable files in {data_dir}:")
    for file in available_files:
        print(f"  - {file}")
        
    # Check for expected data files
    for symbol in symbols:
        expected_file = os.path.join(data_dir, f"{symbol}_{timeframe}.csv")
        if os.path.exists(expected_file):
            logger.info(f"Found data file: {expected_file}")
        else:
            logger.warning(f"Data file not found: {expected_file}")
            print(f"\nWARNING: Data file {expected_file} not found!")
    
    # Handle timezone-aware date parsing
    eastern = pytz.timezone('US/Eastern')  # NYSE timezone
    
    if start_date:
        if isinstance(start_date, str):
            try:
                start_date = pd.to_datetime(start_date)
                if start_date.tzinfo is None:
                    start_date = eastern.localize(start_date)
            except Exception as e:
                logger.warning(f"Error parsing start_date: {e}")
    
    if end_date:
        if isinstance(end_date, str):
            try:
                end_date = pd.to_datetime(end_date)
                if end_date.tzinfo is None:
                    end_date = eastern.localize(end_date)
            except Exception as e:
                logger.warning(f"Error parsing end_date: {e}")
    
    # --- Setup Event System ---
    
    # Create event system
    event_bus = EventBus(use_weak_refs=False)  # Use strong refs for testing
    event_manager = EventManager(event_bus)
    
    # Create event tracker
    tracker = EventTracker()
    
    # Register event tracking
    for event_type in EventType:
        event_bus.register(event_type, tracker.track_event)
    
    # --- Setup Data Components ---
    
    # Create data source
    data_source = CSVDataSource(
        data_dir=data_dir,
        filename_pattern='{symbol}_{timeframe}.csv',
        date_column='timestamp',
        date_format=None,  # Let pandas auto-detect the format
        column_map={
            'open': ['Open'],
            'high': ['High'],
            'low': ['Low'],
            'close': ['Close'],
            'volume': ['Volume'],
            'trend': ['trend']  # Map trend column for dummy strategy
        }
    )
    
    # Create bar emitter
    bar_emitter = BarEmitter("backtest_bar_emitter", event_bus)
    bar_emitter.start()
    
    # Create data handler
    data_handler = HistoricalDataHandler(data_source, bar_emitter)
    
    # --- Setup Portfolio ---
    
    # Create portfolio with initial capital
    initial_capital = 100000.0
    portfolio = PortfolioManager(initial_cash=initial_capital, event_bus=event_bus)
    
    # --- Setup Risk Manager ---
    
    # Create simplified risk manager that allows long and short positions
    risk_manager = SimpleRiskManager(
        portfolio=portfolio,
        event_bus=event_bus,
        fixed_size=fixed_position_size
    )
    
    # --- Setup Execution Components ---
    
    # Create broker
    broker = SimulatedBroker(fill_emitter=event_bus)  # Direct connection to event bus
    
    # Create execution engine
    execution_engine = ExecutionEngine(broker_interface=broker, event_bus=event_bus)
    
    # --- Setup Strategy ---
    
    # Create strategy based on parameter
    if use_dummy_strategy:
        strategy = DummyStrategy(
            name="dummy_strategy",
            symbols=symbols
        )
    else:
        # Default to MA Crossover for real data
        from ma_crossover_strategy import MovingAverageCrossoverStrategy
        strategy = MovingAverageCrossoverStrategy(
            name="ma_crossover",
            symbols=symbols,
            fast_window=10,
            slow_window=30
        )
    
    strategy.set_event_bus(event_bus)
    
    # Initial market price setup
    for symbol in symbols:
        broker.update_market_data(symbol, {"price": 100.0})  # Default price
    
    # --- Register Components with Event Manager ---
    
    # Register all components
    event_manager.register_component('strategy', strategy, [EventType.BAR])
    event_manager.register_component('risk', risk_manager, [EventType.SIGNAL])
    event_manager.register_component('execution', execution_engine, [EventType.ORDER])
    event_manager.register_component('portfolio', portfolio, [EventType.FILL])
    
    # --- Run Backtest ---
    
    # Load data
    for symbol in symbols:
        data_handler.load_data(symbol, start_date=start_date, end_date=end_date, timeframe=timeframe)
    
    # Process bars for each symbol
    equity_curve = []
    
    # Go through data chronologically
    for symbol in symbols:
        bar_count = 0
        
        # Process bars
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
            
            # Update broker's market data with current price
            broker.update_market_data(symbol, {"price": bar.get_close()})
            
            # Record equity with current price
            market_prices = {symbol: bar.get_close()}
            equity_curve.append({
                'timestamp': bar.get_timestamp(),
                'equity': portfolio.get_equity(market_prices)
            })
            
            bar_count += 1
            
            if bar_count % 100 == 0:
                logger.info(f"Processed {bar_count} bars for {symbol}, Portfolio equity: ${portfolio.get_equity(market_prices):,.2f}")
        
        logger.info(f"Completed processing {bar_count} bars for {symbol}")
        logger.info(f"Final Portfolio Cash: ${portfolio.cash:,.2f}")
    
    # --- Calculate Results ---
    
    # Calculate performance metrics
    equity_df = pd.DataFrame(equity_curve)
    if not equity_df.empty:
        equity_df.set_index('timestamp', inplace=True)
    
    # Extract key metrics
    initial_equity = initial_capital
    final_equity = portfolio.get_equity()
    returns = (final_equity / initial_equity) - 1
    
    # Compile results
    results = {
        'initial_equity': initial_equity,
        'final_equity': final_equity,
        'return': returns,
        'return_pct': returns * 100,
        'trade_count': len(tracker.events[EventType.FILL]),
        'signal_count': len(tracker.events[EventType.SIGNAL]),
        'order_count': len(tracker.events[EventType.ORDER]),
        'equity_curve': equity_df
    }
    
    # --- Print Summary ---
    
    print("\n=== Backtest Summary ===")
    print(f"Initial Equity: ${results['initial_equity']:,.2f}")
    print(f"Final Equity: ${results['final_equity']:,.2f}")
    print(f"Return: {results['return_pct']:.2f}%")
    print(f"Signals Generated: {results['signal_count']}")
    print(f"Orders Placed: {results['order_count']}")
    print(f"Trades Executed: {results['trade_count']}")
    
    # Print event tracker summary
    print("\n=== Event Summary ===")
    for event_type, count in tracker.get_summary().items():
        print(f"{event_type}: {count}")
    
    # Report on portfolio positions
    print("\n=== Portfolio Positions ===")
    positions = portfolio.get_all_positions()
    if positions:
        for symbol, position in positions.items():
            print(f"{symbol}: {position.quantity} shares, Cost basis: ${position.cost_basis:.2f}")
    else:
        print("No positions in portfolio")
    
    # Plot equity curve if enough data
    if results and len(equity_curve) > 0:
        # Create simple equity curve plot
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df.index, equity_df['equity'])
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('equity_curve.png')
        plt.close()
        print("\nEquity curve saved to 'equity_curve.png'")
    
    return results, tracker, portfolio


def run_backtest_with_synthetic_data(data_dir=None, 
                                      symbol="SYNTH", 
                                      days=10,
                                      fixed_position_size=100):
    """
    Run backtest with synthetic data to validate system behavior.
    
    Args:
        data_dir: Directory for data (temporary dir if None)
        symbol: Symbol to use
        days: Number of days of synthetic data
        fixed_position_size: Fixed position size
    """
    # Create temporary directory if needed
    using_temp_dir = False
    if data_dir is None:
        data_dir = tempfile.mkdtemp()
        using_temp_dir = True
    
    print(f"Using directory for synthetic data: {data_dir}")
    
    # Create synthetic data
    create_synthetic_data(data_dir, symbol, days)
    
    # Run backtest with the dummy strategy
    results, tracker, portfolio = run_backtest(
        data_dir=data_dir,
        symbols=[symbol],
        timeframe="1m",
        fixed_position_size=fixed_position_size,
        use_dummy_strategy=True
    )
    
    # Verify the results against expectations
    print("\n=== Validation Results ===")
    
    # In our synthetic data setup, we have 2 trend changes:
    # - Day 3: Uptrend -> Downtrend 
    # - Day 7: Downtrend -> Uptrend
    expected_signals = 2
    actual_signals = len(tracker.events[EventType.SIGNAL])
    
    print(f"Expected signals: {expected_signals}")
    print(f"Actual signals: {actual_signals}")
    
    # Each trend change should generate 2 orders for a position flip (close existing, open new)
    expected_orders = expected_signals * 2
    actual_orders = len(tracker.events[EventType.ORDER])
    
    print(f"Expected orders: {expected_orders}")
    print(f"Actual orders: {actual_orders}")
    
    # Calculate trading performance
    starting_equity = results['initial_equity']
    ending_equity = results['final_equity']
    profit_loss = ending_equity - starting_equity
    
    print(f"Profit/Loss: ${profit_loss:,.2f}")
    
    # Validation logic
    system_working = True
    
    if actual_signals != expected_signals:
        print("\nWARNING: Signal count doesn't match expectations.")
        system_working = False
    
    if actual_orders != expected_orders:
        print("\nWARNING: Order count doesn't match expectations.")
        system_working = False
    
    # With our perfect strategy on synthetic data, we should be profitable
    if profit_loss <= 0:
        print("\nWARNING: Strategy should be profitable with synthetic data and perfect signals.")
        system_working = False
    
    if system_working:
        print("\nSUCCESS: Backtest system appears to be working correctly!")
    else:
        print("\nWARNING: Backtest system may have issues that need investigation.")
    
    # Clean up if using temp dir
    if using_temp_dir:
        import shutil
        shutil.rmtree(data_dir)
    
    return results, tracker, portfolio


if __name__ == "__main__":
    # Run backtest with synthetic data
    results, tracker, portfolio = run_backtest_with_synthetic_data(
        symbol="SYNTH",
        days=10,
        fixed_position_size=100
    )
