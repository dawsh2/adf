#!/usr/bin/env python
"""
Simple Backtest Script (Modified)

This script runs a complete backtest of a Moving Average Crossover strategy with
a simplified risk management approach that allows only one position at a time.
"""
import datetime
import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pytz  # Add pytz for timezone handling

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import core components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType, Event, SignalEvent, OrderEvent, FillEvent, BarEvent
from src.core.events.event_emitters import BarEmitter
from src.core.events.event_utils import create_fill_event, create_order_event

# Import data components
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler

# Import execution components
from src.execution.brokers.simulated import SimulatedBroker
from src.execution.execution_base import ExecutionEngine
from src.execution.portfolio import PortfolioManager

# Import risk components
from src.strategy.risk.position_sizer import PositionSizer
from src.strategy.risk.risk_manager import SimpleRiskManager

# Import strategy
from src.models.components.base import StrategyBase

class MovingAverageCrossoverStrategy:
    """Simple moving average crossover strategy."""
    
    def __init__(self, name, symbols, fast_window=5, slow_window=15):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            symbols: List of symbols to trade
            fast_window: Fast moving average window
            slow_window: Slow moving average window
        """
        self.name = name
        self.event_bus = None
        
        # Settings
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.fast_window = fast_window
        self.slow_window = slow_window
        
        # Store price history
        self.price_history = {symbol: [] for symbol in self.symbols}
        
        # State
        self.last_ma_values = {symbol: {'fast': None, 'slow': None} for symbol in self.symbols}
        self.signals = []
        
        logger.info(f"Initialized MA Crossover strategy: fast={fast_window}, slow={slow_window}")
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
        return self
    
    def on_bar(self, event):
        """Process a bar event."""
        if not isinstance(event, BarEvent):
            return None
            
        symbol = event.get_symbol()
        if symbol not in self.symbols:
            return None
            
        # Add price to history
        close_price = event.get_close()
        self.price_history[symbol].append(close_price)
        
        logger.debug(f"Adding price {close_price} for {symbol}, history size: {len(self.price_history[symbol])}")
        
        # Keep history manageable
        if len(self.price_history[symbol]) > self.slow_window + 10:
            self.price_history[symbol] = self.price_history[symbol][-(self.slow_window + 10):]
        
        # Check if we have enough data
        if len(self.price_history[symbol]) < self.slow_window:
            return None
            
        # Calculate MAs
        fast_ma = sum(self.price_history[symbol][-self.fast_window:]) / self.fast_window
        slow_ma = sum(self.price_history[symbol][-self.slow_window:]) / self.slow_window
        
        logger.debug(f"MAs for {symbol}: fast={fast_ma:.2f}, slow={slow_ma:.2f}")
        
        # Get previous MA values
        prev_fast = self.last_ma_values[symbol]['fast']
        prev_slow = self.last_ma_values[symbol]['slow']
        
        # Update MA values
        self.last_ma_values[symbol]['fast'] = fast_ma
        self.last_ma_values[symbol]['slow'] = slow_ma
        
        # Skip if no previous values
        if prev_fast is None or prev_slow is None:
            logger.debug(f"No previous MA values for {symbol}, skipping signal generation")
            return None
            
        # Check for crossover
        signal = None
        
        # Buy signal: fast MA crosses above slow MA
        if fast_ma > slow_ma and prev_fast <= prev_slow:
            logger.info(f"BUY SIGNAL: {symbol} fast MA crossed above slow MA")
            signal = SignalEvent(
                signal_value=SignalEvent.BUY,
                price=close_price,
                symbol=symbol,
                rule_id=self.name,
                confidence=1.0,
                metadata={
                    'fast_ma': fast_ma,
                    'slow_ma': slow_ma
                },
                timestamp=event.get_timestamp()
            )
            
        # Sell signal: fast MA crosses below slow MA
        elif fast_ma < slow_ma and prev_fast >= prev_slow:
            logger.info(f"SELL SIGNAL: {symbol} fast MA crossed below slow MA")
            signal = SignalEvent(
                signal_value=SignalEvent.SELL,
                price=close_price,
                symbol=symbol,
                rule_id=self.name,
                confidence=1.0,
                metadata={
                    'fast_ma': fast_ma,
                    'slow_ma': slow_ma
                },
                timestamp=event.get_timestamp()
            )
        
        # Emit signal if generated
        if signal:
            self.signals.append(signal)
            
            if self.event_bus:
                logger.debug(f"Emitting signal: {symbol} {'BUY' if signal.get_signal_value() == SignalEvent.BUY else 'SELL'}")
                self.event_bus.emit(signal)
        
        return signal
    
    def reset(self):
        """Reset the strategy state."""
        self.price_history = {symbol: [] for symbol in self.symbols}
        self.last_ma_values = {symbol: {'fast': None, 'slow': None} for symbol in self.symbols}
        self.signals = []


class SimpleRiskManager:
    """
    Simplified risk manager that handles long and short positions.
    Only allows one position at a time (either long or short).
    """
    
    def __init__(self, portfolio, event_bus=None, fixed_size=100):
        """Initialize the simplified risk manager."""
        self.portfolio = portfolio
        self.event_bus = event_bus
        self.fixed_size = fixed_size
        self.position_state = {symbol: 0 for symbol in ["SPY"]}  # 0=neutral, 1=long, -1=short
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
        return self
    
    def on_signal(self, event):
        """Process a signal event and produce an order if appropriate."""
        if not isinstance(event, SignalEvent):
            return
        
        # Extract signal details
        symbol = event.get_symbol()
        signal_value = event.get_signal_value()
        price = event.get_price()
        
        # Initialize position state if not exists
        if symbol not in self.position_state:
            self.position_state[symbol] = 0
        
        # Get current position state (0=neutral, 1=long, -1=short)
        current_state = self.position_state[symbol]
        
        # BUY signal (signal_value = 1)
        if signal_value == SignalEvent.BUY:
            if current_state == 0:  # If neutral, go long
                # Create BUY order
                order = create_order_event(
                    symbol=symbol,
                    order_type="MARKET",
                    direction="BUY",
                    quantity=self.fixed_size,
                    price=price,
                    timestamp=event.get_timestamp()
                )
                self.position_state[symbol] = 1  # Mark as long
                if self.event_bus:
                    logger.info(f"Creating BUY order for {symbol}: {self.fixed_size} shares @ {price} (neutral → long)")
                    self.event_bus.emit(order)
                    
            elif current_state == -1:  # If short, cover first then go long
                # Close short position first
                cover_order = create_order_event(
                    symbol=symbol,
                    order_type="MARKET",
                    direction="BUY",
                    quantity=self.fixed_size,
                    price=price,
                    timestamp=event.get_timestamp()
                )
                if self.event_bus:
                    logger.info(f"Creating BUY order for {symbol}: {self.fixed_size} shares @ {price} (covering short)")
                    self.event_bus.emit(cover_order)
                
                # Then go long
                buy_order = create_order_event(
                    symbol=symbol,
                    order_type="MARKET",
                    direction="BUY",
                    quantity=self.fixed_size,
                    price=price,
                    timestamp=event.get_timestamp()
                )
                self.position_state[symbol] = 1  # Mark as long
                if self.event_bus:
                    logger.info(f"Creating BUY order for {symbol}: {self.fixed_size} shares @ {price} (going long)")
                    self.event_bus.emit(buy_order)
        
        # SELL signal (signal_value = -1)
        elif signal_value == SignalEvent.SELL:
            if current_state == 1:  # If long, sell to close
                # Create SELL order to close long
                order = create_order_event(
                    symbol=symbol,
                    order_type="MARKET",
                    direction="SELL",
                    quantity=self.fixed_size,
                    price=price,
                    timestamp=event.get_timestamp()
                )
                if self.event_bus:
                    logger.info(f"Creating SELL order for {symbol}: {self.fixed_size} shares @ {price} (closing long)")
                    self.event_bus.emit(order)
                
                # Then go short
                short_order = create_order_event(
                    symbol=symbol,
                    order_type="MARKET",
                    direction="SELL",
                    quantity=self.fixed_size,
                    price=price,
                    timestamp=event.get_timestamp()
                )
                self.position_state[symbol] = -1  # Mark as short
                if self.event_bus:
                    logger.info(f"Creating SELL order for {symbol}: {self.fixed_size} shares @ {price} (going short)")
                    self.event_bus.emit(short_order)
                    
            elif current_state == 0:  # If neutral, go short
                # Create SELL order to go short
                order = create_order_event(
                    symbol=symbol,
                    order_type="MARKET",
                    direction="SELL",
                    quantity=self.fixed_size,
                    price=price,
                    timestamp=event.get_timestamp()
                )
                self.position_state[symbol] = -1  # Mark as short
                if self.event_bus:
                    logger.info(f"Creating SELL order for {symbol}: {self.fixed_size} shares @ {price} (neutral → short)")
                    self.event_bus.emit(order)
    
    def reset(self):
        """Reset the risk manager state."""
        self.position_state = {symbol: 0 for symbol in self.position_state}


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
               timeframe='1m', fast_window=5, slow_window=15, fixed_position_size=100):
    """
    Run a backtest with the given parameters.
    
    Args:
        data_dir: Directory containing data files
        symbols: Symbol or list of symbols to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        timeframe: Data timeframe
        fast_window: Fast moving average window
        slow_window: Slow moving average window
        fixed_position_size: Fixed position size for each trade
        
    Returns:
        tuple: (results, event_tracker, portfolio)
    """
    if symbols is None:
        symbols = ['SPY']
    
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
    
    if start_date and end_date:
        logger.info(f"Using date range: {start_date} to {end_date}")
    
    logger.info(f"Starting backtest for {symbols} from {start_date} to {end_date}")
    
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
            'volume': ['Volume']
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
    
    # Create simplified risk manager that only allows one position at a time
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
    
    # Create moving average crossover strategy
    strategy = MovingAverageCrossoverStrategy(
        name="ma_crossover",
        symbols=symbols,
        fast_window=fast_window,
        slow_window=slow_window
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
            
            # Record equity
            equity_curve.append({
                'timestamp': bar.get_timestamp(),
                'equity': portfolio.get_equity({symbol: bar.get_close()})
            })
            
            bar_count += 1
            
            if bar_count % 100 == 0:
                logger.info(f"Processed {bar_count} bars for {symbol}, Portfolio equity: ${portfolio.get_equity():,.2f}")
        
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
    

    
    return results, tracker, portfolio


if __name__ == "__main__":
    # Use absolute path with tilde expansion
    DATA_DIR = "~/adf/data"  # Path to your data directory
    DATA_DIR = os.path.expanduser(DATA_DIR)  # Expand tilde to full path
    
    print(f"Using data directory: {DATA_DIR}")
    
    # Use a date range that's within your data
    results, tracker, portfolio = run_backtest(
        data_dir=DATA_DIR,
        symbols=["SPY"],
        start_date="2024-03-26",  # Start date
        end_date="2024-04-10",    # End date
        timeframe="1m",
        fast_window=10,
        slow_window=30,
        fixed_position_size=100   # Fixed position size
    )
