import datetime
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import os

# Set up logging - increase to DEBUG level
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import core components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType, Event, SignalEvent, OrderEvent, FillEvent, BarEvent
from src.core.events.event_utils import create_bar_event, create_signal_event, create_order_event
from src.core.events.event_emitters import BarEmitter

# Import data components
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler

# Import execution components
from src.execution.brokers.simulated import SimulatedBroker
from src.execution.execution_base import ExecutionEngine
from src.execution.portfolio import PortfolioManager

# Import risk components
from src.strategy.risk.position_sizer import PositionSizer
from src.strategy.risk.risk_manager import RiskManager

# Import backtest runner
from backtest_runner import BacktestRunner

# Simple MA Crossover Strategy with position tracking
class SimpleMACrossoverStrategy:
    """Position-aware Moving Average Crossover strategy."""
    
    def __init__(self, name, symbols, fast_window=10, slow_window=20):
        """Initialize the strategy."""
        self.name = name
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.event_bus = None
        
        # Price history for each symbol
        self.price_history = {symbol: [] for symbol in self.symbols}
        
        # Track MA values
        self.ma_values = {
            symbol: {
                'fast': [],
                'slow': []
            } for symbol in self.symbols
        }
        
        # Track positions - we'll update this from fill events
        self.positions = {symbol: 0 for symbol in self.symbols}
        
        # Track generated signals
        self.signals = []
        
        # Track last signal direction to avoid duplicates
        self.last_signal_type = {symbol: None for symbol in self.symbols}
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
        # Register for FILL events to track positions
        if event_bus:
            event_bus.register(EventType.FILL, self.on_fill)
    
    def on_fill(self, event):
        """Track position from fill events."""
        if not isinstance(event, FillEvent):
            return
            
        symbol = event.get_symbol()
        if symbol not in self.symbols:
            return
            
        # Update position
        if event.get_direction() == 'BUY':
            self.positions[symbol] += event.get_quantity()
        elif event.get_direction() == 'SELL':
            self.positions[symbol] -= event.get_quantity()
            
        logger.debug(f"Strategy updated position for {symbol}: {self.positions[symbol]}")
    
    def on_bar(self, event):
        """Process a bar event to generate signals."""
        if not isinstance(event, BarEvent):
            return None
            
        symbol = event.get_symbol()
        if symbol not in self.symbols:
            return None
            
        # Add price to history
        price = event.get_close()
        self.price_history[symbol].append(price)
        
        # Ensure we have enough data
        if len(self.price_history[symbol]) < self.slow_window:
            return None
            
        # Keep history size manageable
        max_history = max(self.fast_window, self.slow_window) * 3
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
        
        # Calculate moving averages
        prices = self.price_history[symbol]
        fast_ma = sum(prices[-self.fast_window:]) / self.fast_window
        slow_ma = sum(prices[-self.slow_window:]) / self.slow_window
        
        # Store MA values
        self.ma_values[symbol]['fast'].append(fast_ma)
        self.ma_values[symbol]['slow'].append(slow_ma)
        
        # Keep MA history manageable too
        if len(self.ma_values[symbol]['fast']) > max_history:
            self.ma_values[symbol]['fast'] = self.ma_values[symbol]['fast'][-max_history:]
            self.ma_values[symbol]['slow'] = self.ma_values[symbol]['slow'][-max_history:]
        
        # Need at least two points to detect crossover
        if len(self.ma_values[symbol]['fast']) < 2:
            return None
            
        # Get current and previous values
        curr_fast = self.ma_values[symbol]['fast'][-1]
        curr_slow = self.ma_values[symbol]['slow'][-1]
        prev_fast = self.ma_values[symbol]['fast'][-2]
        prev_slow = self.ma_values[symbol]['slow'][-2]
        
        # Check for crossovers
        signal = None
        current_position = self.positions[symbol]
        
        # Buy signal: Fast MA crosses above Slow MA (only if we don't have a position)
        if curr_fast > curr_slow and prev_fast <= prev_slow and current_position <= 0:
            # Generate buy signal
            signal = SignalEvent(
                signal_value=SignalEvent.BUY,
                price=price,
                symbol=symbol,
                rule_id=self.name,
                confidence=1.0,
                metadata={
                    'fast_ma': curr_fast,
                    'slow_ma': curr_slow
                },
                timestamp=event.get_timestamp()
            )
            
            # Update last signal type
            self.last_signal_type[symbol] = SignalEvent.BUY
            
        # Sell signal: Fast MA crosses below Slow MA (only if we have a position)
        elif curr_fast < curr_slow and prev_fast >= prev_slow and current_position > 0:
            # Generate sell signal
            signal = SignalEvent(
                signal_value=SignalEvent.SELL,
                price=price,
                symbol=symbol,
                rule_id=self.name,
                confidence=1.0,
                metadata={
                    'fast_ma': curr_fast,
                    'slow_ma': curr_slow
                },
                timestamp=event.get_timestamp()
            )
            
            # Update last signal type
            self.last_signal_type[symbol] = SignalEvent.SELL
        
        # Emit signal if generated
        if signal:
            self.signals.append(signal)
            
            if self.event_bus:
                logger.debug(f"Strategy emitting signal: {symbol} {signal.get_signal_value()} " + 
                            f"(position: {current_position}) MA:{curr_fast:.2f}/{curr_slow:.2f}")
                self.event_bus.emit(signal)
                
        return signal
    
    def reset(self):
        """Reset strategy state."""
        self.price_history = {symbol: [] for symbol in self.symbols}
        self.ma_values = {symbol: {'fast': [], 'slow': []} for symbol in self.symbols}
        self.positions = {symbol: 0 for symbol in self.symbols}
        self.signals = []
        self.last_signal_type = {symbol: None for symbol in self.symbols}

class SimplePositionSizer:
    """Simplified position sizer with fixed trade size."""
    
    def __init__(self, fixed_size=100):
        """Initialize with fixed position size."""
        self.fixed_size = fixed_size
    
    def calculate_position_size(self, symbol, direction, price, portfolio=None, signal=None):
        """Calculate position size (fixed amount)."""
        # Always use the fixed size
        return self.fixed_size

class SimpleRiskManager:
    """Simplified risk manager that doesn't complicate position sizing."""
    
    def __init__(self, event_bus=None):
        """Initialize risk manager."""
        self.event_bus = event_bus
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
    
    def on_signal(self, event):
        """Process signal events into orders."""
        if not isinstance(event, SignalEvent):
            return None
            
        # Extract signal details
        symbol = event.get_symbol()
        signal_value = event.get_signal_value()
        price = event.get_price()
        
        # Determine order direction
        if signal_value == SignalEvent.BUY:
            direction = "BUY"
        elif signal_value == SignalEvent.SELL:
            direction = "SELL"
        else:
            return None  # Ignore neutral signals
        
        # Use simple fixed size
        quantity = 100
        
        # Create order
        order = OrderEvent(
            symbol=symbol,
            order_type="MARKET",
            direction=direction,
            quantity=quantity,
            price=price,
            timestamp=event.get_timestamp()
        )
        
        # Emit order
        if self.event_bus:
            logger.debug(f"Risk manager creating order: {symbol} {direction} {quantity} @ {price:.2f}")
            self.event_bus.emit(order)
            
        return order

def run_fixed_backtest(data_dir='./data', symbols=None, start_date='2024-03-25', end_date='2024-04-05',
                      timeframe='1m', fast_window=5, slow_window=15):
    """Run a fixed backtest with simplified components."""
    if symbols is None:
        symbols = ['SPY']
    
    # Expand user directory
    if data_dir.startswith('~'):
        data_dir = os.path.expanduser(data_dir)
    
    print(f"Starting fixed backtest for {symbols} from {start_date} to {end_date}")
    print(f"Using MA parameters: fast={fast_window}, slow={slow_window}")
    
    # Create event system
    event_bus = EventBus()
    event_manager = EventManager(event_bus)
    
    # Create data source
    data_source = CSVDataSource(
        data_dir=data_dir,
        filename_pattern='{symbol}_{timeframe}.csv',
        date_column='timestamp',
        date_format=None,
        column_map={
            'open': ['Open'],
            'high': ['High'],
            'low': ['Low'],
            'close': ['Close'],
            'volume': ['Volume']
        }
    )
    
    # Create emitters
    bar_emitter = BarEmitter("bar_emitter", event_bus)
    bar_emitter.start()
    
    fill_emitter = BarEmitter("fill_emitter", event_bus)
    fill_emitter.start()
    
    # Create data handler
    data_handler = HistoricalDataHandler(data_source, bar_emitter)
    
    # Fix timezone issues
    try:
        start_dt = pd.to_datetime(start_date).tz_localize('UTC')
        end_dt = pd.to_datetime(end_date).tz_localize('UTC')
        end_dt = end_dt + pd.Timedelta(days=1)
    except Exception as e:
        logger.warning(f"Error processing date strings: {e}")
        start_dt = start_date
        end_dt = end_date
    
    # Load data
    for symbol in symbols:
        data_handler.load_data(symbol, timeframe=timeframe, start_date=start_dt, end_date=end_dt)
    
    # Get initial prices
    initial_prices = {}
    for symbol in symbols:
        bars = data_handler.get_latest_bars(symbol, 1)
        if bars and len(bars) > 0:
            initial_prices[symbol] = bars[0].get_close()
        else:
            initial_prices[symbol] = 100.0
    
    # Create portfolio
    initial_capital = 100000.0
    portfolio = PortfolioManager(initial_cash=initial_capital, event_bus=event_bus)
    
    # Create strategy
    strategy = SimpleMACrossoverStrategy(
        name="ma_crossover",
        symbols=symbols,
        fast_window=fast_window,
        slow_window=slow_window
    )
    strategy.set_event_bus(event_bus)
    
    # Create risk manager
    risk_manager = SimpleRiskManager(event_bus)
    
    # Create broker
    broker = SimulatedBroker(fill_emitter=fill_emitter)
    
    # Set initial market data
    for symbol, price in initial_prices.items():
        logger.debug(f"Setting initial market price for {symbol}: {price}")
        broker.update_market_data(symbol, {"price": price})
    
    # Create execution engine
    execution_engine = ExecutionEngine(broker_interface=broker)
    execution_engine.set_event_bus(event_bus)
    
    # Register components with event manager
    event_manager.register_component('strategy', strategy, [EventType.BAR])
    event_manager.register_component('risk_manager', risk_manager, [EventType.SIGNAL])
    event_manager.register_component('execution', execution_engine, [EventType.ORDER])
    event_manager.register_component('portfolio', portfolio, [EventType.FILL])
    
    # Track events for analysis
    signal_events = []
    order_events = []
    fill_events = []
    
    # Register event trackers
    def track_signal(event):
        if event.get_type() == EventType.SIGNAL:
            signal_events.append(event)
    
    def track_order(event):
        if event.get_type() == EventType.ORDER:
            order_events.append(event)
    
    def track_fill(event):
        if event.get_type() == EventType.FILL:
            fill_events.append(event)
    
    event_bus.register(EventType.SIGNAL, track_signal)
    event_bus.register(EventType.ORDER, track_order)
    event_bus.register(EventType.FILL, track_fill)
    
    # Create backtest runner
    backtest = BacktestRunner(
        data_handler=data_handler,
        strategy=strategy,
        risk_manager=risk_manager,
        execution_engine=execution_engine,
        portfolio=portfolio,
        event_bus=event_bus,
        event_manager=event_manager
    )
    
    # Run backtest
    results = backtest.run(symbols, start_dt, end_dt, timeframe)
    
    # Display results
    print("\n=== Backtest Summary ===")
    print(f"Signal events: {len(signal_events)}")
    print(f"Order events: {len(order_events)}")
    print(f"Fill events: {len(fill_events)}")
    
    # Analyze trades
    if fill_events:
        fills_by_direction = {
            'BUY': [f for f in fill_events if f.get_direction() == 'BUY'],
            'SELL': [f for f in fill_events if f.get_direction() == 'SELL']
        }
        
        print(f"\nTotal buys: {len(fills_by_direction['BUY'])}")
        print(f"Total sells: {len(fills_by_direction['SELL'])}")
        
        # Calculate P&L
        if fills_by_direction['BUY'] and fills_by_direction['SELL']:
            total_buy_value = sum(f.get_price() * f.get_quantity() for f in fills_by_direction['BUY'])
            total_sell_value = sum(f.get_price() * f.get_quantity() for f in fills_by_direction['SELL'])
            
            total_buy_qty = sum(f.get_quantity() for f in fills_by_direction['BUY'])
            total_sell_qty = sum(f.get_quantity() for f in fills_by_direction['SELL'])
            
            avg_buy_price = total_buy_value / total_buy_qty if total_buy_qty > 0 else 0
            avg_sell_price = total_sell_value / total_sell_qty if total_sell_qty > 0 else 0
            
            print(f"Average buy price: ${avg_buy_price:.2f}")
            print(f"Average sell price: ${avg_sell_price:.2f}")
            
            if avg_buy_price > 0 and avg_sell_price > 0:
                profit_per_share = avg_sell_price - avg_buy_price
                total_realized_profit = profit_per_share * min(total_buy_qty, total_sell_qty)
                
                print(f"Profit per share: ${profit_per_share:.2f}")
                print(f"Total realized profit: ${total_realized_profit:.2f}")
    
    # Display portfolio results
    if results:
        print("\nBacktest Results:")
        print(f"Initial Capital: ${results['initial_equity']:,.2f}")
        print(f"Final Equity: ${results['final_equity']:,.2f}")
        print(f"Return: {results['return_pct']:.2f}%")
        print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        
        # Check for final positions
        final_positions = {}
        for symbol in symbols:
            position = portfolio.get_position(symbol)
            qty = position.quantity if position else 0
            final_positions[symbol] = qty
        
        print("\nFinal Positions:")
        for symbol, qty in final_positions.items():
            print(f"{symbol}: {qty} shares")
        
        print(f"Final Cash: ${portfolio.cash:,.2f}")
    
    return results, backtest, {
        'signals': signal_events,
        'orders': order_events,
        'fills': fill_events,
        'event_bus': event_bus
    }

if __name__ == "__main__":
    results, backtest, debug_data = run_fixed_backtest(
        data_dir='~/adf/data',
        symbols=['SPY'],
        start_date='2024-03-25',
        end_date='2024-04-05',
        timeframe='1m',
        fast_window=5,
        slow_window=15
    )
