import datetime
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
# Set up logging - increase to DEBUG level
logging.basicConfig(level=logging.DEBUG,  # Changed from INFO to DEBUG
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Import core components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType, Event, SignalEvent, OrderEvent, FillEvent
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
# Import strategy
from ma_crossover_strategy import MovingAverageCrossoverStrategy
# Import backtest runner
from backtest_runner import BacktestRunner

# Create debug wrappers for critical components
class DebugRiskManager:
    """Debugging wrapper for RiskManager"""
    def __init__(self, risk_manager):
        self.risk_manager = risk_manager
        
    def __getattr__(self, name):
        return getattr(self.risk_manager, name)
    
    def on_signal(self, event):
        logger.debug(f"RISK MANAGER received signal: {event.get_symbol()} {event.get_signal_value()} at {event.get_timestamp()}")
        
        # Calculate position size before processing
        symbol = event.get_symbol()
        signal_value = event.get_signal_value()
        price = event.get_price()
        
        direction = 'BUY' if signal_value == 1 else 'SELL' if signal_value == -1 else 'NEUTRAL'
        
        # Get position size calculation
        try:
            quantity = self.risk_manager.position_sizer.calculate_position_size(
                symbol, direction, price, self.risk_manager.portfolio, event
            )
            logger.debug(f"Position sizer calculated quantity: {quantity}")
            
            # Apply risk limits
            adjusted_quantity = self.risk_manager._apply_risk_limits(symbol, direction, quantity, price)
            logger.debug(f"After risk limits, adjusted quantity: {adjusted_quantity}")
            
            # Then call the original method
            result = self.risk_manager.on_signal(event)
            
            # Log the result
            logger.debug(f"Risk manager processed signal, result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in risk manager: {e}", exc_info=True)
            return None

class DebugExecutionEngine:
    """Debugging wrapper for ExecutionEngine"""
    def __init__(self, execution_engine):
        self.execution_engine = execution_engine
        
    def __getattr__(self, name):
        return getattr(self.execution_engine, name)
    
    def on_order(self, event):
        logger.debug(f"EXECUTION ENGINE received order: {event.get_symbol()} {event.get_direction()} {event.get_quantity()} at {event.get_price()}")
        
        # Then call the original method
        result = self.execution_engine.on_order(event)
        
        # Log the result
        logger.debug(f"Execution engine processed order, result: {result}")
        return result

class EventBusTracer:
    """Tracing wrapper for EventBus"""
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.emitted_events = {event_type: [] for event_type in EventType}
        
    def __getattr__(self, name):
        return getattr(self.event_bus, name)
    
    def emit(self, event):
        event_type = event.get_type()
        logger.debug(f"EVENT BUS: Emitting {event_type.name} event")
        
        # Store the event
        self.emitted_events[event_type].append(event)
        
        # Call the original method
        result = self.event_bus.emit(event)
        
        # Log the result
        logger.debug(f"Event bus emitted event, event count for {event_type.name}: {len(self.emitted_events[event_type])}")
        return result

def run_debug_backtest(data_dir='./data', symbols=None, start_date='2024-03-25', end_date='2024-04-05',
                timeframe='1m', fast_window=5, slow_window=15):
    """Run a backtest with extensive debugging."""
    if symbols is None:
        symbols = ['SPY']
    
    # Expand user directory if tilde is present
    if data_dir.startswith('~'):
        data_dir = os.path.expanduser(data_dir)
    
    logger.info(f"Starting debug backtest for {symbols} from {start_date} to {end_date}")
    logger.info(f"Using MA parameters: fast={fast_window}, slow={slow_window}")
    
    # Create event system with tracing
    event_bus = EventBus()
    traced_bus = EventBusTracer(event_bus)  # Wrap with tracer
    event_manager = EventManager(traced_bus)
    
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
    
    # Create bar emitter
    bar_emitter = BarEmitter("backtest_bar_emitter", traced_bus)
    bar_emitter.start()
    
    # Create data handler
    data_handler = HistoricalDataHandler(data_source, bar_emitter)
    
    # Fix timezone issues
    try:
        start_dt = pd.to_datetime(start_date).tz_localize('UTC')
        end_dt = pd.to_datetime(end_date).tz_localize('UTC')
        end_dt = end_dt + pd.Timedelta(days=1)
        logger.info(f"Using date range: {start_dt} to {end_dt} (UTC)")
    except Exception as e:
        logger.warning(f"Error processing date strings: {e}")
        start_dt = start_date
        end_dt = end_date
    
    # Load data
    logger.info(f"Loading data for {symbols}...")
    for symbol in symbols:
        data_handler.load_data(symbol, timeframe=timeframe, start_date=start_dt, end_date=end_dt)
    
    # Create portfolio
    initial_capital = 100000.0
    portfolio = PortfolioManager(initial_cash=initial_capital)
    
    # Create position sizer with higher position size for testing
    position_sizer = PositionSizer(method='fixed', params={'shares': 100})  # Fixed 100 shares
    
    # Create risk manager with relaxed constraints
    risk_manager = RiskManager(
        portfolio=portfolio,
        position_sizer=position_sizer,
        risk_limits={
            'max_position_size': 1000,  # Very high limit
            'max_exposure': 1.0,        # Allow up to 100% exposure
            'min_trade_size': 1         # Allow even small trades
        },
        event_bus=traced_bus  # Pass event bus to risk manager
    )
    
    # Wrap risk manager with debug wrapper
    debug_risk_manager = DebugRiskManager(risk_manager)
    
    # Create execution components
    broker = SimulatedBroker()
    execution_engine = ExecutionEngine(broker_interface=broker)
    execution_engine.set_event_bus(traced_bus)  # Set the event bus for the execution engine
    
    # Wrap execution engine with debug wrapper
    debug_execution_engine = DebugExecutionEngine(execution_engine)
    
    # Create strategy
    strategy = MovingAverageCrossoverStrategy(
        name="ma_crossover",
        symbols=symbols,
        fast_window=fast_window,
        slow_window=slow_window
    )
    strategy.set_event_bus(traced_bus)  # Set the event bus for the strategy
    
    # Register event handlers specifically to debug the flow
    traced_bus.register(EventType.SIGNAL, debug_risk_manager.on_signal)
    traced_bus.register(EventType.ORDER, debug_execution_engine.on_order)
    
    # Create event trackers
    signal_events = []
    order_events = []
    fill_events = []
    
    # Register tracking handlers
    def track_signal(event):
        if event.get_type() == EventType.SIGNAL:
            signal_events.append(event)
            logger.debug(f"TRACKER: Signal {len(signal_events)}: {event.get_symbol()} {event.get_signal_value()}")
    
    def track_order(event):
        if event.get_type() == EventType.ORDER:
            order_events.append(event)
            logger.debug(f"TRACKER: Order {len(order_events)}: {event.get_symbol()} {event.get_direction()} {event.get_quantity()}")
    
    def track_fill(event):
        if event.get_type() == EventType.FILL:
            fill_events.append(event)
            logger.debug(f"TRACKER: Fill {len(fill_events)}: {event.get_symbol()} {event.get_direction()} {event.get_quantity()}")
    
    traced_bus.register(EventType.SIGNAL, track_signal)
    traced_bus.register(EventType.ORDER, track_order)
    traced_bus.register(EventType.FILL, track_fill)
    
    # Create backtest runner
    backtest = BacktestRunner(
        data_handler=data_handler,
        strategy=strategy,
        risk_manager=debug_risk_manager,
        execution_engine=debug_execution_engine,
        portfolio=portfolio,
        event_bus=traced_bus,
        event_manager=event_manager
    )
    
    # Run backtest
    results = backtest.run(symbols, start_dt, end_dt, timeframe)
    
    # Display debug information
    print("\n=== Debug Summary ===")
    print(f"Signal events: {len(signal_events)}")
    print(f"Order events: {len(order_events)}")
    print(f"Fill events: {len(fill_events)}")
    
    # Analyze the disconnect
    if len(signal_events) > 0 and len(order_events) == 0:
        print("\nIssue identified: Signals are not being converted to orders.")
        print("Possible causes:")
        print("1. Risk manager rejecting all signals")
        print("2. Event routing issue between signals and orders")
        
        # Debug first few signals in detail
        print("\nDetailed analysis of the first 5 signals:")
        for i, signal in enumerate(signal_events[:5]):
            symbol = signal.get_symbol()
            signal_value = signal.get_signal_value()
            price = signal.get_price()
            direction = 'BUY' if signal_value == 1 else 'SELL' if signal_value == -1 else 'NEUTRAL'
            
            print(f"\nSignal {i+1}:")
            print(f"  Symbol: {symbol}")
            print(f"  Direction: {direction}")
            print(f"  Price: {price}")
            print(f"  Timestamp: {signal.get_timestamp()}")
            
            # Calculate position size manually
            try:
                quantity = position_sizer.calculate_position_size(
                    symbol, direction, price, portfolio, signal
                )
                print(f"  Position sizer would calculate quantity: {quantity}")
                
                # Apply risk limits manually
                adjusted_quantity = risk_manager._apply_risk_limits(symbol, direction, quantity, price)
                print(f"  After risk limits, adjusted quantity would be: {adjusted_quantity}")
                
                # Check portfolio state
                position = portfolio.get_position(symbol)
                if position:
                    print(f"  Current position: {position.quantity} shares")
                else:
                    print(f"  No current position")
                    
                # Calculate key risk metrics
                equity = portfolio.get_equity()
                print(f"  Portfolio equity: ${equity:,.2f}")
                trade_value = price * quantity
                print(f"  Trade value: ${trade_value:,.2f} ({trade_value/equity:.2%} of equity)")
                
            except Exception as e:
                print(f"  Error analyzing signal: {e}")
    
    # Display results
    if results:
        print("\nBacktest Results:")
        print(f"Initial Capital: ${results['initial_equity']:,.2f}")
        print(f"Final Equity: ${results['final_equity']:,.2f}")
        
    return results, backtest, {
        'signals': signal_events,
        'orders': order_events,
        'fills': fill_events,
        'event_bus': traced_bus
    }

if __name__ == "__main__":
    results, backtest, debug_data = run_debug_backtest(
        data_dir='~/adf/data',
        symbols=['SPY'],
        start_date='2024-03-25',
        end_date='2024-04-05',
        timeframe='1m',
        fast_window=5,
        slow_window=15
    )
