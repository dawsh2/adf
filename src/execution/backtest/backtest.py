# src/execution/backtest/backtest.py
import pandas as pd
import numpy as np
import datetime

# Add these imports
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType, BarEvent, SignalEvent, OrderEvent, FillEvent
from src.execution.portfolio import PortfolioManager
from src.execution.brokers.simulated import SimulatedBroker
from src.strategy.risk.risk_manager import SimpleRiskManager

import logging
logger = logging.getLogger(__name__)

# src/execution/backtest/backtest.py

# src/execution/backtest/backtest.py - replace run_backtest function
def run_backtest(component, data_handler, start_date=None, end_date=None, timestamp_translator=None, **kwargs):
    """
    Enhanced version of run_backtest that properly handles events and timestamps.
    
    Args:
        component: Strategy component to test
        data_handler: Data handler with market data
        start_date: Start date for backtest (string or datetime)
        end_date: End date for backtest (string or datetime)
        timestamp_translator: Optional function to translate timestamps
        **kwargs: Additional parameters
        
    Returns:
        tuple: (equity_curve, trades)
    """
    import logging
    import pandas as pd
    logger = logging.getLogger(__name__)
    
    from src.core.events.event_bus import EventBus
    from src.core.events.event_manager import EventManager
    from src.core.events.event_types import EventType
    from src.execution.portfolio import PortfolioManager
    from src.execution.brokers.simulated import SimulatedBroker
    from src.strategy.risk.risk_manager import SimpleRiskManager
    
    # Import datetime utilities if available
    try:
        from src.data.datetime_utils import parse_timestamp
        
        # Convert date inputs to datetime if they're strings
        if isinstance(start_date, str):
            start_date = parse_timestamp(start_date)
        if isinstance(end_date, str):
            end_date = parse_timestamp(end_date)
    except ImportError:
        # Fall back to pandas if datetime_utils not available
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
    
    # Create fresh event system for each run
    event_bus = EventBus()
    event_manager = EventManager(event_bus)
    
    # Set event bus for component
    if hasattr(component, 'set_event_bus'):
        component.set_event_bus(event_bus)
    
    # Create portfolio
    portfolio = PortfolioManager(initial_cash=10000.0)
    portfolio.set_event_bus(event_bus)
    
    # Create broker - make sure it emits fills directly
    broker = SimulatedBroker(fill_emitter=event_bus)
    broker.set_event_bus(event_bus)
    
    # Create risk manager
    risk_manager = SimpleRiskManager(portfolio)
    risk_manager.set_event_bus(event_bus)
    
    # Register components with event manager
    event_manager.register_component('strategy', component, [EventType.BAR])
    event_manager.register_component('risk_manager', risk_manager, [EventType.SIGNAL])
    event_manager.register_component('broker', broker, [EventType.ORDER])
    event_manager.register_component('portfolio', portfolio, [EventType.FILL])
    
    # Reset components
    component.reset()
    risk_manager.reset()
    portfolio.reset()
    data_handler.reset()
    
    # Add logging to track event flow
    event_counts = {
        'bars': 0,
        'signals': 0,
        'orders': 0,
        'fills': 0
    }
    trades = []  # Track trades for analysis
    
    def track_event(event):
        """Track events for analysis and debugging."""
        event_type = event.get_type()
        if event_type == EventType.BAR:
            event_counts['bars'] += 1
        elif event_type == EventType.SIGNAL:
            event_counts['signals'] += 1
            logger.debug(f"Signal generated: {event.get_symbol()} {event.get_signal_value()}")
        elif event_type == EventType.ORDER:
            event_counts['orders'] += 1
            logger.debug(f"Order generated: {event.get_symbol()} {event.get_direction()} {event.get_quantity()} @ {event.get_price()}")
        elif event_type == EventType.FILL:
            event_counts['fills'] += 1
            # Track fill details for analysis
            if hasattr(event, 'get_symbol') and hasattr(event, 'get_direction'):
                trades.append({
                    'timestamp': event.get_timestamp(),
                    'symbol': event.get_symbol(),
                    'direction': event.get_direction(),
                    'quantity': event.get_quantity(),
                    'price': event.get_price(),
                    'commission': event.get_commission() if hasattr(event, 'get_commission') else 0,
                    'pnl': 0  # Will be calculated later
                })
                logger.debug(f"Fill executed: {event.get_symbol()} {event.get_direction()} {event.get_quantity()} @ {event.get_price()}")
    
    # Register tracking for all event types
    for event_type in [EventType.BAR, EventType.SIGNAL, EventType.ORDER, EventType.FILL]:
        event_bus.register(event_type, track_event)
    
    # Track equity and timestamps
    equity_values = [portfolio.cash]
    timestamps = [pd.Timestamp.now()]  # Use pd.Timestamp for consistency
    
    # Get symbols from component
    symbols = component.symbols if hasattr(component, 'symbols') else []
    
    # Run backtest
    logger.info(f"Starting fixed backtest with {len(symbols)} symbols")
    bars_processed = 0
    
    # Process each symbol
    for symbol in symbols:
        data_handler.reset()
        
        # Process each bar
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
            
            bars_processed += 1
            
            # Get and possibly translate timestamp
            bar_timestamp = bar.get_timestamp()
            original_timestamp = bar_timestamp
            
            # Apply timestamp translation if provided
            if timestamp_translator and callable(timestamp_translator):
                try:
                    bar_timestamp = timestamp_translator(bar_timestamp)
                    # Log some translations for debugging
                    if bars_processed <= 3 or bars_processed % 100 == 0:
                        logger.debug(f"Translated timestamp: {original_timestamp} -> {bar_timestamp}")
                except Exception as e:
                    logger.error(f"Error translating timestamp: {e}")
            
            # Make timestamps comparable by removing timezone info if necessary
            if start_date is not None:
                # Make date comparison more robust
                try:
                    # Try the direct comparison first
                    skip_bar = bar_timestamp < start_date
                except TypeError:
                    # Handle timezone differences by converting to naive datetimes
                    bar_ts_naive = bar_timestamp.replace(tzinfo=None) if hasattr(bar_timestamp, 'tzinfo') else bar_timestamp
                    start_naive = start_date.replace(tzinfo=None) if hasattr(start_date, 'tzinfo') else start_date
                    skip_bar = bar_ts_naive < start_naive
                
                if skip_bar:
                    continue
            
            if end_date is not None:
                # Make date comparison more robust
                try:
                    # Try the direct comparison first
                    break_loop = bar_timestamp > end_date
                except TypeError:
                    # Handle timezone differences by converting to naive datetimes
                    bar_ts_naive = bar_timestamp.replace(tzinfo=None) if hasattr(bar_timestamp, 'tzinfo') else bar_timestamp
                    end_naive = end_date.replace(tzinfo=None) if hasattr(end_date, 'tzinfo') else end_date
                    break_loop = bar_ts_naive > end_naive
                
                if break_loop:
                    break
            
            # Track what we're processing
            if bars_processed <= 3 or bars_processed % 100 == 0:
                logger.debug(f"Processing bar {bars_processed}: {symbol} @ {bar_timestamp}")
            
            # Emit bar event directly
            event_bus.emit(bar)
            
            # Track timestamp and equity
            timestamps.append(bar_timestamp)
            equity_values.append(portfolio.get_equity())
    
    # Create DataFrame with timestamp and equity columns
    equity_curve = pd.DataFrame({
        'timestamp': timestamps[:len(equity_values)],
        'equity': equity_values
    })
    
    # Remove duplicates if any
    equity_curve = equity_curve.drop_duplicates(subset=['timestamp'])
    
    # Calculate PnL for each trade
    if len(trades) > 1:
        for i in range(1, len(trades)):
            prev_trade = trades[i-1]
            curr_trade = trades[i]
            
            if prev_trade['symbol'] == curr_trade['symbol']:
                if prev_trade['direction'] != curr_trade['direction']:
                    # Calculate PnL for this pair of trades
                    price_diff = curr_trade['price'] - prev_trade['price']
                    if prev_trade['direction'] == 'BUY':
                        pnl = price_diff * prev_trade['quantity']
                    else:
                        pnl = -price_diff * prev_trade['quantity']
                    
                    curr_trade['pnl'] = pnl
    
    # Log statistics
    logger.info(f"Backtest complete - Events: {event_counts}")
    logger.info(f"Final portfolio: Cash={portfolio.cash}, Positions={len(portfolio.positions)}")
    logger.info(f"Final equity: {equity_values[-1]}")
    
    # Log position details
    if portfolio.positions:
        logger.info("Final positions:")
        for symbol, position in portfolio.positions.items():
            logger.info(f"  {symbol}: {position.quantity} shares @ {position.cost_basis:.2f}")
    
    return equity_curve, trades
