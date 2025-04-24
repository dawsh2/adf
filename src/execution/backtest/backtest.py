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

def run_backtest(component, data_handler, start_date=None, end_date=None, **kwargs):
    """
    Fixed version of run_backtest that properly handles events and creates fills.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    from src.core.events.event_bus import EventBus
    from src.core.events.event_manager import EventManager
    from src.core.events.event_types import EventType
    from src.execution.portfolio import PortfolioManager
    from src.execution.brokers.simulated import SimulatedBroker
    from src.strategy.risk.risk_manager import SimpleRiskManager
    
    # Convert date inputs to datetime if they're strings
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
        event_type = event.get_type()
        if event_type == EventType.BAR:
            event_counts['bars'] += 1
        elif event_type == EventType.SIGNAL:
            event_counts['signals'] += 1
        elif event_type == EventType.ORDER:
            event_counts['orders'] += 1
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
                    'pnl': 0  # Will be calculated later
                })
    
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
    while True:
        # Get next bar for each symbol
        any_bars = False
        for symbol in symbols:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                continue
                
            # Check date range if specified
            bar_timestamp = bar.get_timestamp()
            if start_date and bar_timestamp < start_date:
                continue
            if end_date and bar_timestamp > end_date:
                continue
                
            any_bars = True
            
            # Emit bar event
            event_bus.emit(bar)
            
            # Track timestamp and equity
            timestamps.append(bar_timestamp)
            equity_values.append(portfolio.get_equity())
        
        # If no bars for any symbol, end backtest
        if not any_bars:
            break
    
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
    
    # Create DataFrame with timestamp and equity columns
    equity_curve = pd.DataFrame({
        'timestamp': timestamps[:len(equity_values)],
        'equity': equity_values
    })
    
    # Remove duplicates if any
    equity_curve = equity_curve.drop_duplicates(subset=['timestamp'])
    
    # Log statistics
    logger.info(f"Backtest complete - Events: {event_counts}")
    logger.info(f"Final portfolio: Cash={portfolio.cash}, Positions={len(portfolio.positions)}")
    logger.info(f"Final equity: {equity_values[-1]}")
    
    # Return tuple as expected by evaluate_backtest
    return equity_curve, trades
