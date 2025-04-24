"""
Complete fixed backtest implementation with proper order tracking and position liquidation.
"""
import pandas as pd
import numpy as np
import datetime
import logging

from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType, BarEvent, SignalEvent, OrderEvent, FillEvent
from src.core.events.event_utils import create_order_event, create_fill_event
from src.execution.portfolio import PortfolioManager
from src.execution.brokers.simulated import SimulatedBroker
from src.strategy.risk.risk_manager import SimpleRiskManager

logger = logging.getLogger(__name__)

def run_backtest(component, data_handler, start_date=None, end_date=None, timestamp_translator=None, **kwargs):
    """
    Enhanced version of run_backtest that properly handles events, timestamps, and position liquidation.
    
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
    
    # Create portfolio
    portfolio = PortfolioManager(initial_cash=10000.0)
    portfolio.set_event_bus(event_bus)
    
    # Create broker with explicit fill emitter
    broker = SimulatedBroker(fill_emitter=event_bus)
    
    # Create risk manager - explicitly set broker
    risk_manager = SimpleRiskManager(portfolio, event_bus)
    risk_manager.broker = broker  # Direct connection to broker
    
    # Set event bus for component
    if hasattr(component, 'set_event_bus'):
        component.set_event_bus(event_bus)
    
    # Track events for analysis
    event_counts = {
        'bars': 0,
        'signals': 0,
        'orders': 0,
        'fills': 0
    }
    trades = []
    
    def track_event(event):
        """Track events for analysis and debugging."""
        event_type = event.get_type()
        if event_type == EventType.BAR:
            event_counts['bars'] += 1
        elif event_type == EventType.SIGNAL:
            event_counts['signals'] += 1
            # Debug log signal
            logger.debug(f"Signal generated: {event.get_symbol()} {event.get_signal_value()}")
        elif event_type == EventType.ORDER:
            event_counts['orders'] += 1
            # Debug log order
            logger.debug(f"Order generated: {event.get_symbol()} {event.get_direction()} {event.get_quantity()} @ {event.get_price()}")
        elif event_type == EventType.FILL:
            event_counts['fills'] += 1
            # Track trade for analysis
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
    
    # Register handlers for components
    event_manager.register_component('strategy', component, [EventType.BAR])
    event_manager.register_component('risk_manager', risk_manager, [EventType.SIGNAL])
    event_manager.register_component('broker', broker, [EventType.ORDER])
    event_manager.register_component('portfolio', portfolio, [EventType.FILL])
    
    # Register tracking for all event types
    for event_type in [EventType.BAR, EventType.SIGNAL, EventType.ORDER, EventType.FILL]:
        event_bus.register(event_type, track_event)
    
    # Reset components
    component.reset()
    risk_manager.reset()
    portfolio.reset()
    data_handler.reset()
    
    # Track equity and timestamps
    equity_values = [portfolio.cash]
    timestamps = [pd.Timestamp.now()]
    
    # Get symbols from component
    symbols = component.symbols if hasattr(component, 'symbols') else []
    
    # Run backtest
    logger.info(f"Starting fixed backtest with {len(symbols)} symbols")
    bars_processed = 0
    last_bar_timestamp = None  # To track last bar timestamp for position liquidation
    
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
            last_bar_timestamp = bar_timestamp  # Track for position liquidation
            
            # Apply timestamp translation if provided
            if timestamp_translator and callable(timestamp_translator):
                try:
                    bar_timestamp = timestamp_translator(bar_timestamp)
                except Exception as e:
                    logger.error(f"Error translating timestamp: {e}")
            
            # Apply date range filtering
            if start_date is not None:
                try:
                    # Handle timezone differences
                    if hasattr(bar_timestamp, 'tzinfo') and bar_timestamp.tzinfo is not None:
                        if hasattr(start_date, 'tzinfo') and start_date.tzinfo is None:
                            start_date = start_date.replace(tzinfo=bar_timestamp.tzinfo)
                    elif hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None:
                        bar_timestamp = bar_timestamp.replace(tzinfo=start_date.tzinfo)
                        
                    if bar_timestamp < start_date:
                        continue
                except TypeError as e:
                    logger.error(f"Error comparing timestamps: {e}")
                    # Try naive comparison
                    if bar_timestamp.replace(tzinfo=None) < start_date.replace(tzinfo=None):
                        continue
            
            if end_date is not None:
                try:
                    # Handle timezone differences
                    if hasattr(bar_timestamp, 'tzinfo') and bar_timestamp.tzinfo is not None:
                        if hasattr(end_date, 'tzinfo') and end_date.tzinfo is None:
                            end_date = end_date.replace(tzinfo=bar_timestamp.tzinfo)
                    elif hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None:
                        bar_timestamp = bar_timestamp.replace(tzinfo=end_date.tzinfo)
                    
                    if bar_timestamp > end_date:
                        break
                except TypeError as e:
                    logger.error(f"Error comparing timestamps: {e}")
                    # Try naive comparison
                    if bar_timestamp.replace(tzinfo=None) > end_date.replace(tzinfo=None):
                        break
            
            # Update market data in broker to ensure correct execution prices
            if hasattr(broker, 'update_market_data'):
                market_data = {
                    'price': bar.get_close(),
                    'timestamp': bar_timestamp
                }
                broker.update_market_data(symbol, market_data)
            
            # Emit bar event
            event_bus.emit(bar)
            
            # Track equity after each bar
            current_equity = portfolio.get_equity()
            equity_values.append(current_equity)
            timestamps.append(bar_timestamp)
    
    # Liquidate all positions at the end of the backtest
    if portfolio.positions:
        logger.info("Liquidating all positions at the end of the backtest")
        
        for symbol, position in list(portfolio.positions.items()):
            # Skip if no position
            if position.quantity == 0:
                continue
                
            # Get last price for the symbol
            last_price = broker.get_market_price(symbol)
            
            # Determine direction for liquidation
            direction = "SELL" if position.quantity > 0 else "BUY"
            quantity = abs(position.quantity)
            
            # Create liquidation order
            liquidation_order = create_order_event(
                symbol=symbol,
                order_type="MARKET",
                direction=direction,
                quantity=quantity,
                price=last_price,
                timestamp=last_bar_timestamp
            )
            
            # Process liquidation order
            logger.info(f"Liquidating position: {symbol} {direction} {quantity} @ {last_price}")
            broker.place_order(liquidation_order)
            
            # Add to order count
            event_counts['orders'] += 1
    
    # Calculate final equity after liquidation
    final_equity = portfolio.get_equity()
    equity_values.append(final_equity)
    timestamps.append(timestamps[-1] if timestamps else pd.Timestamp.now())
    
    # Create equity curve DataFrame
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
    logger.info(f"Final equity: {final_equity}")
    
    # Log position details
    if portfolio.positions:
        logger.info("Final positions:")
        for symbol, position in portfolio.positions.items():
            logger.info(f"  {symbol}: {position.quantity} shares @ {position.cost_basis:.2f}")
    
    return equity_curve, trades
