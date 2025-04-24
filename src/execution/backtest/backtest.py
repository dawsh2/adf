"""
Complete fixed backtest implementation with proper position tracking, liquidation and equity calculation.
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
from src.strategy.risk.risk_manager import SimplePassthroughRiskManager

logger = logging.getLogger(__name__)

def run_backtest(component, data_handler, start_date=None, end_date=None, timestamp_translator=None, **kwargs):
    """
    Enhanced version of run_backtest that properly handles trades and P&L calculation.
    
    Args:
        component: Strategy component to test
        data_handler: Data handler with market data
        start_date: Start date for backtest (string or datetime)
        end_date: End date for backtest (string or datetime)
        timestamp_translator: Optional function to translate timestamps
        **kwargs: Additional parameters including:
            initial_cash: Starting cash amount (default: 10000.0)
            position_size: Default position size (default: 100)
            slippage_model: Model for price slippage
            debug: Enable verbose debug logging
            
    Returns:
        tuple: (equity_curve, trades)
    """
    import logging
    import pandas as pd
    from src.execution.trade_tracker import TradeTracker  # Import the new TradeTracker
    
    logger = logging.getLogger(__name__)
    
    # Extract additional parameters
    initial_cash = kwargs.get('initial_cash', 10000.0)
    position_size = kwargs.get('position_size', 100)
    slippage_model = kwargs.get('slippage_model', None)
    debug = kwargs.get('debug', False)
    
    # Set debug logging if requested
    if debug:
        logger.setLevel(logging.DEBUG)
    
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
    portfolio = PortfolioManager(initial_cash=initial_cash)
    portfolio.set_event_bus(event_bus)
    
    # Create broker with explicit fill emitter
    broker = SimulatedBroker(fill_emitter=event_bus, slippage_model=slippage_model)
    broker.set_event_bus(event_bus)
    
    # Create risk manager with position sizing
    risk_manager = SimplePassthroughRiskManager(portfolio=portfolio, event_bus=event_bus)
    risk_manager.broker = broker  # Direct connection to broker
    
    # Set event bus for component
    if hasattr(component, 'set_event_bus'):
        component.set_event_bus(event_bus)
    
    # Create trade tracker
    trade_tracker = TradeTracker(initial_cash=initial_cash)
    
    # Track events for analysis
    event_counts = {
        'bars': 0,
        'signals': 0,
        'orders': 0,
        'fills': 0
    }
    
    # Track market prices for position valuation
    market_prices = {}
    
    def track_event(event):
        """Track events for analysis and updating trade tracker."""
        event_type = event.get_type()
        
        if event_type == EventType.BAR:
            event_counts['bars'] += 1
            
            # Update market prices
            symbol = event.get_symbol()
            price = event.get_close()
            timestamp = event.get_timestamp()
            market_prices[symbol] = price
            
            # Update trade tracker equity
            trade_tracker.update_equity(timestamp, market_prices)
            
            # Debug log
            if debug and event_counts['bars'] % 100 == 0:
                logger.debug(f"Bar {event_counts['bars']}: {symbol} @ {price:.2f}")
                
        elif event_type == EventType.SIGNAL:
            event_counts['signals'] += 1
            # Debug log signal
            if debug:
                symbol = event.get_symbol()
                signal_value = event.get_signal_value()
                logger.debug(f"Signal: {symbol} {signal_value}")
                
        elif event_type == EventType.ORDER:
            event_counts['orders'] += 1
            # Debug log order
            if debug:
                symbol = event.get_symbol()
                direction = event.get_direction()
                quantity = event.get_quantity()
                logger.debug(f"Order: {symbol} {direction} {quantity}")
                
        elif event_type == EventType.FILL:
            event_counts['fills'] += 1
            
            # Process fill with trade tracker
            fill_data = {
                'timestamp': event.get_timestamp(),
                'symbol': event.get_symbol(),
                'direction': event.get_direction(),
                'quantity': event.get_quantity(),
                'price': event.get_price(),
                'commission': event.get_commission() if hasattr(event, 'get_commission') else 0.0
            }
            
            pnl = trade_tracker.process_fill(fill_data)
            
            # Debug log fill
            if debug:
                logger.debug(f"Fill: {fill_data['symbol']} {fill_data['direction']} " +
                           f"{fill_data['quantity']} @ {fill_data['price']:.2f}, PnL: {pnl:.2f}")
    
    # Register event tracking for all event types
    for event_type in [EventType.BAR, EventType.SIGNAL, EventType.ORDER, EventType.FILL]:
        event_bus.register(event_type, track_event)
    
    # Direct registration of event handlers
    event_bus.register(EventType.SIGNAL, risk_manager.on_signal)
    event_bus.register(EventType.ORDER, broker.place_order)
    event_bus.register(EventType.FILL, portfolio.on_fill)
    
    # Register component with event manager
    event_manager.register_component('strategy', component, [EventType.BAR])
    
    # Reset components
    component.reset()
    risk_manager.reset()
    portfolio.reset()
    data_handler.reset()
    
    # Get symbols from component
    symbols = component.symbols if hasattr(component, 'symbols') else []
    
    # Run backtest
    logger.info(f"Starting backtest with {len(symbols)} symbols")
    last_bar_timestamp = None
    
    # Process each symbol
    for symbol in symbols:
        data_handler.reset()
        
        # Process each bar
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
            
            # Get and possibly translate timestamp
            bar_timestamp = bar.get_timestamp()
            last_bar_timestamp = bar_timestamp
            
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
                    if (hasattr(bar_timestamp, 'tzinfo') and bar_timestamp.tzinfo is not None and
                        hasattr(start_date, 'tzinfo') and start_date.tzinfo is None):
                        start_date = start_date.replace(tzinfo=bar_timestamp.tzinfo)
                    elif (hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None and
                         hasattr(bar_timestamp, 'tzinfo') and bar_timestamp.tzinfo is None):
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
                    if (hasattr(bar_timestamp, 'tzinfo') and bar_timestamp.tzinfo is not None and
                        hasattr(end_date, 'tzinfo') and end_date.tzinfo is None):
                        end_date = end_date.replace(tzinfo=bar_timestamp.tzinfo)
                    elif (hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None and
                         hasattr(bar_timestamp, 'tzinfo') and bar_timestamp.tzinfo is None):
                        bar_timestamp = bar_timestamp.replace(tzinfo=end_date.tzinfo)
                    
                    if bar_timestamp > end_date:
                        break
                except TypeError as e:
                    logger.error(f"Error comparing timestamps: {e}")
                    # Try naive comparison
                    if bar_timestamp.replace(tzinfo=None) > end_date.replace(tzinfo=None):
                        break
            
            # Update broker's market data
            price = bar.get_close()
            broker.update_market_data(symbol, {
                'price': price,
                'timestamp': bar_timestamp
            })
            
            # Emit bar event
            event_bus.emit(bar)
    
    # Liquidate all positions at the end of the backtest
    if last_bar_timestamp:
        logger.info("Liquidating all positions at the end of the backtest")
        trade_tracker.liquidate_positions(last_bar_timestamp, market_prices)
    
    # Get equity curve and trades
    equity_curve = trade_tracker.get_equity_curve()
    trades = trade_tracker.get_closed_trades()
    
    # Log statistics
    logger.info(f"=== BACKTEST COMPLETE ===")
    logger.info(f"Events processed: {event_counts}")
    logger.info(f"Trades executed: {len(trades)}")
    
    # Get trade statistics
    trade_stats = trade_tracker.get_trade_statistics()
    logger.info(f"Trade statistics: {trade_stats}")
    
    return equity_curve, trades
