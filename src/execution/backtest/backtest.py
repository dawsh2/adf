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
    risk_manager = SimpleRiskManager(portfolio, event_bus, fixed_size=position_size)
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
            if debug:
                logger.debug(f"Signal generated: {event.get_symbol()} {event.get_signal_value()}")
        elif event_type == EventType.ORDER:
            event_counts['orders'] += 1
            # Debug log order
            if debug:
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
                if debug:
                    logger.debug(f"Fill executed: {event.get_symbol()} {event.get_direction()} {event.get_quantity()} @ {event.get_price()}")
    
    # Direct registration of event handlers to ensure they're properly connected
    event_bus.register(EventType.SIGNAL, risk_manager.on_signal)
    event_bus.register(EventType.ORDER, broker.place_order)
    event_bus.register(EventType.FILL, portfolio.on_fill)
    
    # Register component with event manager
    event_manager.register_component('strategy', component, [EventType.BAR])
    
    # Register tracking for all event types
    for event_type in [EventType.BAR, EventType.SIGNAL, EventType.ORDER, EventType.FILL]:
        event_bus.register(event_type, track_event)
    
    # Reset components
    component.reset()
    risk_manager.reset()
    portfolio.reset()
    data_handler.reset()
    
    # Get symbols from component
    symbols = component.symbols if hasattr(component, 'symbols') else []
    
    # Dictionary to track market prices for all symbols
    market_prices = {}
    
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
            
            # Update market price data (used for position valuation)
            current_price = bar.get_close()
            market_prices[symbol] = current_price
            
            # Update market data in broker to ensure correct execution prices
            if hasattr(broker, 'update_market_data'):
                market_data = {
                    'price': current_price,
                    'timestamp': bar_timestamp
                }
                broker.update_market_data(symbol, market_data)
            
            # Emit bar event
            event_bus.emit(bar)
    
    # Liquidate all positions at the end of the backtest - fix issue with position liquidation
    position_details = portfolio.get_position_details(market_prices)
    if position_details:
        logger.info("Liquidating all positions at the end of the backtest")
        
        for position in position_details:
            symbol = position['symbol']
            quantity = position['quantity']
            
            # Skip if no position
            if quantity == 0:
                continue
                
            # Get last price for the symbol
            last_price = market_prices.get(symbol, position['cost_basis'])
            
            # Determine direction for liquidation (opposite of position direction)
            direction = "SELL" if quantity > 0 else "BUY"
            abs_quantity = abs(quantity)
            
            # Create liquidation order
            liquidation_order = create_order_event(
                symbol=symbol,
                order_type="MARKET",
                direction=direction,
                quantity=abs_quantity,
                price=last_price,
                timestamp=last_bar_timestamp
            )
            
            # Process liquidation order
            logger.info(f"Liquidating position: {symbol} {direction} {abs_quantity} @ {last_price}")
            broker.place_order(liquidation_order)
            
            # Add to order count
            event_counts['orders'] += 1
    
    # Clean up position dictionary by removing positions with zero quantity
    def cleanup_zero_positions(portfolio):
        """Remove positions with zero quantity."""
        zero_positions = []
        for symbol, position in portfolio.positions.items():
            if position.quantity == 0:
                zero_positions.append(symbol)
        
        # Remove the zero positions
        for symbol in zero_positions:
            del portfolio.positions[symbol]
        
        if zero_positions:
            logger.info(f"Cleaned up {len(zero_positions)} zero positions from portfolio")

    # After all positions are liquidated, clean up the portfolio
    cleanup_zero_positions(portfolio)
    
    # Create equity curve DataFrame - fix equity calculation
    # First check if portfolio has equity history
    if hasattr(portfolio, 'equity_history') and portfolio.equity_history:
        # Use portfolio's detailed equity history
        equity_curve = pd.DataFrame(portfolio.equity_history)
    else:
        # Fall back to basic calculation
        equity_curve = pd.DataFrame({
            'timestamp': [portfolio.last_update_time],
            'equity': [portfolio.get_equity(market_prices)]
        })
    
    # Calculate PnL for each trade using trade pairs
    pnl_transactions = []
    
    # Group trades by symbol
    trades_by_symbol = {}
    for trade in trades:
        symbol = trade['symbol']
        if symbol not in trades_by_symbol:
            trades_by_symbol[symbol] = []
        trades_by_symbol[symbol].append(trade)
    
    # Process each symbol's trades
    for symbol, symbol_trades in trades_by_symbol.items():
        # Sort by timestamp
        symbol_trades.sort(key=lambda x: x['timestamp'])
        
        # Track position and cost basis for PnL calculation
        position = 0
        cost_basis = 0
        total_cost = 0
        
        for trade in symbol_trades:
            direction = trade['direction']
            quantity = trade['quantity']
            price = trade['price']
            
            # Calculate trade PnL
            if direction == 'BUY':
                # Buying increases position
                if position == 0:
                    # New position
                    position = quantity
                    cost_basis = price
                    total_cost = quantity * price
                else:
                    # Adding to position
                    old_cost = position * cost_basis
                    new_cost = quantity * price
                    position += quantity
                    total_cost = old_cost + new_cost
                    cost_basis = total_cost / position if position > 0 else 0
                
                trade['pnl'] = 0  # No P&L on buys
                
            elif direction == 'SELL':
                if position > 0:
                    # Selling from a long position
                    sell_quantity = min(position, quantity)
                    sell_cost = sell_quantity * cost_basis
                    sell_proceeds = sell_quantity * price
                    trade_pnl = sell_proceeds - sell_cost
                    
                    # Update position
                    position -= sell_quantity
                    if position > 0:
                        # Position reduced but still open
                        total_cost = position * cost_basis
                    else:
                        # Position closed
                        position = 0
                        cost_basis = 0
                        total_cost = 0
                    
                    trade['pnl'] = trade_pnl
                else:
                    # Short selling
                    trade['pnl'] = 0  # P&L will be calculated on covering
            
            # Add to PnL transactions for detailed analysis
            pnl_transactions.append({
                'timestamp': trade['timestamp'],
                'symbol': symbol,
                'direction': direction,
                'quantity': quantity,
                'price': price,
                'position_after': position,
                'cost_basis_after': cost_basis,
                'pnl': trade['pnl']
            })
    
    # Get final equity and position details
    final_equity = portfolio.get_equity(market_prices)
    final_positions = portfolio.get_position_details(market_prices)
    
    # Log statistics
    logger.info(f"Backtest complete - Events: {event_counts}")
    logger.info(f"Final portfolio: Cash={portfolio.cash}, Positions={len(final_positions)}")
    logger.info(f"Final equity: {final_equity}")
    
    # Log detailed position information
    if final_positions:
        logger.info("Final positions:")
        for position in final_positions:
            logger.info(f"  {position['symbol']}: {position['quantity']} shares @ {position['cost_basis']:.2f}, " +
                       f"Market value: {position['market_value']:.2f}, P&L: {position['total_pnl']:.2f}")
    
    return equity_curve, trades
