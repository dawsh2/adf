"""
Simplified backtest implementation for validation purposes.
"""
import logging
import pandas as pd
import numpy as np
import datetime

from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType, BarEvent, SignalEvent, OrderEvent, FillEvent
from src.core.events.event_utils import create_order_event, create_fill_event
from src.execution.portfolio import PortfolioManager
from src.execution.brokers.simulated import SimulatedBroker

# Import our simplified risk manager
# from path.to.simplified_risk_manager import SimplePassthroughRiskManager
# Use the line below if the simplified_risk_manager is in the same directory
# from simplified_risk_manager import SimplePassthroughRiskManager

logger = logging.getLogger(__name__)

class SimpleEventTracker:
    """Simple event counter that tracks events by type."""
    
    def __init__(self):
        self.event_counts = {
            EventType.BAR: 0,
            EventType.SIGNAL: 0,
            EventType.ORDER: 0,
            EventType.FILL: 0
        }
        self.events = {
            EventType.BAR: [],
            EventType.SIGNAL: [],
            EventType.ORDER: [],
            EventType.FILL: []
        }
    
    def track_event(self, event):
        """Track an event occurrence."""
        event_type = event.get_type()
        if event_type in self.event_counts:
            self.event_counts[event_type] += 1
            self.events[event_type].append(event)
    
    def get_event_count(self, event_type):
        """Get count of events by type."""
        return self.event_counts.get(event_type, 0)
    
    def get_events(self, event_type):
        """Get list of events by type."""
        return self.events.get(event_type, [])

def run_simplified_backtest(component, data_handler, risk_manager_class=None, start_date=None, end_date=None, **kwargs):
    """
    Simplified backtest function that focuses on direct mapping between signals and trades.
    
    Args:
        component: Strategy component to test
        data_handler: Data handler with market data
        risk_manager_class: Risk manager class (defaults to SimplePassthroughRiskManager if available)
        start_date: Start date for backtest
        end_date: End date for backtest
        **kwargs: Additional parameters
            
    Returns:
        tuple: (equity_curve, trades, event_tracker)
    """
    logger.info("Starting simplified backtest...")
    
    # Extract parameters
    initial_cash = kwargs.get('initial_cash', 10000.0)
    debug = kwargs.get('debug', False)
    
    # Create fresh event system 
    event_bus = EventBus()
    event_manager = EventManager(event_bus)
    
    # Create event tracker
    event_tracker = SimpleEventTracker()
    
    # Create portfolio
    portfolio = PortfolioManager(initial_cash=initial_cash)
    portfolio.set_event_bus(event_bus)
    
    # Create broker
    broker = SimulatedBroker(fill_emitter=event_bus)
    broker.set_event_bus(event_bus)
    
    # Create risk manager (use specified class if provided)
    if risk_manager_class:
        risk_manager = risk_manager_class(portfolio, event_bus)
    else:
        # Try to import and use SimplePassthroughRiskManager
        try:
            from src.strategies.risk.risk_manager import SimplePassthroughRiskManager
            risk_manager = SimplePassthroughRiskManager(portfolio, event_bus)
        except ImportError:
            # Fall back to regular SimpleRiskManager with fixed position size
            from src.strategy.risk.risk_manager import SimpleRiskManager
            risk_manager = SimpleRiskManager(portfolio, event_bus)
            # Override position sizing method
            risk_manager.calculate_position_size = lambda symbol, price, signal_strength=1.0: 1
    
    # Set debug mode if requested
    if hasattr(risk_manager, 'set_debug') and debug:
        risk_manager.set_debug(debug)
    
    # Set broker for direct access
    if hasattr(risk_manager, 'broker'):
        risk_manager.broker = broker
    
    # Set event bus for component
    if hasattr(component, 'set_event_bus'):
        component.set_event_bus(event_bus)
    
    # Track trades
    trades = []
    
    def track_event(event):
        """Track events for analysis."""
        # Use the event tracker
        event_tracker.track_event(event)
        
        # Log specific event types
        event_type = event.get_type()
        
        if event_type == EventType.SIGNAL:
            if debug:
                symbol = event.get_symbol()
                signal_value = event.get_signal_value()
                price = event.get_price()
                logger.debug(f"Signal: {symbol} {signal_value} @ {price:.2f}")
        
        elif event_type == EventType.ORDER:
            if debug:
                symbol = event.get_symbol()
                direction = event.get_direction()
                quantity = event.get_quantity() 
                price = event.get_price()
                logger.debug(f"Order: {symbol} {direction} {quantity} @ {price:.2f}")
        
        elif event_type == EventType.FILL:
            # Track fills as trades
            symbol = event.get_symbol()
            direction = event.get_direction()
            quantity = event.get_quantity()
            price = event.get_price()
            
            if debug:
                logger.debug(f"Fill: {symbol} {direction} {quantity} @ {price:.2f}")
            
            # Add to trades list
            trades.append({
                'timestamp': event.get_timestamp(),
                'symbol': symbol,
                'direction': direction,
                'quantity': quantity,
                'price': price,
                'commission': event.get_commission() if hasattr(event, 'get_commission') else 0
            })
    
    # Register event handlers
    event_bus.register(EventType.SIGNAL, risk_manager.on_signal)
    event_bus.register(EventType.ORDER, broker.place_order)
    event_bus.register(EventType.FILL, portfolio.on_fill)
    
    # Register event tracking
    for event_type in [EventType.BAR, EventType.SIGNAL, EventType.ORDER, EventType.FILL]:
        event_bus.register(event_type, track_event)
    
    # Register component with event manager
    event_manager.register_component('strategy', component, [EventType.BAR])
    
    # Reset components
    component.reset()
    portfolio.reset()
    data_handler.reset()
    if hasattr(risk_manager, 'reset'):
        risk_manager.reset()
    
    # Get symbols
    symbols = component.symbols if hasattr(component, 'symbols') else []
    if not symbols:
        logger.error("No symbols found in component")
        return None, [], event_tracker
    
    # Track market prices for valuation
    market_prices = {}
    
    # Process data
    logger.info(f"Processing data for {len(symbols)} symbols...")
    
    for symbol in symbols:
        data_handler.reset()
        
        # Process each bar
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
            
            # Extract bar details
            timestamp = bar.get_timestamp()
            current_price = bar.get_close()
            
            # Filter by date if needed
            if start_date and timestamp < start_date:
                continue
                
            if end_date and timestamp > end_date:
                break
            
            # Update market data
            market_prices[symbol] = current_price
            broker.update_market_data(symbol, {'price': current_price, 'timestamp': timestamp})
            
            # Emit bar event
            event_bus.emit(bar)
    
    # Liquidate positions at end of test
    logger.info("Backtest processing complete, liquidating positions...")
    
    position_details = portfolio.get_position_details(market_prices)
    for position in position_details:
        symbol = position['symbol']
        quantity = position['quantity']
        
        if quantity == 0:
            continue
            
        # Determine liquidation direction
        direction = "SELL" if quantity > 0 else "BUY"
        abs_quantity = abs(quantity)
        price = market_prices.get(symbol, position['cost_basis'])
        
        # Create and process liquidation order
        liquidation_order = create_order_event(
            symbol=symbol,
            order_type="MARKET",
            direction=direction,
            quantity=abs_quantity,
            price=price
        )
        
        logger.info(f"Liquidating: {symbol} {direction} {abs_quantity} @ {price:.2f}")
        broker.place_order(liquidation_order)
    
    # Create equity curve
    equity_curve = pd.DataFrame(portfolio.equity_history) if hasattr(portfolio, 'equity_history') else None
    
    # If equity curve is empty or None, create a simple one
    if equity_curve is None or len(equity_curve) == 0:
        equity_curve = pd.DataFrame({
            'timestamp': [datetime.datetime.now()],
            'equity': [portfolio.get_equity(market_prices)]
        })
    
    # Calculate simple PnL for trades (no need for complex logic)
    # Just calculate the trade P&L directly as price change * quantity
    for trade in trades:
        if trade['direction'] == 'BUY':
            trade['pnl'] = 0  # Don't calculate P&L for buys
        else:  # For SELL trades
            # Find the most recent BUY trade for this symbol
            buy_price = None
            for prev_trade in reversed(trades):
                if prev_trade['symbol'] == trade['symbol'] and prev_trade['direction'] == 'BUY':
                    buy_price = prev_trade['price']
                    break
            
            if buy_price:
                # P&L is simply (sell_price - buy_price) * quantity
                trade['pnl'] = (trade['price'] - buy_price) * trade['quantity']
            else:
                trade['pnl'] = 0
    
    # Log summary
    logger.info(f"Backtest complete - {len(trades)} trades executed")
    logger.info(f"Events: Bars={event_tracker.get_event_count(EventType.BAR)}, " +
               f"Signals={event_tracker.get_event_count(EventType.SIGNAL)}, " +
               f"Orders={event_tracker.get_event_count(EventType.ORDER)}, " +
               f"Fills={event_tracker.get_event_count(EventType.FILL)}")
    
    initial_equity = equity_curve['equity'].iloc[0] if len(equity_curve) > 0 else initial_cash
    final_equity = portfolio.get_equity(market_prices)
    
    logger.info(f"Initial equity: ${initial_equity:.2f}")
    logger.info(f"Final equity: ${final_equity:.2f}")
    logger.info(f"Return: {((final_equity/initial_equity - 1) * 100):.2f}%")
    
    # Return results and event tracker
    return equity_curve, trades, event_tracker
