# src/execution/backtest/simulator.py
import pandas as pd
import numpy as np
import datetime
import logging


def run_backtest(component, data_handler, start_date=None, end_date=None, **kwargs):
    """
    Run a backtest for a component using the provided data handler.
    
    Args:
        component: Strategy component to test
        data_handler: Data handler with market data
        start_date: Start date for backtest
        end_date: End date for backtest
        **kwargs: Additional parameters
        
    Returns:
        tuple: (equity_curve DataFrame, trades)
    """
    # Set up logging
    logger = logging.getLogger("backtest")
    logger.info(f"Starting backtest with data from {start_date} to {end_date}")
    
    # Create event system
    from src.core.events.event_bus import EventBus
    from src.core.events.event_manager import EventManager
    from src.core.events.event_types import EventType
    from src.core.events.event_utils import create_fill_event, EventTracker
    
    # Create event bus and manager
    event_bus = EventBus()
    event_manager = EventManager(event_bus)
    
    # Create event tracker for debugging
    event_tracker = EventTracker(name="backtest_tracker")
    event_bus.register(EventType.BAR, event_tracker.track_event)
    event_bus.register(EventType.SIGNAL, event_tracker.track_event)
    event_bus.register(EventType.ORDER, event_tracker.track_event)
    event_bus.register(EventType.FILL, event_tracker.track_event)
    
    # Set event bus for component
    if hasattr(component, 'set_event_bus'):
        component.set_event_bus(event_bus)
        logger.debug(f"Set event bus for component: {component.name if hasattr(component, 'name') else 'unnamed'}")
    
    # Create portfolio for tracking positions and PnL
    from src.execution.portfolio import PortfolioManager
    portfolio = PortfolioManager(initial_cash=10000.0)
    portfolio.set_event_bus(event_bus)
    logger.debug(f"Created portfolio with initial cash: {portfolio.cash}")
    
    # Create simulated broker
    from src.execution.brokers.simulated import SimulatedBroker
    broker = SimulatedBroker(fill_emitter=event_bus)
    broker.set_event_bus(event_bus)
    logger.debug("Created simulated broker with event bus connection")
    
    # Create risk manager for order generation
    from src.strategy.risk.risk_manager import SimpleRiskManager
    risk_manager = SimpleRiskManager(portfolio, event_bus)
    
    # Add direct broker reference to risk manager
    risk_manager.broker = broker  # This is a key addition
    logger.debug("Added broker reference to risk manager")
    
    # Register components with event manager
    event_manager.register_component('strategy', component, [EventType.BAR])
    event_manager.register_component('risk_manager', risk_manager, [EventType.SIGNAL])
    event_manager.register_component('portfolio', portfolio, [EventType.FILL])
    logger.debug("Registered components with event manager")
    
    # Reset components
    component.reset()
    risk_manager.reset()
    portfolio.reset()
    logger.debug("Reset all components")
    
    # Reset data handler
    data_handler.reset()
    
    # Get symbols from component
    symbols = component.symbols if hasattr(component, 'symbols') else []
    logger.info(f"Running backtest for symbols: {symbols}")
    
    # Track equity and timestamps
    equity_values = [portfolio.cash]  # Start with initial cash
    timestamps = [datetime.datetime.now() if start_date is None else start_date]  # Start time
    
    # Track market prices for position valuation
    market_prices = {symbol: 0.0 for symbol in symbols}
    
    # Run backtest
    bar_count = 0
    signal_count = 0
    order_count = 0
    fill_count = 0
    
    logger.info("Starting backtest loop")
    
    while True:
        # Get next bar for each symbol
        bars = []
        for symbol in symbols:
            bar = data_handler.get_next_bar(symbol)
            if bar:
                bars.append(bar)
                # Update market data in broker for accurate fills
                market_prices[symbol] = bar.get_close()
                broker.update_market_data(symbol, {
                    'price': bar.get_close(),
                    'timestamp': bar.get_timestamp()
                })
        
        # If no bars, end backtest
        if not bars:
            logger.info("No more bars available, ending backtest")
            break
        
        # Process bars
        for bar in bars:
            # Log the bar being processed
            symbol = bar.get_symbol()
            close_price = bar.get_close()
            timestamp = bar.get_timestamp()
            logger.debug(f"Processing bar: {symbol} @ {timestamp} - Close: {close_price:.2f}")
            
            # Update market prices dictionary
            market_prices[symbol] = close_price
            
            # Emit bar event
            event_bus.emit(bar)
            bar_count += 1
            
            # Track timestamp from bar
            timestamps.append(timestamp)
            
            # Get current portfolio value using market prices
            current_equity = portfolio.get_equity(market_prices)
            
            # Log portfolio state periodically
            if bar_count % 10 == 0 or bar_count == 1:
                logger.debug(f"Portfolio state: Cash={portfolio.cash:.2f}, Equity={current_equity:.2f}")
                for sym, pos in portfolio.positions.items():
                    logger.debug(f"  Position: {sym} - {pos.quantity} shares @ {pos.cost_basis:.2f} (Market: {market_prices.get(sym, 0):.2f})")
            
            # Track equity values
            equity_values.append(current_equity)
            
            # Update event counts for logging
            signal_count = event_tracker.get_event_count(EventType.SIGNAL)
            order_count = event_tracker.get_event_count(EventType.ORDER)
            fill_count = event_tracker.get_event_count(EventType.FILL)
    
    # Log summary statistics
    logger.info(f"Backtest complete - Processed {bar_count} bars")
    logger.info(f"Generated {signal_count} signals, {order_count} orders, and {fill_count} fills")
    logger.info(f"Final portfolio: Cash={portfolio.cash:.2f}, Positions={len(portfolio.positions)}")
    logger.info(f"Final equity: {equity_values[-1]:.2f}")
    
    # Create DataFrame with timestamp and equity columns
    import pandas as pd
    equity_curve = pd.DataFrame({
        'timestamp': timestamps[:len(equity_values)],  # Ensure same length
        'equity': equity_values
    })
    
    # Get trades count
    trades = len(portfolio.fill_history) if hasattr(portfolio, 'fill_history') else 0
    logger.info(f"Total trades executed: {trades}")
    
    # Return tuple as expected by evaluate_backtest
    return equity_curve, trades
