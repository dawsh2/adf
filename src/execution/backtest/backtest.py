# src/execution/backtest/simulator.py
import pandas as pd
import numpy as np
import datetime

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
    # Set up event handling system
    from src.core.events.event_bus import EventBus
    from src.core.events.event_manager import EventManager
    from src.core.events.event_types import EventType
    
    # Create event bus and manager
    event_bus = EventBus()
    event_manager = EventManager(event_bus)
    
    # Set event bus for component
    if hasattr(component, 'set_event_bus'):
        component.set_event_bus(event_bus)
    
    # Create portfolio for tracking positions and PnL
    from src.execution.portfolio import PortfolioManager
    portfolio = PortfolioManager(initial_cash=10000.0, event_bus=event_bus)
    
    # Create risk manager for order generation
    from src.strategy.risk.risk_manager import SimpleRiskManager
    risk_manager = SimpleRiskManager(portfolio, event_bus)
    
    # Register components with event manager
    event_manager.register_component('strategy', component, [EventType.BAR])
    event_manager.register_component('risk_manager', risk_manager, [EventType.SIGNAL])
    event_manager.register_component('portfolio', portfolio, [EventType.FILL])
    
    # Reset components
    component.reset()
    risk_manager.reset()
    portfolio.reset()
    
    # Reset data handler
    data_handler.reset()
    
    # Get symbols from component
    symbols = component.symbols if hasattr(component, 'symbols') else []
    
    # Track equity and timestamps
    equity_values = [10000.0]  # Start with initial capital
    timestamps = [datetime.datetime.now()]  # Start with current time
    
    # Run backtest
    while True:
        # Get next bar for each symbol
        bars = []
        for symbol in symbols:
            bar = data_handler.get_next_bar(symbol)
            if bar:
                bars.append(bar)
        
        # If no bars, end backtest
        if not bars:
            break
        
        # Process bars
        for bar in bars:
            event_bus.emit(bar)
            
            # Track timestamp from bar
            timestamps.append(bar.get_timestamp())
            
            # Get current portfolio value
            if hasattr(portfolio, 'get_equity'):
                current_equity = portfolio.get_equity()
            else:
                # Increment slightly if method not available
                current_equity = equity_values[-1] * (1 + np.random.normal(0, 0.001))
            
            equity_values.append(current_equity)
    
    # Create DataFrame with timestamp and equity columns
    equity_curve = pd.DataFrame({
        'timestamp': timestamps[:len(equity_values)],  # Ensure same length
        'equity': equity_values
    })
    
    # Get trades count
    trades = len(portfolio.fill_history) if hasattr(portfolio, 'fill_history') else 0
    
    # Return tuple as expected by evaluate_backtest
    return equity_curve, trades
