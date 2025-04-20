import datetime
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union

from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType

class BacktestRunner:
    """
    Runner for backtesting trading strategies.
    Coordinates the flow of data, events, and component interactions.
    """
    
    def __init__(self, data_handler, strategy, risk_manager, execution_engine, portfolio, 
                event_bus=None, event_manager=None):
        """
        Initialize the backtest runner.
        
        Args:
            data_handler: Data handler for market data
            strategy: Trading strategy to test
            risk_manager: Risk manager for position sizing
            execution_engine: Execution engine for order routing
            portfolio: Portfolio manager for tracking state
            event_bus: Event bus for communication
            event_manager: Event manager for component registration
        """
        self.data_handler = data_handler
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.execution_engine = execution_engine
        self.portfolio = portfolio
        
        # Create event system if not provided
        self.event_bus = event_bus or EventBus()
        self.event_manager = event_manager or EventManager(self.event_bus)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.equity_curve = []
        self.trade_history = []
        
        # Initialize event connections
        self._setup_event_connections()
    
    def _setup_event_connections(self):
        """Set up event connections between components."""
        # Set event bus on all components
        for component in [self.strategy, self.risk_manager, 
                         self.execution_engine, self.portfolio]:
            if hasattr(component, 'set_event_bus'):
                component.set_event_bus(self.event_bus)
        
        # Register components with event manager
        self.event_manager.register_component('strategy', self.strategy, [EventType.BAR])
        self.event_manager.register_component('risk', self.risk_manager, [EventType.SIGNAL])
        self.event_manager.register_component('execution', self.execution_engine, [EventType.ORDER])
        self.event_manager.register_component('portfolio', self.portfolio, [EventType.FILL])
    
    def run(self, symbols, start_date=None, end_date=None, timeframe='1d'):
        """
        Run the backtest.
        
        Args:
            symbols: Symbol or list of symbols to trade
            start_date: Start date for the backtest
            end_date: End date for the backtest
            timeframe: Timeframe for the data ('1d', '1h', etc.)
            
        Returns:
            dict: Backtest results
        """
        # Convert single symbol to list
        if isinstance(symbols, str):
            symbols = [symbols]
            
        self.logger.info(f"Starting backtest for {symbols} from {start_date} to {end_date}")
        
        # Reset components
        self._reset_components()
        
        # Load data
        self.data_handler.load_data(symbols, start_date, end_date, timeframe)
        
        # Track equity at each step
        timestamps = []
        
        # Process data bar by bar
        for symbol in symbols:
            bar_count = 0
            while True:
                bar = self.data_handler.get_next_bar(symbol)
                if bar is None:
                    break
                
                # Record equity after processing each bar
                timestamps.append(bar.get_timestamp())
                self.equity_curve.append({
                    'timestamp': bar.get_timestamp(),
                    'equity': self.portfolio.get_equity({symbol: bar.get_close()})
                })
                
                bar_count += 1
            
            self.logger.info(f"Processed {bar_count} bars for {symbol}")
        
        # Generate performance metrics
        results = self._calculate_performance()
        
        self.logger.info(f"Backtest completed. Final equity: ${results['final_equity']:,.2f}")
        
        return results
    
    def _reset_components(self):
        """Reset all components to initial state."""
        if hasattr(self.data_handler, 'reset'):
            self.data_handler.reset()
            
        if hasattr(self.strategy, 'reset'):
            self.strategy.reset()
            
        if hasattr(self.portfolio, 'reset'):
            self.portfolio.reset()
            
        # Clear tracking data
        self.equity_curve = []
        self.trade_history = []
    
    def _calculate_performance(self):
        """Calculate performance metrics."""
        # Convert equity curve to DataFrame
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.set_index('timestamp', inplace=True)
        else:
            # Create empty DataFrame with initial capital if no trades
            equity_df = pd.DataFrame({
                'equity': [self.portfolio.initial_cash]
            }, index=[datetime.datetime.now()])
        
        # Extract key metrics
        initial_equity = equity_df['equity'].iloc[0]
        final_equity = equity_df['equity'].iloc[-1]
        returns = (final_equity / initial_equity) - 1
        
        # Calculate drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] / equity_df['peak']) - 1
        max_drawdown = equity_df['drawdown'].min()
        
        # Get trade information
        if hasattr(self.portfolio, 'fill_history'):
            trade_count = len(self.portfolio.fill_history)
        else:
            trade_count = 0
        
        # Compile results
        results = {
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'return': returns,
            'return_pct': returns * 100,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'trade_count': trade_count,
            'equity_curve': equity_df
        }
        
        return results
    
    def get_equity_curve(self):
        """Get the equity curve as a DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.equity_curve)
        df.set_index('timestamp', inplace=True)
        return df
