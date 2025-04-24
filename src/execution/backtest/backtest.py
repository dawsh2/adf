"""
Streamlined backtest implementation that leverages dependency injection and configuration.
"""
import logging
import pandas as pd
import numpy as np
import datetime
from typing import Dict, Any, List, Optional

from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType
from src.execution.portfolio import PortfolioManager
from src.execution.brokers.simulated import SimulatedBroker
from src.strategy.risk.risk_manager import SimplePassthroughRiskManager
from src.execution.trade_tracker import TradeTracker

logger = logging.getLogger(__name__)

class BacktestRunner:
    """Streamlined backtesting system that leverages dependency injection."""
    
    def __init__(self, config=None):
        """
        Initialize the backtest runner.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or {}
        self.initial_cash = self.config.get('initial_cash', 10000.0)
        self.event_bus = None
        self.event_manager = None
        self.portfolio = None
        self.broker = None
        self.risk_manager = None
        self.trade_tracker = None
    
    def setup(self):
        """Set up the backtest environment."""
        # Create event system
        self.event_bus = EventBus()
        self.event_manager = EventManager(self.event_bus)
        
        # Create portfolio
        self.portfolio = PortfolioManager(initial_cash=self.initial_cash)
        self.portfolio.set_event_bus(self.event_bus)
        
        # Create broker
        slippage_model = self.config.get('slippage_model')
        self.broker = SimulatedBroker(
            fill_emitter=self.event_bus,
            slippage_model=slippage_model
        )
        self.broker.set_event_bus(self.event_bus)
        
        # Create risk manager
        self.risk_manager = SimplePassthroughRiskManager(
            portfolio=self.portfolio,
            event_bus=self.event_bus
        )
        self.risk_manager.broker = self.broker
        
        # Create trade tracker
        self.trade_tracker = TradeTracker(initial_cash=self.initial_cash)
        
        # Register event handlers
        self._register_event_handlers()
        
        return self
    
    def _register_event_handlers(self):
        """Register event handlers for the backtest."""
        # Create market data dictionary for tracking
        self.market_prices = {}
        
        # Register core event handlers
        self.event_bus.register(EventType.SIGNAL, self.risk_manager.on_signal)
        self.event_bus.register(EventType.ORDER, self.broker.place_order)
        self.event_bus.register(EventType.FILL, self.portfolio.on_fill)
        
        # Register event tracking handler
        self.event_bus.register(EventType.BAR, self._on_bar)
        self.event_bus.register(EventType.FILL, self._on_fill)
        
        # Track event counts for reporting
        self.event_counts = {
            'bars': 0,
            'signals': 0,
            'orders': 0,
            'fills': 0
        }
        
        # Register count trackers
        self.event_bus.register(EventType.BAR, lambda e: self._count_event('bars'))
        self.event_bus.register(EventType.SIGNAL, lambda e: self._count_event('signals'))
        self.event_bus.register(EventType.ORDER, lambda e: self._count_event('orders'))
        self.event_bus.register(EventType.FILL, lambda e: self._count_event('fills'))
    
    def _count_event(self, event_type):
        """Increment event counter."""
        self.event_counts[event_type] += 1
    
    def _on_bar(self, event):
        """Process bar events for market data tracking."""
        # Update market prices
        symbol = event.get_symbol()
        price = event.get_close()
        timestamp = event.get_timestamp()
        
        self.market_prices[symbol] = price
        
        # Update broker's market data
        self.broker.update_market_data(symbol, {
            'price': price,
            'timestamp': timestamp
        })
        
        # Update trade tracker equity
        self.trade_tracker.update_equity(timestamp, self.market_prices)
    
    def _on_fill(self, event):
        """Process fill events for trade tracking."""
        # Extract fill details
        fill_data = {
            'timestamp': event.get_timestamp(),
            'symbol': event.get_symbol(),
            'direction': event.get_direction(),
            'quantity': event.get_quantity(),
            'price': event.get_price(),
            'commission': event.get_commission() if hasattr(event, 'get_commission') else 0.0
        }
        
        # Process with trade tracker
        self.trade_tracker.process_fill(fill_data)
    
    def run_strategy(self, strategy, data_handler, start_date=None, end_date=None):
        """
        Run a backtest with a single strategy.
        
        Args:
            strategy: Strategy to test
            data_handler: Data handler with market data
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            tuple: (equity_curve, trades)
        """
        # Reset all components
        self._reset_components()
        
        # Connect strategy to event system
        if hasattr(strategy, 'set_event_bus'):
            strategy.set_event_bus(self.event_bus)
        
        # Register strategy with event manager
        self.event_manager.register_component('strategy', strategy, [EventType.BAR])
        
        # Reset strategy
        strategy.reset()
        
        # Get symbols from strategy
        symbols = strategy.symbols if hasattr(strategy, 'symbols') else []
        
        # Process each symbol
        last_timestamp = None
        for symbol in symbols:
            # Reset data handler for this symbol
            data_handler.reset()
            
            # Process each bar
            while True:
                bar = data_handler.get_next_bar(symbol)
                if bar is None:
                    break
                
                # Get timestamp
                timestamp = bar.get_timestamp()
                last_timestamp = timestamp
                
                # Apply date filtering
                if start_date is not None and timestamp < start_date:
                    continue
                    
                if end_date is not None and timestamp > end_date:
                    break
                
                # Emit bar event
                self.event_bus.emit(bar)
        
        # Liquidate positions at the end
        if last_timestamp:
            logger.info("Liquidating positions at the end of the backtest")
            self.trade_tracker.liquidate_positions(last_timestamp, self.market_prices)
        
        # Get results
        equity_curve = self.trade_tracker.get_equity_curve()
        trades = self.trade_tracker.get_closed_trades()
        
        # Log statistics
        logger.info(f"=== BACKTEST COMPLETE ===")
        logger.info(f"Events processed: {self.event_counts}")
        logger.info(f"Trades executed: {len(trades)}")
        
        # Return results
        return equity_curve, trades
    
    def _reset_components(self):
        """Reset all backtest components."""
        self.portfolio.reset()
        self.risk_manager.reset()
        
        # Reset event counts
        self.event_counts = {
            'bars': 0,
            'signals': 0,
            'orders': 0,
            'fills': 0
        }
        
        # Clear market prices
        self.market_prices = {}
    
    def compare_strategies(self, strategies, data_handler, strategy_names=None, start_date=None, end_date=None):
        """
        Compare multiple strategies on the same data.
        
        Args:
            strategies: List of strategies to compare
            data_handler: Data handler with market data
            strategy_names: Optional list of strategy names
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            dict: Comparison results
        """
        results = []
        
        # Use default names if not provided
        if strategy_names is None:
            strategy_names = [getattr(s, 'name', f"Strategy {i+1}") for i, s in enumerate(strategies)]
        
        # Run backtest for each strategy
        for i, strategy in enumerate(strategies):
            logger.info(f"Running backtest for {strategy_names[i]}...")
            
            # Reset data handler
            data_handler.reset()
            
            # Run backtest
            equity_curve, trades = self.run_strategy(strategy, data_handler, start_date, end_date)
            
            # Calculate performance metrics
            return_pct = ((equity_curve['equity'].iloc[-1] / equity_curve['equity'].iloc[0]) - 1) * 100
            
            # Get trade statistics
            trade_stats = self.trade_tracker.get_trade_statistics()
            
            # Check for regime filter stats
            regime_stats = None
            if hasattr(strategy, 'get_regime_stats'):
                regime_stats = strategy.get_regime_stats()
            
            # Store results
            results.append({
                'name': strategy_names[i],
                'equity_curve': equity_curve,
                'trades': trades,
                'return': return_pct,
                'trade_stats': trade_stats,
                'regime_stats': regime_stats
            })
            
            # Log performance
            logger.info(f"{strategy_names[i]}: Return={return_pct:.2f}%, Trades={len(trades)}")
            
            # Log regime stats if available
            if regime_stats:
                passed = regime_stats.get('passed_signals', 0)
                filtered = regime_stats.get('filtered_signals', 0)
                total = passed + filtered
                filter_rate = filtered / total * 100 if total > 0 else 0
                
                logger.info(f"Regime Stats: Passed={passed}, Filtered={filtered}, Rate={filter_rate:.2f}%")
        
        return results

# Function for backward compatibility with existing code
def run_backtest(component, data_handler, start_date=None, end_date=None, **kwargs):
    """
    Simplified run_backtest function for backward compatibility.
    
    Args:
        component: Strategy component to test
        data_handler: Data handler with market data
        start_date: Start date for backtest
        end_date: End date for backtest
        **kwargs: Additional parameters
            
    Returns:
        tuple: (equity_curve, trades)
    """
    # Create and set up backtest runner
    runner = BacktestRunner(config=kwargs)
    runner.setup()
    
    # Run backtest
    return runner.run_strategy(component, data_handler, start_date, end_date)
