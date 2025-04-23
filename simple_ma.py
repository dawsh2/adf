#!/usr/bin/env python
"""
Regime-Filtered Moving Average Crossover Strategy

This script implements a composite strategy where a regime filter determines when
the underlying moving average crossover strategy can go long or short.
"""
import os
import datetime
import logging
import itertools
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from enum import Enum

# Import strategy and core components
from src.strategy.strategies.ma_crossover import MovingAverageCrossoverStrategy
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType, FillEvent
from src.core.events.event_utils import create_signal_event
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler
from src.execution.portfolio import PortfolioManager
from src.strategy.risk.risk_manager import SimpleRiskManager
from src.execution.position import Position
from src.analytics.performance.calculator import PerformanceCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SAMPLE_SYMBOL = "SAMPLE"
DATA_DIR = "./data"
INITIAL_CAPITAL = 100000.0
START_DATE = "2020-01-01"
END_DATE = "2020-12-31"

# Parameter grid for optimization
PARAM_GRID = {
    'fast_window': [1, 2, 3, 4, 5, 10, 15, 20, 40],
    'slow_window': [5, 10, 15, 20, 30, 40, 50, 100],
    'regime_window': [20, 40, 60, 100]  # Window for regime detection
}

# Define market regimes
class MarketRegime(Enum):
    """Enumeration of market regime types."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class SimpleRegimeDetector:
    """
    Simple regime detector based on price trends.
    
    This detector identifies:
    - BULLISH: When price is above its moving average and trending up
    - BEARISH: When price is below its moving average and trending down
    - SIDEWAYS: When price is oscillating around its moving average
    """
    
    def __init__(self, window=50):
        """
        Initialize the regime detector.
        
        Args:
            window: Window size for regime detection
        """
        self.window = window
        self.prices = []
        self.regimes = []
        self.current_regime = MarketRegime.UNKNOWN
    
    def update(self, price):
        """
        Update detector with new price and detect current regime.
        
        Args:
            price: Current price
            
        Returns:
            MarketRegime: Detected market regime
        """
        # Add price to history
        self.prices.append(price)
        
        # Keep prices limited to twice the window size
        if len(self.prices) > self.window * 2:
            self.prices = self.prices[-self.window * 2:]
            
        # Need enough data for regime detection
        if len(self.prices) < self.window:
            self.current_regime = MarketRegime.UNKNOWN
            self.regimes.append(self.current_regime)
            return self.current_regime
            
        # Get relevant price data
        recent_prices = self.prices[-self.window:]
        
        # Calculate simple moving average
        ma = sum(recent_prices) / len(recent_prices)
        
        # Calculate short-term trend (last 20% of the window)
        trend_window = max(int(self.window * 0.2), 2)
        recent_trend = self.prices[-trend_window:]
        
        trend_direction = 0
        if len(recent_trend) >= 2:
            # Positive slope indicates uptrend, negative indicates downtrend
            trend_direction = recent_trend[-1] - recent_trend[0]
        
        # Current price relative to moving average
        current_price = self.prices[-1]
        price_position = current_price - ma
        
        # Detect regime
        if price_position > 0 and trend_direction > 0:
            # Price above MA and trending up
            self.current_regime = MarketRegime.BULLISH
        elif price_position < 0 and trend_direction < 0:
            # Price below MA and trending down
            self.current_regime = MarketRegime.BEARISH
        else:
            # Mixed signals, likely sideways
            self.current_regime = MarketRegime.SIDEWAYS
        
        # Store regime
        self.regimes.append(self.current_regime)
        
        return self.current_regime
    
    def get_current_regime(self):
        """Get the current market regime."""
        return self.current_regime
    
    def reset(self):
        """Reset the detector state."""
        self.prices = []
        self.regimes = []
        self.current_regime = MarketRegime.UNKNOWN


class RegimeFilteredStrategy:
    """
    Regime-filtered moving average crossover strategy.
    
    This strategy applies a regime filter to the signals generated
    by a moving average crossover strategy, restricting when the
    strategy can go long or short based on the detected market regime.
    """
    
    def __init__(self, base_strategy, regime_detector, regime_rules=None):
        """
        Initialize the regime-filtered strategy.
        
        Args:
            base_strategy: Base strategy (e.g., MA crossover)
            regime_detector: Regime detector instance
            regime_rules: Optional rules for which signals are allowed in each regime
        """
        self.strategy = base_strategy
        self.detector = regime_detector
        
        # Default regime rules if none provided
        self.regime_rules = regime_rules or {
            MarketRegime.BULLISH: {'allow_long': True, 'allow_short': False},  # Only allow long in bullish
            MarketRegime.BEARISH: {'allow_long': False, 'allow_short': True},  # Only allow short in bearish
            MarketRegime.SIDEWAYS: {'allow_long': True, 'allow_short': True},  # Allow both in sideways
            MarketRegime.UNKNOWN: {'allow_long': True, 'allow_short': True}    # Allow both when regime unknown
        }
        
        # Get properties from base strategy
        self.name = f"regime_filtered_{base_strategy.name}"
        self.symbols = base_strategy.symbols
    
    def set_event_bus(self, event_bus):
        """Pass through event bus to base strategy."""
        self.strategy.set_event_bus(event_bus)
        return self
    
    def on_bar(self, bar):
        """
        Process a bar event with regime filtering.
        
        Args:
            bar: BarEvent to process
            
        Returns:
            Optional[SignalEvent]: Generated signal event or None
        """
        # Update regime detector with latest price
        price = bar.get_close()
        current_regime = self.detector.update(price)
        
        # Process bar with base strategy
        signal = self.strategy.on_bar(bar)
        
        # If no signal, nothing to filter
        if signal is None:
            return None
            
        # Get signal direction
        signal_value = signal.get_signal_value()
        is_long = signal_value > 0
        is_short = signal_value < 0
        
        # Get allowed actions for current regime
        regime_rule = self.regime_rules.get(current_regime, 
                                            {'allow_long': True, 'allow_short': True})
        
        # Filter signals based on regime
        if (is_long and not regime_rule['allow_long']) or (is_short and not regime_rule['allow_short']):
            # Signal does not match allowed actions for current regime
            logger.debug(f"Signal {signal_value} filtered out by {current_regime.value} regime")
            return None
            
        # Signal passes regime filter
        logger.debug(f"Signal {signal_value} allowed in {current_regime.value} regime")
        return signal
    
    def reset(self):
        """Reset the strategy and detector."""
        self.strategy.reset()
        self.detector.reset()


class DirectBroker:
    """Direct broker implementation that updates portfolio without event system."""
    
    def __init__(self, portfolio):
        """Initialize direct broker with portfolio reference."""
        self.portfolio = portfolio
        self.orders_processed = 0
        self.fills_created = 0
        self.market_prices = {}  # Store latest market prices
    
    def update_market_price(self, symbol, price):
        """Update the market price for a symbol."""
        self.market_prices[symbol] = price
    
    def place_order(self, order):
        """Execute order directly and update portfolio."""
        self.orders_processed += 1
        
        symbol = order.get_symbol()
        direction = order.get_direction()
        quantity = order.get_quantity()
        price = order.get_price()
        
        # Limit order size to a percentage of account to avoid catastrophic losses
        max_position_size = max(1, int(self.portfolio.cash * 0.1 / price))  # Max 10% of account per position
        if quantity > max_position_size:
            logger.debug(f"Reducing order size from {quantity} to {max_position_size} to limit risk")
            quantity = max_position_size
            
        if quantity == 0:
            logger.debug(f"Order size reduced to zero, skipping order")
            return
        
        logger.debug(f"DirectBroker: Processing order #{self.orders_processed}: {symbol} {direction} {quantity} @ {price}")
        
        # Calculate commission (fixed + percentage)
        commission = min(max(1.0, 0.001 * price * quantity), 20.0)  # More realistic commission model
        
        # Update portfolio directly
        if direction == 'BUY':
            # Check if we have enough cash
            cost = quantity * price + commission
            if cost > self.portfolio.cash:
                # Scale down to what we can afford
                affordable_quantity = max(1, int((self.portfolio.cash - commission) / price))
                if affordable_quantity <= 0:
                    logger.debug(f"Insufficient funds to place buy order, needed ${cost:.2f}, have ${self.portfolio.cash:.2f}")
                    return
                logger.debug(f"Reducing order size from {quantity} to {affordable_quantity} due to insufficient funds")
                quantity = affordable_quantity
                cost = quantity * price + commission
            
            # If no position exists, create one
            if symbol not in self.portfolio.positions:
                self.portfolio.positions[symbol] = Position(symbol)
            
            # Add to position
            position = self.portfolio.positions[symbol]
            position.add_quantity(quantity, price)
            
            # Update cash
            self.portfolio.cash -= cost
            
        elif direction == 'SELL':
            # Check if we have the position to sell
            position = self.portfolio.get_position(symbol)
            if position is None or position.quantity < quantity:
                # We're creating or increasing a short position
                if position is None:
                    self.portfolio.positions[symbol] = Position(symbol)
                    position = self.portfolio.positions[symbol]
                
                # Use reduce_quantity method which handles short positions
                position.reduce_quantity(quantity, price)
                
                # Update cash (add proceeds - commission)
                self.portfolio.cash += (quantity * price - commission)
            else:
                # We're reducing a long position
                position.reduce_quantity(quantity, price)
                
                # Update cash
                self.portfolio.cash += (quantity * price - commission)
        
        self.fills_created += 1
        
        # Log portfolio state after fill
        logger.debug(f"Fill #{self.fills_created}: {symbol} {direction} {quantity} @ {price}")
        logger.debug(f"Portfolio after fill - Cash: ${self.portfolio.cash:.2f}")
        
        position = self.portfolio.positions.get(symbol)
        if position and position.quantity != 0:
            logger.debug(f"Position: {position.quantity} shares @ {position.cost_basis:.2f}")
    
    def get_stats(self):
        """Get broker statistics."""
        return {
            'orders_processed': self.orders_processed,
            'fills_created': self.fills_created
        }
        
    def close_all_positions(self, symbol):
        """
        Close all positions at current market prices.
        
        Args:
            symbol: Symbol to close positions for
        """
        position = self.portfolio.get_position(symbol)
        if position is None or position.quantity == 0:
            logger.debug("No positions to close")
            return
            
        # Get current market price
        price = self.market_prices.get(symbol)
        if price is None:
            logger.warning(f"No market price available for {symbol}, cannot close position")
            return
            
        quantity = abs(position.quantity)
        direction = 'SELL' if position.quantity > 0 else 'BUY'
        
        logger.info(f"Closing position at end of backtest: {symbol} {direction} {quantity} @ {price}")
        
        # Calculate commission
        commission = min(max(1.0, 0.001 * price * quantity), 20.0)
        
        if direction == 'SELL':
            # Closing a long position
            self.portfolio.cash += (quantity * price - commission)
        else:
            # Closing a short position
            cost = quantity * price + commission
            if cost > self.portfolio.cash:
                logger.warning(f"Insufficient funds to close short position, needed ${cost:.2f}, have ${self.portfolio.cash:.2f}")
                # Close what we can afford
                affordable_quantity = max(1, int((self.portfolio.cash - commission) / price))
                if affordable_quantity <= 0:
                    return
                quantity = affordable_quantity
                cost = quantity * price + commission
                logger.warning(f"Partially closing position: {affordable_quantity} of {abs(position.quantity)} shares")
            
            self.portfolio.cash -= cost
        
        # Update position
        if direction == 'SELL':
            position.reduce_quantity(quantity, price)
        else:
            position.add_quantity(quantity, price)
            
        self.orders_processed += 1
        self.fills_created += 1
        
        logger.info(f"Position closed: {symbol}, Final cash: ${self.portfolio.cash:.2f}")


def run_regime_filtered_backtest(symbol, start_date, end_date, params, inverse_signals=False):
    """
    Run a backtest with regime-filtered strategy.
    
    Args:
        symbol: Symbol to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        params: Strategy parameters
        inverse_signals: Whether to invert the strategy signals
        
    Returns:
        dict: Performance metrics
    """
    # Create core components
    portfolio = PortfolioManager(initial_cash=INITIAL_CAPITAL)
    
    # Create direct broker
    broker = DirectBroker(portfolio)
    
    # Create base strategy
    base_strategy = MovingAverageCrossoverStrategy(
        name="ma_crossover",
        symbols=symbol,
        fast_window=params['fast_window'],
        slow_window=params['slow_window']
    )
    
    # Create regime detector
    regime_detector = SimpleRegimeDetector(window=params.get('regime_window', 50))
    
    # Create regime-filtered strategy
    strategy = RegimeFilteredStrategy(base_strategy, regime_detector)
    
    # Create risk manager with reference to broker
    risk_manager = SimpleRiskManager(portfolio, None)
    
    # Reduce fixed size for safer trading
    risk_manager.fixed_size = 10  # Smaller position size
    
    # Override risk manager's emit method to call broker directly
    def direct_emit_order(order):
        broker.place_order(order)
        return True
    
    # Replace risk manager's order emission method with direct broker call
    risk_manager._emit_order = direct_emit_order
    
    # Create data components
    data_source = CSVDataSource(data_dir=DATA_DIR)
    data_handler = HistoricalDataHandler(data_source, None)
    
    # Load data
    data_handler.load_data(symbol, start_date=start_date, end_date=end_date, timeframe='1m')
    
    # Run backtest
    signal_type = "INVERTED" if inverse_signals else "NORMAL"
    filter_type = "with REGIME FILTERING"
    logger.info(f"Running {signal_type} backtest {filter_type} with params: {params}")
    
    # Process all bars
    bar_count = 0
    equity_curve = []
    signal_count = 0
    regime_stats = {regime: 0 for regime in MarketRegime}
    filtered_signals = 0
    passed_signals = 0
    
    while True:
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break
            
        bar_count += 1
        
        # Update broker's market data
        close_price = bar.get_close()
        broker.update_market_price(symbol, close_price)
        
        # Get the base strategy signal before filtering
        base_signal = base_strategy.on_bar(bar)
        
        # Process bar with regime-filtered strategy directly
        signal = strategy.on_bar(bar)
        
        # Count regimes
        current_regime = regime_detector.get_current_regime()
        regime_stats[current_regime] += 1
        
        # Count filtered signals
        if base_signal and not signal:
            filtered_signals += 1
        elif base_signal and signal:
            passed_signals += 1
        
        # If signal generated and inverse_signals is true, flip the signal
        if signal and inverse_signals:
            # Get original signal value
            original_value = signal.get_signal_value()
            
            # Create a new signal with the opposite direction
            signal = create_signal_event(
                signal_value=-original_value,  # Invert the signal
                price=signal.get_price(),
                symbol=signal.get_symbol(),
                rule_id=signal.data.get('rule_id'),
                confidence=signal.data.get('confidence', 1.0),
                metadata=signal.data.get('metadata'),
                timestamp=signal.get_timestamp()
            )
        
        # If signal generated, process with risk manager
        if signal:
            signal_count += 1
            direction_desc = "BUY" if signal.get_signal_value() == 1 else "SELL"
            logger.debug(f"Signal #{signal_count}: {signal.get_symbol()} {direction_desc} @ {signal.get_price()}")
            risk_manager.on_signal(signal)
        
        # After processing, record portfolio state
        timestamp = bar.get_timestamp()
        
        # Calculate current equity
        position = portfolio.get_position(symbol)
        position_value = 0
        
        if position and position.quantity != 0:
            position_value = position.quantity * close_price
        
        current_equity = portfolio.cash + position_value
        
        # Store equity point
        equity_curve.append({
            'timestamp': timestamp,
            'equity': current_equity
        })
        
        if bar_count % 100 == 0:
            logger.debug(f"Processed {bar_count} bars... Current equity: ${current_equity:.2f}")
    
    # Close all positions at the end of the backtest
    logger.info("Closing all positions at the end of the backtest")
    broker.close_all_positions(symbol)
    
    # Calculate final equity after closing positions
    final_cash = portfolio.cash
    final_equity = final_cash  # All positions should be closed, so equity equals cash
    
    # Add final equity point after position closing
    if equity_curve:
        last_timestamp = equity_curve[-1]['timestamp']
        equity_curve.append({
            'timestamp': last_timestamp,  # Use the same timestamp as the last bar
            'equity': final_equity
        })
    
    # Print regime statistics
    logger.info("\nRegime Statistics:")
    for regime, count in regime_stats.items():
        percentage = (count / bar_count) * 100 if bar_count > 0 else 0
        logger.info(f"  {regime.value}: {count} bars ({percentage:.1f}%)")
    
    # Print signal filtering statistics
    total_signals = filtered_signals + passed_signals
    if total_signals > 0:
        logger.info("\nSignal Filtering Statistics:")
        logger.info(f"  Total base signals: {total_signals}")
        logger.info(f"  Filtered signals: {filtered_signals} ({filtered_signals/total_signals*100:.1f}%)")
        logger.info(f"  Passed signals: {passed_signals} ({passed_signals/total_signals*100:.1f}%)")
    
    # Print broker stats
    broker_stats = broker.get_stats()
    logger.info(f"Broker stats: Orders={broker_stats['orders_processed']}, Fills={broker_stats['fills_created']}")
    
    # Print final portfolio state
    logger.info(f"Processed total of {bar_count} bars")
    logger.info(f"Final cash: ${portfolio.cash:.2f}")
    
    # Calculate performance metrics
    calculator = PerformanceCalculator()
    metrics = calculator.calculate(equity_curve)
    
    signal_type = "inverted" if inverse_signals else "normal"
    logger.info(f"{signal_type.capitalize()} regime-filtered backtest complete. Final equity: ${metrics['final_equity']:.2f}")
    logger.info(f"Total return: {metrics['total_return_pct']:.2f}%")
    logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    
    return metrics


def grid_search_optimization(symbol, start_date, end_date, param_grid):
    """
    Perform grid search optimization testing both normal and inverted signals.
    
    Args:
        symbol: Symbol to trade
        start_date: Start date for optimization
        end_date: End date for optimization
        param_grid: Parameter grid to search
        
    Returns:
        tuple: (best_params, best_metrics, all_results, best_inverted)
    """
    # Generate all parameter combinations
    param_names = sorted(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    combinations = list(itertools.product(*param_values))
    
    # Filter valid combinations
    valid_combinations = []
    for combo in combinations:
        params = {name: value for name, value in zip(param_names, combo)}
        if 'fast_window' in params and 'slow_window' in params:
            if params['fast_window'] < params['slow_window']:
                valid_combinations.append(params)
    
    logger.info(f"Grid search: evaluating {len(valid_combinations)} parameter combinations x 2 (normal + inverted)")
    
    # Track best parameters
    best_params = None
    best_metrics = None
    best_return = float('-inf')
    best_inverted = False
    all_results = []
    
    # Test each parameter combination
    for params in valid_combinations:
        # Test with normal signals
        logger.info(f"Testing params: {params} (normal signals)")
        metrics_normal = run_regime_filtered_backtest(symbol, start_date, end_date, params, inverse_signals=False)
        
        # Test with inverted signals
        logger.info(f"Testing params: {params} (inverted signals)")
        metrics_inverted = run_regime_filtered_backtest(symbol, start_date, end_date, params, inverse_signals=True)
        
        # Store results
        result_normal = {
            'params': params.copy(),
            'metrics': metrics_normal,
            'inverted': False
        }
        result_inverted = {
            'params': params.copy(),
            'metrics': metrics_inverted,
            'inverted': True
        }
        all_results.append(result_normal)
        all_results.append(result_inverted)
        
        # Compare returns and update best if improved
        return_normal = metrics_normal['total_return_pct']
        return_inverted = metrics_inverted['total_return_pct']
        
        logger.info(f"Parameters: {params}, Normal return: {return_normal:.2f}%, " +
                   f"Inverted return: {return_inverted:.2f}%")
        
        if return_normal > best_return:
            best_return = return_normal
            best_params = params.copy()
            best_metrics = metrics_normal
            best_inverted = False
            logger.info(f"New best strategy (normal): {params}, Return: {return_normal:.2f}%")
            
        if return_inverted > best_return:
            best_return = return_inverted
            best_params = params.copy()
            best_metrics = metrics_inverted
            best_inverted = True
            logger.info(f"New best strategy (inverted): {params}, Return: {return_inverted:.2f}%")
    
    return best_params, best_metrics, all_results, best_inverted


def main():
    """Main function to run the optimization and backtest."""
    logger.info("Starting Regime-Filtered MA Crossover strategy optimization")
    
    # Run grid search optimization
    best_params, best_metrics, all_results, best_inverted = grid_search_optimization(
        SAMPLE_SYMBOL, START_DATE, END_DATE, PARAM_GRID
    )
    
    # Print best results
    logger.info("\n" + "="*50)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("="*50)
    
    if best_params is None:
        logger.warning("No valid parameters found!")
        return
    
    signal_type = "INVERTED" if best_inverted else "NORMAL"
    logger.info(f"Best parameters: {best_params} with {signal_type} signals")
    logger.info(f"Best return: {best_metrics['total_return_pct']:.2f}%")
    logger.info(f"Best Sharpe ratio: {best_metrics['sharpe_ratio']:.2f}")
    logger.info(f"Max drawdown: {best_metrics['max_drawdown_pct']:.2f}%")
    
    # Plot results table
    logger.info("\nAll Results (sorted by return):")
    logger.info(f"{'Fast MA':<8} {'Slow MA':<8} {'Regime':<8} {'Signals':<10} {'Return %':<10} {'Sharpe':<8} {'Drawdown %':<10}")
    logger.info("-"*70)
    
    # Sort results by return (descending)
    sorted_results = sorted(all_results, key=lambda x: x['metrics']['total_return_pct'], reverse=True)
    
    # Print top 15 results
    for i, result in enumerate(sorted_results[:15]):
        params = result['params']
        metrics = result['metrics']
        signal_type = "INVERTED" if result['inverted'] else "NORMAL"
        logger.info(f"{params['fast_window']:<8} {params['slow_window']:<8} {params.get('regime_window', 'N/A'):<8} "
                   f"{signal_type:<10} {metrics['total_return_pct']:>9.2f}% {metrics['sharpe_ratio']:>7.2f} "
                   f"{metrics['max_drawdown_pct']:>9.2f}%")
    
    logger.info("\nOptimization complete!")
    
    # Run final backtest with best parameters and render equity curve
    logger.info("\nRunning final backtest with best parameters...")
    signal_type = "inverted" if best_inverted else "normal"
    run_regime_filtered_backtest(SAMPLE_SYMBOL, START_DATE, END_DATE, best_params, inverse_signals=best_inverted)
    
    logger.info(f"\nBest strategy found: Regime-Filtered MA Crossover with fast={best_params['fast_window']}, "
               f"slow={best_params['slow_window']}, regime_window={best_params.get('regime_window', 'N/A')}, "
               f"using {signal_type} signals")


if __name__ == "__main__":
    main()
