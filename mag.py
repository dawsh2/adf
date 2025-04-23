#!/usr/bin/env python
"""
Regime-Adaptive Moving Average Crossover Strategy

This script implements a trading strategy that uses different optimized
parameters depending on the detected market regime.
"""
import os
import datetime
import logging
import itertools
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from enum import Enum
from collections import defaultdict

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
SAMPLE_SYMBOL = "SPY"
DATA_DIR = "./data"
INITIAL_CAPITAL = 100000.0
START_DATE = "2024-03-26"
END_DATE = "20204-04-26"

# Parameter grid for optimization
PARAM_GRID = {
    'fast_window': [1, 2, 3, 4, 5, 10, 15, 20, 40],
    'slow_window': [5, 10, 15, 20, 30, 40, 50, 100]
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


class RegimeAdaptiveStrategy:
    """
    Regime-adaptive moving average crossover strategy.
    
    This strategy adjusts its parameters based on the detected market regime,
    using optimized parameter sets for each regime type.
    """
    
    def __init__(self, regime_detector, regime_params=None):
        """
        Initialize the regime-adaptive strategy.
        
        Args:
            regime_detector: Regime detector instance
            regime_params: Dict mapping regimes to parameter sets
        """
        self.detector = regime_detector
        self.regime_params = regime_params or {}
        
        # Set default parameters for each regime if not provided
        for regime in MarketRegime:
            if regime not in self.regime_params:
                self.regime_params[regime] = {'fast_window': 5, 'slow_window': 20}
        
        # Create strategies for each regime
        self.strategies = {}
        for regime, params in self.regime_params.items():
            self.strategies[regime] = MovingAverageCrossoverStrategy(
                name=f"ma_{regime.value}",
                symbols=None,  # Will be set later
                fast_window=params.get('fast_window', 5),
                slow_window=params.get('slow_window', 20)
            )
        
        # Initialize common properties
        self.name = "regime_adaptive_ma"
        self.symbols = []
        self.event_bus = None
        
        # Tracking variables
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_changes = []
        self.parameter_changes = []
    
    def set_event_bus(self, event_bus):
        """Set event bus for all strategies."""
        self.event_bus = event_bus
        for strategy in self.strategies.values():
            strategy.set_event_bus(event_bus)
        return self
    
    def set_symbols(self, symbols):
        """Set symbols to trade."""
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        for strategy in self.strategies.values():
            strategy.symbols = self.symbols
    
    def on_bar(self, bar):
        """
        Process a bar event with regime-adaptive parameters.
        
        Args:
            bar: BarEvent to process
            
        Returns:
            Optional[SignalEvent]: Generated signal event or None
        """
        # Extract symbol and verify we should process it
        symbol = bar.get_symbol()
        if symbol not in self.symbols:
            return None
        
        # Update regime detector with latest price
        price = bar.get_close()
        new_regime = self.detector.update(price)
        
        # If regime changed, record it
        if new_regime != self.current_regime:
            self.regime_changes.append((bar.get_timestamp(), self.current_regime, new_regime))
            self.current_regime = new_regime
            logger.debug(f"Regime changed to {new_regime.value} at {bar.get_timestamp()}")
        
        # Get strategy for current regime
        active_strategy = self.strategies[self.current_regime]
        
        # Process bar with the regime-specific strategy
        signal = active_strategy.on_bar(bar)
        
        # If signal generated, attach regime information
        if signal:
            # Add regime info to metadata
            metadata = signal.data.get('metadata', {})
            metadata['regime'] = self.current_regime.value
            metadata['regime_params'] = self.regime_params[self.current_regime]
            signal.data['metadata'] = metadata
            
            logger.debug(f"Signal generated in {self.current_regime.value} regime: "
                        f"{symbol} {signal.get_signal_value()} @ {signal.get_price()}")
        
        return signal
    
    def reset(self):
        """Reset the strategy and its components."""
        self.detector.reset()
        for strategy in self.strategies.values():
            strategy.reset()
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_changes = []
        self.parameter_changes = []


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


def run_backtest_for_regime(symbol, start_date, end_date, params, regime=None, regime_window=50, inverse_signals=False):
    """
    Run a backtest evaluating a specific regime's parameters across all data,
    but only taking signals when in the specified regime.
    
    Args:
        symbol: Symbol to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        params: Strategy parameters
        regime: Specific regime to test parameters for (None for all data)
        regime_window: Window for regime detection
        inverse_signals: Whether to invert the strategy signals
        
    Returns:
        dict: Dict with performance metrics and regime stats
    """
    # Create core components
    portfolio = PortfolioManager(initial_cash=INITIAL_CAPITAL)
    
    # Create direct broker
    broker = DirectBroker(portfolio)
    
    # Create strategy
    strategy = MovingAverageCrossoverStrategy(
        name="ma_crossover",
        symbols=symbol,
        fast_window=params['fast_window'],
        slow_window=params['slow_window']
    )
    
    # Create regime detector for filtering bars
    regime_detector = SimpleRegimeDetector(window=regime_window)
    
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
    regime_desc = f"in {regime.value} regime" if regime else "across all regimes"
    logger.info(f"Running {signal_type} backtest {regime_desc} with params: {params}")
    
    # Process all bars
    bar_count = 0
    equity_curve = []
    signal_count = 0
    regime_stats = {r: 0 for r in MarketRegime}
    regime_returns = {r: [] for r in MarketRegime}
    current_regime_equity = {r: None for r in MarketRegime}
    
    while True:
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break
            
        bar_count += 1
        
        # Update broker's market data
        close_price = bar.get_close()
        broker.update_market_price(symbol, close_price)
        
        # Get current regime
        current_regime = regime_detector.update(close_price)
        regime_stats[current_regime] += 1
        
        # Process bar with strategy directly - this generates signals for all bars
        signal = strategy.on_bar(bar)
        
        # If signal generated, apply regime filtering if requested
        if signal:
            # If we're testing for a specific regime, only consider signals in that regime
            if regime and current_regime != regime:
                # Skip signals from other regimes
                signal = None
                
            # If we're using inverted signals, invert the signal
            elif inverse_signals:
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
        
        # If signal generated and passes regime filter, process with risk manager
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
            'equity': current_equity,
            'regime': current_regime.value
        })
        
        # Track regime-specific returns
        if current_regime_equity[current_regime] is None:
            current_regime_equity[current_regime] = current_equity
        else:
            # Calculate return since last bar in this regime
            prev_equity = current_regime_equity[current_regime]
            pct_change = (current_equity / prev_equity) - 1
            regime_returns[current_regime].append(pct_change)
            current_regime_equity[current_regime] = current_equity
        
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
        last_regime = equity_curve[-1]['regime']
        equity_curve.append({
            'timestamp': last_timestamp,  # Use the same timestamp as the last bar
            'equity': final_equity,
            'regime': last_regime
        })
    
    # Calculate performance metrics
    if equity_curve:
        calculator = PerformanceCalculator()
        metrics = calculator.calculate(equity_curve)
        
        # Calculate regime-specific performance metrics
        regime_metrics = {}
        for r in MarketRegime:
            returns = regime_returns[r]
            if returns:
                ann_return = np.mean(returns) * 252 * 100 if returns else 0  # Annualized return in percent
                volatility = np.std(returns) * np.sqrt(252) * 100 if len(returns) > 1 else 0  # Annualized volatility
                sharpe = (ann_return / volatility) if volatility > 0 else 0
                regime_metrics[r] = {
                    'count': regime_stats[r],
                    'ann_return': ann_return,
                    'volatility': volatility,
                    'sharpe': sharpe
                }
            else:
                regime_metrics[r] = {
                    'count': regime_stats[r],
                    'ann_return': 0,
                    'volatility': 0,
                    'sharpe': 0
                }
    else:
        metrics = {}
        regime_metrics = {r: {'count': 0, 'ann_return': 0, 'volatility': 0, 'sharpe': 0} for r in MarketRegime}
    
    return {
        'metrics': metrics,
        'regime_stats': regime_stats,
        'regime_metrics': regime_metrics,
        'bar_count': bar_count,
        'signal_count': signal_count
    }
     
def optimize_regime_parameters(symbol, start_date, end_date, param_grid, regime_window=50, inverse_signals=False):
    """
    Optimize MA parameters separately for each market regime.
    
    Args:
        symbol: Symbol to trade
        start_date: Start date for optimization
        end_date: End date for optimization
        param_grid: Parameter grid to search
        regime_window: Window for regime detection
        inverse_signals: Whether to test inverted signals
        
    Returns:
        dict: Best parameters for each regime
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
    
    logger.info(f"Regime optimization: testing {len(valid_combinations)} parameter combinations for each regime")
    
    # First, run a backtest over all data to get regime distribution
    logger.info("Running initial backtest to determine regime distribution...")
    baseline_params = {'fast_window': 10, 'slow_window': 30}
    baseline_result = run_backtest_for_regime(
        symbol, start_date, end_date, baseline_params, regime=None, regime_window=regime_window
    )
    
    regime_stats = baseline_result['regime_stats']
    
    # Print regime distribution
    logger.info("Regime Distribution:")
    total_bars = sum(regime_stats.values())
    for regime, count in regime_stats.items():
        percentage = (count / total_bars) * 100 if total_bars > 0 else 0
        logger.info(f"  {regime.value}: {count} bars ({percentage:.1f}%)")
    
    # Determine which regimes have enough data for optimization
    min_regime_bars = 100  # Minimum bars needed for optimization
    regimes_to_optimize = []
    
    for regime, count in regime_stats.items():
        if count >= min_regime_bars:
            regimes_to_optimize.append(regime)
            logger.info(f"Will optimize for {regime.value} regime ({count} bars)")
        else:
            logger.info(f"Insufficient data for {regime.value} regime ({count} bars), using baseline parameters")
    
    # Optimize parameters for each regime
    best_params = {r: baseline_params.copy() for r in MarketRegime}  # Start with baseline
    best_metrics = {}
    all_results = {}
    
    for regime in regimes_to_optimize:
        logger.info(f"\n{'='*50}")
        logger.info(f"Optimizing parameters for {regime.value} regime")
        logger.info(f"{'='*50}")
        
        # Track best parameters for this regime
        regime_best_params = None
        regime_best_metrics = None
        regime_best_sharpe = float('-inf')
        regime_best_inverted = False
        regime_results = []
        
        # Test each parameter combination for this regime
        for params in valid_combinations:
            # Test with normal signals
            logger.info(f"Testing in {regime.value} regime: {params} (normal signals)")
            result_normal = run_backtest_for_regime(
                symbol, start_date, end_date, params, regime=regime, 
                regime_window=regime_window, inverse_signals=False
            )
            
            # Test with inverted signals if requested
            if inverse_signals:
                logger.info(f"Testing in {regime.value} regime: {params} (inverted signals)")
                result_inverted = run_backtest_for_regime(
                    symbol, start_date, end_date, params, regime=regime, 
                    regime_window=regime_window, inverse_signals=True
                )
            
            # Extract metrics
            metrics_normal = result_normal['metrics']
            normal_sharpe = metrics_normal.get('sharpe_ratio', 0)
            
            # Store results
            result_entry_normal = {
                'params': params.copy(),
                'metrics': metrics_normal,
                'inverted': False
            }
            regime_results.append(result_entry_normal)
            
            logger.info(f"  Normal signals - Sharpe: {normal_sharpe:.2f}, " +
                       f"Return: {metrics_normal.get('total_return_pct', 0):.2f}%")
            
            # Update best if improved
            if normal_sharpe > regime_best_sharpe:
                regime_best_sharpe = normal_sharpe
                regime_best_params = params.copy()
                regime_best_metrics = metrics_normal
                regime_best_inverted = False
            
            # Check inverted results if tested
            if inverse_signals:
                metrics_inverted = result_inverted['metrics']
                inverted_sharpe = metrics_inverted.get('sharpe_ratio', 0)
                
                result_entry_inverted = {
                    'params': params.copy(),
                    'metrics': metrics_inverted,
                    'inverted': True
                }
                regime_results.append(result_entry_inverted)
                
                logger.info(f"  Inverted signals - Sharpe: {inverted_sharpe:.2f}, " +
                           f"Return: {metrics_inverted.get('total_return_pct', 0):.2f}%")
                
                if inverted_sharpe > regime_best_sharpe:
                    regime_best_sharpe = inverted_sharpe
                    regime_best_params = params.copy()
                    regime_best_metrics = metrics_inverted
                    regime_best_inverted = True
        
        # Store best parameters for this regime
        if regime_best_params:
            best_params[regime] = regime_best_params
            best_metrics[regime] = regime_best_metrics
            
            # Add inverted flag to parameters if needed
            if regime_best_inverted:
                best_params[regime]['invert_signals'] = True
            
            logger.info(f"\nBest parameters for {regime.value} regime: {best_params[regime]}")
            logger.info(f"Best Sharpe ratio: {regime_best_sharpe:.2f}")
            logger.info(f"Return: {regime_best_metrics.get('total_return_pct', 0):.2f}%")
        
        # Store all results for this regime
        all_results[regime] = regime_results
    
    return {
        'best_params': best_params,
        'best_metrics': best_metrics,
        'all_results': all_results,
        'regime_stats': regime_stats
    }


def run_regime_adaptive_backtest(symbol, start_date, end_date, regime_params, regime_window=50):
    """
    Run a backtest with regime-adaptive parameters.
    
    Args:
        symbol: Symbol to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        regime_params: Dict mapping regimes to parameter sets
        regime_window: Window for regime detection
        
    Returns:
        dict: Performance metrics
    """
    # Create core components
    portfolio = PortfolioManager(initial_cash=INITIAL_CAPITAL)
    
    # Create direct broker
    broker = DirectBroker(portfolio)
    
    # Create regime detector
    regime_detector = SimpleRegimeDetector(window=regime_window)
    
    # Create regime-adaptive strategy
    strategy = RegimeAdaptiveStrategy(regime_detector, regime_params)
    strategy.set_symbols(symbol)
    
    # Make sure all sub-strategies have the symbol set
    for regime, sub_strategy in strategy.strategies.items():
        sub_strategy.symbols = [symbol] if not isinstance(symbol, list) else symbol
        # Initialize empty price history for each symbol in each strategy
        sub_strategy.prices = {sym: [] for sym in sub_strategy.symbols}
    
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
    logger.info(f"Running regime-adaptive backtest with optimized parameters for each regime")
    
    # Process all bars
    bar_count = 0
    equity_curve = []
    signal_count = 0
    regime_stats = {r: 0 for r in MarketRegime}
    regime_duration = {r: [] for r in MarketRegime}
    current_regime = None
    regime_start_bar = 0
    signal_by_regime = {r: 0 for r in MarketRegime}
    
    while True:
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break
            
        bar_count += 1
        
        # Update broker's market data
        close_price = bar.get_close()
        broker.update_market_price(symbol, close_price)
        
        # Process bar with regime-adaptive strategy
        signal = strategy.on_bar(bar)
        
        # Track regime statistics
        detected_regime = regime_detector.get_current_regime()
        regime_stats[detected_regime] += 1
        
        # Track regime durations
        if detected_regime != current_regime:
            if current_regime is not None:
                duration = bar_count - regime_start_bar
                regime_duration[current_regime].append(duration)
            current_regime = detected_regime
            regime_start_bar = bar_count
        
        # If signal generated, process with risk manager
        if signal:
            signal_count += 1
            signal_by_regime[detected_regime] += 1
            direction_desc = "BUY" if signal.get_signal_value() == 1 else "SELL"
            
            # Get parameters used for this signal
            params_used = regime_params[detected_regime]
            invert = params_used.get('invert_signals', False)
            
            logger.debug(f"Signal #{signal_count} in {detected_regime.value} regime: "
                       f"{signal.get_symbol()} {direction_desc} @ {signal.get_price()} "
                       f"(Fast: {params_used.get('fast_window')}, Slow: {params_used.get('slow_window')}"
                       f"{', Inverted' if invert else ''})")
            
            # Process signal
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
            'equity': current_equity,
            'regime': detected_regime.value
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
        last_regime = equity_curve[-1]['regime']
        equity_curve.append({
            'timestamp': last_timestamp,  # Use the same timestamp as the last bar
            'equity': final_equity,
            'regime': last_regime
        })
    
    # Print regime statistics
    logger.info("\nRegime Statistics:")
    for regime, count in regime_stats.items():
        percentage = (count / bar_count) * 100 if bar_count > 0 else 0
        avg_duration = sum(regime_duration[regime]) / len(regime_duration[regime]) if regime_duration[regime] else 0
        signals = signal_by_regime[regime]
        signal_density = signals / count if count > 0 else 0
        
        logger.info(f"  {regime.value}: {count} bars ({percentage:.1f}%), "
                  f"Avg duration: {avg_duration:.1f} bars, "
                  f"Signals: {signals} ({signal_density*100:.2f}% of bars)")
        
        # Show parameters used for this regime
        params_used = regime_params[regime]
        invert = params_used.get('invert_signals', False)
        logger.info(f"    Parameters: Fast MA {params_used.get('fast_window')}, "
                  f"Slow MA {params_used.get('slow_window')}"
                  f"{', Inverted Signals' if invert else ''}")
    
    # Print broker stats
    broker_stats = broker.get_stats()
    logger.info(f"Broker stats: Orders={broker_stats['orders_processed']}, Fills={broker_stats['fills_created']}")
    
    # Print final portfolio state
    logger.info(f"Processed total of {bar_count} bars")
    logger.info(f"Final cash: ${portfolio.cash:.2f}")
    
    # Calculate performance metrics
    calculator = PerformanceCalculator()
    metrics = calculator.calculate(equity_curve)
    
    logger.info(f"Regime-adaptive backtest complete. Final equity: ${metrics['final_equity']:.2f}")
    logger.info(f"Total return: {metrics['total_return_pct']:.2f}%")
    logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    
    return metrics


def baseline_backtest(symbol, start_date, end_date, params, regime_window=50):
    """
    Run a non-adaptive baseline backtest for comparison.
    
    Args:
        symbol: Symbol to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        params: Strategy parameters
        regime_window: Window for regime detection (only for regime tracking)
        
    Returns:
        dict: Performance metrics
    """
    result = run_backtest_for_regime(
        symbol, start_date, end_date, params, regime=None, 
        regime_window=regime_window, inverse_signals=False
    )
    
    metrics = result['metrics']
    regime_metrics = result['regime_metrics']
    
    logger.info("\nBaseline Backtest Results:")
    logger.info(f"Parameters: Fast MA {params['fast_window']}, Slow MA {params['slow_window']}")
    logger.info(f"Total return: {metrics.get('total_return_pct', 0):.2f}%")
    logger.info(f"Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    
    logger.info("\nPerformance by Regime:")
    for regime, r_metrics in regime_metrics.items():
        count = r_metrics['count']
        if count > 0:
            logger.info(f"  {regime.value} ({count} bars): "
                      f"Ann. Return {r_metrics['ann_return']:.2f}%, "
                      f"Volatility {r_metrics['volatility']:.2f}%, "
                      f"Sharpe {r_metrics['sharpe']:.2f}")
    
    return metrics


def main():
    """Main function to run the optimization and backtest."""
    logger.info("Starting Regime-Specific Parameter Optimization")
    
    # Set up regime window
    regime_window = 50  # Window for regime detection
    
    # Step 1: Optimize parameters for each regime
    logger.info(f"\n{'='*50}")
    logger.info("STEP 1: Optimizing Parameters for Each Regime")
    logger.info(f"{'='*50}")
    
    optimization_results = optimize_regime_parameters(
        SAMPLE_SYMBOL, START_DATE, END_DATE, PARAM_GRID, 
        regime_window=regime_window, inverse_signals=True
    )
    
    # Extract regime-specific optimal parameters
    regime_params = optimization_results['best_params']
    
    # Step 2: Run a baseline backtest for comparison
    logger.info(f"\n{'='*50}")
    logger.info("STEP 2: Running Baseline Backtest for Comparison")
    logger.info(f"{'='*50}")
    
    baseline_params = {'fast_window': 10, 'slow_window': 30}  # Default parameters
    baseline_metrics = baseline_backtest(
        SAMPLE_SYMBOL, START_DATE, END_DATE, baseline_params, regime_window=regime_window
    )
    
    # Step 3: Run adaptive backtest with regime-specific parameters
    logger.info(f"\n{'='*50}")
    logger.info("STEP 3: Running Adaptive Backtest with Regime-Specific Parameters")
    logger.info(f"{'='*50}")
    
    adaptive_metrics = run_regime_adaptive_backtest(
        SAMPLE_SYMBOL, START_DATE, END_DATE, regime_params, regime_window=regime_window
    )
    
    # Step 4: Compare results
    logger.info(f"\n{'='*50}")
    logger.info("STEP 4: Performance Comparison")
    logger.info(f"{'='*50}")
    
    baseline_return = baseline_metrics.get('total_return_pct', 0)
    baseline_sharpe = baseline_metrics.get('sharpe_ratio', 0)
    
    adaptive_return = adaptive_metrics.get('total_return_pct', 0)
    adaptive_sharpe = adaptive_metrics.get('sharpe_ratio', 0)
    
    improvement_return = adaptive_return - baseline_return
    improvement_pct = (adaptive_return / baseline_return - 1) * 100 if baseline_return != 0 else float('inf')
    
    logger.info("\nPerformance Comparison:")
    logger.info(f"Metric          Baseline    Adaptive    Improvement")
    logger.info(f"{'-'*50}")
    logger.info(f"Total Return    {baseline_return:>8.2f}%   {adaptive_return:>8.2f}%   {improvement_return:>+8.2f}% ({improvement_pct:>+.1f}%)")
    logger.info(f"Sharpe Ratio    {baseline_sharpe:>8.2f}    {adaptive_sharpe:>8.2f}    {adaptive_sharpe-baseline_sharpe:>+8.2f}")
    
    # Print optimal parameters for each regime
    logger.info("\nOptimal Parameters by Regime:")
    logger.info(f"{'Regime':<12} {'Fast MA':<8} {'Slow MA':<8} {'Invert':<8}")
    logger.info(f"{'-'*40}")
    
    for regime, params in regime_params.items():
        invert = params.get('invert_signals', False)
        logger.info(f"{regime.value:<12} {params.get('fast_window', 'N/A'):<8} {params.get('slow_window', 'N/A'):<8} {str(invert):<8}")
    
    logger.info("\nOptimization and testing complete!")


if __name__ == "__main__":
    main()
