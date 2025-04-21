#!/usr/bin/env python
"""
Regime-Based Optimization Implementation

This demonstrates how to implement regime detection and regime-specific optimization
for trading strategies using the existing component framework.
"""
import os
import numpy as np
import pandas as pd
import datetime
import logging
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict



# Data components
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler

# Event system components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType, Event, SignalEvent, OrderEvent, FillEvent, BarEvent
from src.core.events.event_emitters import BarEmitter
from src.core.events.event_utils import create_fill_event, create_order_event

# Strategy and execution components
from src.execution.brokers.simulated import SimulatedBroker
from src.execution.execution_base import ExecutionEngine
from src.execution.portfolio import PortfolioManager
from src.strategy.risk.risk_manager import SimpleRiskManager


# For grid search
from src.models.optimization.grid_search import GridSearchOptimizer



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)






#################################################
# Moving Average Crossover Strategy
#################################################

class MovingAverageCrossoverStrategy:
    """
    Moving average crossover strategy that generates buy/sell signals
    when fast MA crosses above/below slow MA.
    """
    
    def __init__(self, name, symbols, fast_window=5, slow_window=15, market_filter=None):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            symbols: List of symbols to trade
            fast_window: Fast moving average window
            slow_window: Slow moving average window
            market_filter: Optional market condition filter
        """
        self.name = name
        self.event_bus = None
        
        # Settings
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.market_filter = market_filter
        
        # Store price history
        self.price_history = {symbol: [] for symbol in self.symbols}
        
        # State
        self.last_ma_values = {symbol: {'fast': None, 'slow': None} for symbol in self.symbols}
        self.signals = []
        
        logger.info(f"Initialized MA Crossover strategy: fast={fast_window}, slow={slow_window}")
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
        return self
        
    def set_parameters(self, params):
        """Set strategy parameters."""
        if 'fast_window' in params:
            self.fast_window = params['fast_window']
        if 'slow_window' in params:
            self.slow_window = params['slow_window']
        
        logger.info(f"Updated MA parameters: fast={self.fast_window}, slow={self.slow_window}")
        
        # Reset state as parameter change invalidates previous calculations
        self.reset()
        
        return self
    
    def get_parameters(self):
        """Get current strategy parameters."""
        return {
            'fast_window': self.fast_window,
            'slow_window': self.slow_window
        }
    
    def validate_parameters(self, params):
        """Validate strategy parameters."""
        if 'fast_window' in params and 'slow_window' in params:
            return params['fast_window'] < params['slow_window']
        return True
    
    def on_bar(self, event):
        """Process a bar event."""
        if not isinstance(event, BarEvent):
            return None
            
        symbol = event.get_symbol()
        if symbol not in self.symbols:
            return None
        
        # Apply market filter if available
        if self.market_filter:
            # Update filter with new data
            active = self.market_filter.update(event)
            
            # Skip if filtered out
            if not active:
                reason = self.market_filter.get_reason(symbol)
                logger.debug(f"Signal generation filtered out for {symbol}: {reason}")
                return None
            
        # Add price to history
        close_price = event.get_close()
        self.price_history[symbol].append(close_price)
        
        logger.debug(f"Adding price {close_price} for {symbol}, history size: {len(self.price_history[symbol])}")
        
        # Keep history manageable
        if len(self.price_history[symbol]) > self.slow_window + 10:
            self.price_history[symbol] = self.price_history[symbol][-(self.slow_window + 10):]
        
        # Check if we have enough data
        if len(self.price_history[symbol]) < self.slow_window:
            return None
            
        # Calculate MAs
        fast_ma = sum(self.price_history[symbol][-self.fast_window:]) / self.fast_window
        slow_ma = sum(self.price_history[symbol][-self.slow_window:]) / self.slow_window
        
        logger.debug(f"MAs for {symbol}: fast={fast_ma:.2f}, slow={slow_ma:.2f}")
        
        # Get previous MA values
        prev_fast = self.last_ma_values[symbol]['fast']
        prev_slow = self.last_ma_values[symbol]['slow']
        
        # Update MA values
        self.last_ma_values[symbol]['fast'] = fast_ma
        self.last_ma_values[symbol]['slow'] = slow_ma
        
        # Skip if no previous values
        if prev_fast is None or prev_slow is None:
            logger.debug(f"No previous MA values for {symbol}, skipping signal generation")
            return None
            
        # Check for crossover
        signal = None
        
        # Buy signal: fast MA crosses above slow MA
        if fast_ma > slow_ma and prev_fast <= prev_slow:
            logger.info(f"BUY SIGNAL: {symbol} fast MA crossed above slow MA")
            signal = SignalEvent(
                signal_value=SignalEvent.BUY,
                price=close_price,
                symbol=symbol,
                rule_id=self.name,
                confidence=1.0,
                metadata={
                    'fast_ma': fast_ma,
                    'slow_ma': slow_ma
                },
                timestamp=event.get_timestamp()
            )
            
        # Sell signal: fast MA crosses below slow MA
        elif fast_ma < slow_ma and prev_fast >= prev_slow:
            logger.info(f"SELL SIGNAL: {symbol} fast MA crossed below slow MA")
            signal = SignalEvent(
                signal_value=SignalEvent.SELL,
                price=close_price,
                symbol=symbol,
                rule_id=self.name,
                confidence=1.0,
                metadata={
                    'fast_ma': fast_ma,
                    'slow_ma': slow_ma
                },
                timestamp=event.get_timestamp()
            )
        
        # Emit signal if generated
        if signal:
            self.signals.append(signal)
            
            if self.event_bus:
                logger.debug(f"Emitting signal: {symbol} {'BUY' if signal.get_signal_value() == SignalEvent.BUY else 'SELL'}")
                self.event_bus.emit(signal)
        
        return signal
    
    def reset(self):
        """Reset the strategy state."""
        self.price_history = {symbol: [] for symbol in self.symbols}
        self.last_ma_values = {symbol: {'fast': None, 'slow': None} for symbol in self.symbols}
        self.signals = []


#################################################
# Walk-Forward Optimizer
#################################################

class WalkForwardOptimizer:
    """
    Walk-forward optimization for strategy parameters.
    Uses time-based train/test splits to prevent look-ahead bias.
    """
    
    def __init__(self, train_size=0.6, test_size=0.4, windows=3):
        """
        Initialize the walk-forward optimizer.
        
        Args:
            train_size: Proportion of data for training (in each window)
            test_size: Proportion of data for testing (in each window)
            windows: Number of train/test windows to use
        """
        self.train_size = train_size
        self.test_size = test_size
        self.windows = windows
        self.results = []
        
    def optimize(self, param_grid, data_handler, evaluation_func, start_date=None, end_date=None):
        """
        Perform walk-forward optimization.
        
        Args:
            param_grid: Dictionary mapping parameter names to possible values
            data_handler: Data handler with loaded data
            evaluation_func: Function to evaluate parameter combinations
            start_date: Start date for optimization
            end_date: End date for optimization
            
        Returns:
            dict: Optimization results with best parameters and performance
        """
        # Get data date range (handles None values)
        symbol = data_handler.get_symbols()[0]  # Use first symbol
        all_bars = list(data_handler.bars_history.get(symbol, []))
        if not all_bars:
            logger.warning("No data available for optimization")
            return None
            
        # Sort bars by timestamp
        all_bars.sort(key=lambda bar: bar.get_timestamp())
        
        # Get date range
        if start_date is None:
            start_date = all_bars[0].get_timestamp()
        if end_date is None:
            end_date = all_bars[-1].get_timestamp()
            
        logger.info(f"Walk-forward optimization from {start_date} to {end_date}")
        
        # Create time-based windows
        window_results = []
        date_range = (end_date - start_date).total_seconds()
        window_size = date_range / self.windows
        
        for i in range(self.windows):
            # Calculate window dates
            window_start = start_date + datetime.timedelta(seconds=i * window_size)
            window_end = start_date + datetime.timedelta(seconds=(i + 1) * window_size)
            
            # Calculate train/test split within window
            split_point = window_start + datetime.timedelta(seconds=window_size * self.train_size)
            
            train_start = window_start
            train_end = split_point
            test_start = split_point
            test_end = window_end
            
            logger.info(f"\nWindow {i+1}/{self.windows}:")
            logger.info(f"  Train: {train_start} to {train_end}")
            logger.info(f"  Test:  {test_start} to {test_end}")
            
            # Grid search on training data
            best_params, best_score = self._grid_search(
                param_grid, 
                data_handler,
                evaluation_func,
                train_start, 
                train_end
            )
            
            # Test best parameters on test data
            test_score = evaluation_func(
                best_params, 
                data_handler, 
                test_start, 
                test_end
            )
            
            logger.info(f"  Best parameters: {best_params}")
            logger.info(f"  Train score: {best_score:.4f}, Test score: {test_score:.4f}")
            
            # Store window results
            window_results.append({
                'window': i + 1,
                'params': best_params,
                'train_score': best_score,
                'test_score': test_score,
                'train_period': (train_start, train_end),
                'test_period': (test_start, test_end)
            })
        
        # Aggregate results across windows
        param_scores = defaultdict(list)
        
        # Track test performance for each parameter set
        for result in window_results:
            params_key = self._params_to_key(result['params'])
            param_scores[params_key].append(result['test_score'])
        
        # Calculate average performance
        avg_scores = {}
        for params_key, scores in param_scores.items():
            avg_scores[params_key] = {
                'params': self._key_to_params(params_key),
                'avg_score': np.mean(scores),
                'min_score': min(scores),
                'max_score': max(scores),
                'std_score': np.std(scores),
                'scores': scores
            }
        
        # Find best overall parameters (based on average test score)
        best_key = max(avg_scores.keys(), key=lambda k: avg_scores[k]['avg_score'])
        best_overall = avg_scores[best_key]
        
        # Format results
        aggregated_results = {
            'best_params': best_overall['params'],
            'best_avg_score': best_overall['avg_score'],
            'best_min_score': best_overall['min_score'],
            'best_max_score': best_overall['max_score'],
            'best_std_score': best_overall['std_score'],
            'window_results': window_results,
            'all_params': [avg_scores[k] for k in avg_scores]
        }
        
        # Sort all parameter results by average score (descending)
        aggregated_results['all_params'].sort(key=lambda x: x['avg_score'], reverse=True)
        
        return aggregated_results
        
    def _grid_search(self, param_grid, data_handler, evaluation_func, start_date, end_date):
        """
        Perform grid search on a specific time period.
        
        Args:
            param_grid: Dictionary of parameters to test
            data_handler: Data handler with loaded data
            evaluation_func: Function to evaluate parameters
            start_date: Start date for training
            end_date: End date for training
            
        Returns:
            tuple: (best_params, best_score)
        """
        # Generate all parameter combinations
        param_names = sorted(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        combinations = list(itertools.product(*param_values))
        
        logger.info(f"Grid search: evaluating {len(combinations)} parameter combinations")
        
        # Track best parameters
        best_params = None
        best_score = float('-inf')
        
        # Evaluate each combination
        for i, values in enumerate(combinations):
            # Create parameter dict
            params = {name: value for name, value in zip(param_names, values)}
            
            # Skip invalid parameter combinations (e.g., fast_window >= slow_window)
            if 'fast_window' in params and 'slow_window' in params:
                if params['fast_window'] >= params['slow_window']:
                    continue
            
            # Evaluate on training data
            score = evaluation_func(params, data_handler, start_date, end_date)
            
            # Update best if improved
            if score > best_score:
                best_score = score
                best_params = params
                
            # Progress update
            if (i + 1) % 5 == 0 or (i + 1) == len(combinations):
                logger.info(f"Evaluated {i + 1}/{len(combinations)} combinations")
        
        return best_params, best_score
        
    def _params_to_key(self, params):
        """Convert parameters dict to string key for tracking."""
        return str(sorted(params.items()))
        
    def _key_to_params(self, key):
        """Convert string key back to parameters dict."""
        # Parse string representation of sorted items back to dict
        # This is a simplification - in practice, proper parsing would be needed
        items = eval(key)
        return dict(items)


#################################################
# 1. Regime Detection
#################################################

def run_walk_forward_optimization(data_dir, symbols, start_date, end_date, timeframe):
    """
    Run walk-forward optimization on the moving average strategy.
    
    Args:
        data_dir: Directory containing data files
        symbols: Symbol or list of symbols to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        timeframe: Data timeframe
        
    Returns:
        dict: Optimization results
    """
    print("\n=== Running Walk-Forward Optimization ===")
    
    # Convert symbols to list if needed
    if isinstance(symbols, str):
        symbols = [symbols]
    
    # Define parameter grid
    param_grid = {
        'fast_window': [5, 10, 15, 20],
        'slow_window': [20, 30, 40, 50]
    }
    
    # Create data handler
    data_source = CSVDataSource(
        data_dir=data_dir,
        filename_pattern='{symbol}_{timeframe}.csv',
        date_column='timestamp',
        date_format=None,
        column_map={
            'open': ['Open'],
            'high': ['High'],
            'low': ['Low'],
            'close': ['Close'],
            'volume': ['Volume']
        }
    )
    
    # Create data handler (without event bus, just for data access)
    data_handler = HistoricalDataHandler(data_source, None)
    
    # Load data
    for symbol in symbols:
        data_handler.load_data(symbol, start_date=start_date, end_date=end_date, timeframe=timeframe)
    
    # Create walk-forward optimizer
    optimizer = WalkForwardOptimizer(train_size=0.6, test_size=0.4, windows=3)
    
    # Define evaluation function
    def evaluate_params(params, data_handler, period_start, period_end):
        """
        Run backtest with given parameters and return score.
        
        Args:
            params: Strategy parameters
            data_handler: Data handler with loaded data
            period_start: Start date for evaluation
            period_end: End date for evaluation
            
        Returns:
            float: Performance score (Sharpe ratio)
        """
        fast_window = params['fast_window']
        slow_window = params['slow_window']
        
        print(f"\nTesting parameters: Fast MA: {fast_window}, Slow MA: {slow_window}")
        print(f"Period: {period_start} to {period_end}")
        
        # Run backtest with these parameters
        results, _, _, equity_curve = run_backtest(
            data_dir=data_dir,
            symbols=symbols,
            start_date=period_start,
            end_date=period_end,
            timeframe=timeframe,
            fast_window=fast_window,
            slow_window=slow_window,
            fixed_position_size=100
        )
        
        # Calculate score (could be Sharpe ratio, returns, etc.)
        if not results:
            return float('-inf')
            
        # Use Sharpe ratio as score (could use other metrics)
        score = results.get('sharpe_ratio', 0)
        
        print(f"Parameters: Fast MA: {fast_window}, Slow MA: {slow_window}, Score: {score:.4f}")
        
        return score
    
    # Run walk-forward optimization
    results = optimizer.optimize(param_grid, data_handler, evaluate_params, start_date, end_date)
    
    if not results:
        print("Optimization failed - no data available")
        return None
    
    # Print optimization results
    print("\n=== Walk-Forward Optimization Results ===")
    print(f"Best Parameters: {results['best_params']}")
    print(f"Average Score: {results['best_avg_score']:.4f}")
    print(f"Score Range: {results['best_min_score']:.4f} to {results['best_max_score']:.4f}")
    print(f"Score Standard Deviation: {results['best_std_score']:.4f}")
    
    # Print window results
    print("\nResults by Window:")
    for window in results['window_results']:
        print(f"Window {window['window']}:")
        print(f"  Train: {window['train_period'][0]} to {window['train_period'][1]}")
        print(f"  Test:  {window['test_period'][0]} to {window['test_period'][1]}")
        print(f"  Parameters: {window['params']}")
        print(f"  Train Score: {window['train_score']:.4f}, Test Score: {window['test_score']:.4f}")
    
    # Print top 3 parameter sets
    print("\nTop 3 Parameter Sets (by average test score):")
    for i, result in enumerate(results['all_params'][:3]):
        params = result['params']
        score = result['avg_score']
        print(f"{i+1}. Fast MA: {params['fast_window']}, Slow MA: {params['slow_window']}, Avg Score: {score:.4f}")
    
    return results

class MarketRegime(Enum):
    """Enumeration of market regime types."""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class RegimeDetector:
    """
    Detects market regimes based on various technical indicators.
    """
    
    def __init__(self, lookback_window=20, trend_threshold=0.05, 
                volatility_threshold=0.015, sideways_threshold=0.02):
        """
        Initialize the regime detector.
        
        Args:
            lookback_window: Period for regime analysis
            trend_threshold: Minimum price change for trend detection
            volatility_threshold: Threshold for high volatility regime
            sideways_threshold: Max range for sideways market
        """
        self.lookback_window = lookback_window
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold
        self.sideways_threshold = sideways_threshold
        self.price_history = {}  # symbol -> list of prices
        self.regime_history = {}  # symbol -> list of (timestamp, regime) tuples
    
    def update(self, bar):
        """
        Update detector with new price data and detect current regime.
        
        Args:
            bar: Bar event with price data
            
        Returns:
            MarketRegime: Detected market regime
        """
        symbol = bar.get_symbol()
        close_price = bar.get_close()
        timestamp = bar.get_timestamp()
        
        # Initialize history if needed
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.regime_history[symbol] = []
            
        # Add price to history
        self.price_history[symbol].append((timestamp, close_price))
        
        # Keep history limited to reasonable size
        if len(self.price_history[symbol]) > self.lookback_window * 2:
            self.price_history[symbol] = self.price_history[symbol][-self.lookback_window * 2:]
            
        # Need enough history for regime detection
        if len(self.price_history[symbol]) < self.lookback_window:
            regime = MarketRegime.UNKNOWN
            self.regime_history[symbol].append((timestamp, regime))
            return regime
            
        # Get relevant price data
        recent_prices = [price for _, price in self.price_history[symbol][-self.lookback_window:]]
        
        # Calculate key metrics
        start_price = recent_prices[0]
        end_price = recent_prices[-1]
        price_change = (end_price / start_price) - 1
        
        # Calculate volatility
        returns = [recent_prices[i]/recent_prices[i-1] - 1 for i in range(1, len(recent_prices))]
        volatility = np.std(returns)
        
        # Calculate price range
        price_range = (max(recent_prices) - min(recent_prices)) / np.mean(recent_prices)
        
        # Detect regime
        if volatility > self.volatility_threshold:
            regime = MarketRegime.VOLATILE
        elif abs(price_change) < self.sideways_threshold and price_range < self.sideways_threshold * 2:
            regime = MarketRegime.SIDEWAYS
        elif price_change > self.trend_threshold:
            regime = MarketRegime.UPTREND
        elif price_change < -self.trend_threshold:
            regime = MarketRegime.DOWNTREND
        else:
            regime = MarketRegime.SIDEWAYS
        
        # Store regime
        self.regime_history[symbol].append((timestamp, regime))
        
        logger.debug(f"Detected regime for {symbol} at {timestamp}: {regime.value}")
        return regime
    
    def get_current_regime(self, symbol):
        """Get current regime for a symbol."""
        if not symbol in self.regime_history or not self.regime_history[symbol]:
            return MarketRegime.UNKNOWN
        return self.regime_history[symbol][-1][1]
    
    def get_regime_at(self, symbol, timestamp):
        """Get regime for a symbol at specific timestamp."""
        if not symbol in self.regime_history:
            return MarketRegime.UNKNOWN
            
        # Find the regime at or before the timestamp
        for ts, regime in reversed(self.regime_history[symbol]):
            if ts <= timestamp:
                return regime
                
        return MarketRegime.UNKNOWN
    
    def get_regime_periods(self, symbol, start_date=None, end_date=None):
        """
        Get periods of different regimes for a symbol.
        
        Args:
            symbol: Symbol to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            dict: Dictionary mapping regime types to lists of (start, end) periods
        """
        if not symbol in self.regime_history:
            return {}
            
        history = self.regime_history[symbol]
        
        # Filter by date range if specified
        if start_date or end_date:
            if start_date:
                history = [(ts, regime) for ts, regime in history if ts >= start_date]
            if end_date:
                history = [(ts, regime) for ts, regime in history if ts <= end_date]
                
        if not history:
            return {}
            
        # Find continuous periods of same regime
        regime_periods = defaultdict(list)
        
        current_regime = history[0][1]
        period_start = history[0][0]
        
        for i in range(1, len(history)):
            timestamp, regime = history[i]
            
            # Regime change
            if regime != current_regime:
                # Store previous period
                regime_periods[current_regime].append((period_start, timestamp))
                
                # Start new period
                current_regime = regime
                period_start = timestamp
        
        # Add final period
        if history:
            regime_periods[current_regime].append((period_start, history[-1][0]))
            
        return regime_periods
    
    def reset(self):
        """Reset the detector state."""
        self.price_history = {}
        self.regime_history = {}


#################################################
# 2. Regime-Based Strategy
#################################################

class RegimeAwareStrategy:
    """Extends strategy with regime-based parameter switching."""
    
    def __init__(self, base_strategy, regime_detector):
        """
        Initialize the regime-aware strategy wrapper.
        
        Args:
            base_strategy: Base strategy to enhance with regime awareness
            regime_detector: Regime detector instance
        """
        self.strategy = base_strategy
        self.regime_detector = regime_detector
        self.event_bus = None
        self.name = f"regime_aware_{base_strategy.name}"
        self.symbols = base_strategy.symbols
        
        # Parameter sets for different regimes
        self.regime_parameters = {
            MarketRegime.UPTREND: {},
            MarketRegime.DOWNTREND: {},
            MarketRegime.SIDEWAYS: {},
            MarketRegime.VOLATILE: {},
            MarketRegime.UNKNOWN: {}  # Default parameters
        }
        
        # Current active parameters
        self.active_regime = {symbol: MarketRegime.UNKNOWN for symbol in self.symbols}
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
        self.strategy.set_event_bus(event_bus)
        return self
    
    def set_regime_parameters(self, regime, parameters):
        """
        Set parameters for a specific regime.
        
        Args:
            regime: MarketRegime to set parameters for
            parameters: Parameter dict for this regime
        """
        self.regime_parameters[regime] = parameters
        
        # If this is for the current regime of any symbol, apply immediately
        for symbol in self.symbols:
            if self.active_regime[symbol] == regime:
                logger.info(f"Applying new {regime.value} parameters to {symbol}: {parameters}")
                self.strategy.set_parameters(parameters)
    
    def on_bar(self, event):
        """
        Process a bar event with regime-specific parameters.
        
        Args:
            event: Bar event to process
            
        Returns:
            Signal event or None
        """
        symbol = event.get_symbol()
        
        if symbol not in self.symbols:
            return None
            
        # Detect current regime
        current_regime = self.regime_detector.update(event)
        
        # Check if regime changed
        if current_regime != self.active_regime[symbol]:
            # Switch parameters
            self._switch_parameters(symbol, current_regime)
            
        # Process with current parameters
        return self.strategy.on_bar(event)
    
    def _switch_parameters(self, symbol, regime):
        """
        Switch strategy parameters based on regime.
        
        Args:
            symbol: Symbol that experienced regime change
            regime: New regime for the symbol
        """
        logger.info(f"Regime change for {symbol}: {self.active_regime[symbol]} -> {regime}")
        
        # Update active regime
        self.active_regime[symbol] = regime
        
        # Apply regime-specific parameters
        parameters = self.regime_parameters.get(regime)
        
        if parameters:
            logger.info(f"Switching to {regime.value} parameters for {symbol}: {parameters}")
            self.strategy.set_parameters(parameters)
        else:
            logger.warning(f"No parameters defined for {regime.value} regime")
    
    def reset(self):
        """Reset the strategy and detector."""
        self.strategy.reset()
        self.regime_detector.reset()
        self.active_regime = {symbol: MarketRegime.UNKNOWN for symbol in self.symbols}


#################################################
# 3. Regime-Based Optimizer
#################################################

class RegimeOptimizer:
    """
    Optimizes strategy parameters separately for each market regime.
    """
    
    def __init__(self, regime_detector, base_optimizer):
        """
        Initialize the regime optimizer.
        
        Args:
            regime_detector: Regime detector instance
            base_optimizer: Base optimizer to use for each regime
        """
        self.regime_detector = regime_detector
        self.base_optimizer = base_optimizer
        self.results = {}
    
    def optimize(self, param_grid, data_handler, evaluation_func, 
                start_date=None, end_date=None, min_regime_bars=50):
        """
        Perform regime-specific optimization.
        
        Args:
            param_grid: Dictionary of parameter names to possible values
            data_handler: Data handler with loaded data
            evaluation_func: Function to evaluate parameter combinations
            start_date: Start date for optimization
            end_date: End date for optimization
            min_regime_bars: Minimum bars required for a regime to be optimized
            
        Returns:
            dict: Best parameters for each regime
        """
        symbol = data_handler.get_symbols()[0]  # Use first symbol
        
        # 1. Perform regime detection on historical data
        logger.info(f"Performing regime detection on historical data for {symbol}")
        self._detect_regimes(data_handler, start_date, end_date)
        
        # 2. Segment data by regime
        regime_periods = self.regime_detector.get_regime_periods(symbol, start_date, end_date)
        
        # 3. For each regime, optimize parameters
        regime_params = {}
        
        for regime, periods in regime_periods.items():
            if not periods:
                continue
                
            # Count total bars in this regime
            bar_count = self._count_bars_in_periods(data_handler, symbol, periods)
            
            if bar_count < min_regime_bars:
                logger.info(f"Skipping optimization for {regime.value} - insufficient data ({bar_count} bars)")
                continue
                
            logger.info(f"\n--- Optimizing for {regime.value} regime ({bar_count} bars) ---")
            
            # Optimize for this regime
            best_params, best_score = self._optimize_for_regime(
                param_grid, data_handler, evaluation_func, regime, periods
            )
            
            if best_params:
                logger.info(f"Best parameters for {regime.value}: {best_params}, Score: {best_score:.4f}")
                regime_params[regime] = best_params
            else:
                logger.warning(f"Optimization failed for {regime.value}")
        
        # 4. Set default parameters if any regime is missing
        all_regimes = list(MarketRegime)
        for regime in all_regimes:
            if regime not in regime_params:
                # Use parameters from most similar regime or average of others
                regime_params[regime] = self._get_default_params(regime, regime_params)
                
        # Store and return results
        self.results = {
            'regime_parameters': regime_params,
            'regime_periods': {r.value: periods for r, periods in regime_periods.items()}
        }
        
        return regime_params
    
    def _detect_regimes(self, data_handler, start_date, end_date):
        """Run regime detection on historical data."""
        symbol = data_handler.get_symbols()[0]
        
        # Reset data handler to start of data
        data_handler.reset()
        
        # Create temporary detector for historical analysis
        detector = self.regime_detector
        detector.reset()
        
        # Process each bar to detect regimes
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
                
            # Skip bars outside date range
            if start_date and bar.get_timestamp() < start_date:
                continue
            if end_date and bar.get_timestamp() > end_date:
                break
                
            # Detect regime
            detector.update(bar)
        
        # Reset data handler for optimization
        data_handler.reset()
    
    def _count_bars_in_periods(self, data_handler, symbol, periods):
        """Count total bars across multiple time periods."""
        # Reset data handler
        data_handler.reset()
        
        total_bars = 0
        
        # Process each bar
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
                
            # Check if bar falls in any of the periods
            timestamp = bar.get_timestamp()
            for start, end in periods:
                if start <= timestamp <= end:
                    total_bars += 1
                    break
        
        # Reset data handler
        data_handler.reset()
        
        return total_bars
    
    def _optimize_for_regime(self, param_grid, data_handler, evaluation_func, regime, periods):
        """Optimize parameters for a specific regime."""
        # Define regime-specific evaluation function
        def regime_evaluation(params):
            # Evaluate only on bars in this regime
            return self._evaluate_in_periods(
                params, data_handler, evaluation_func, periods
            )
        
        # Run base optimizer with regime-specific evaluation
        try:
            result = self.base_optimizer.optimize(param_grid, regime_evaluation)
            return result['best_params'], result['best_score']
        except Exception as e:
            logger.error(f"Error optimizing for {regime.value}: {e}")
            return None, float('-inf')
    
    def _evaluate_in_periods(self, params, data_handler, evaluation_func, periods):
        """Evaluate parameters only on bars within specific time periods."""
        # Reset data handler
        data_handler.reset()
        
        symbol = data_handler.get_symbols()[0]
        bars_in_periods = []
        
        # Collect bars that fall within the specified periods
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
                
            # Check if bar falls in any of the periods
            timestamp = bar.get_timestamp()
            for start, end in periods:
                if start <= timestamp <= end:
                    bars_in_periods.append(bar)
                    break
        
        # Reset data handler
        data_handler.reset()
        
        # If no bars in period, return low score
        if not bars_in_periods:
            return float('-inf')
            
        # Evaluate parameters on collected bars
        return evaluation_func(params, bars_in_periods)
    
    def _get_default_params(self, regime, existing_params):
        """Create default parameters for a regime with no optimization data."""
        # If we have parameters for other regimes, use similar or average
        if not existing_params:
            return {}
            
        # Map of regime similarities
        similarities = {
            MarketRegime.UPTREND: [MarketRegime.VOLATILE, MarketRegime.SIDEWAYS, MarketRegime.DOWNTREND],
            MarketRegime.DOWNTREND: [MarketRegime.VOLATILE, MarketRegime.SIDEWAYS, MarketRegime.UPTREND],
            MarketRegime.SIDEWAYS: [MarketRegime.UPTREND, MarketRegime.DOWNTREND, MarketRegime.VOLATILE],
            MarketRegime.VOLATILE: [MarketRegime.UPTREND, MarketRegime.DOWNTREND, MarketRegime.SIDEWAYS],
            MarketRegime.UNKNOWN: [MarketRegime.SIDEWAYS, MarketRegime.UPTREND, MarketRegime.DOWNTREND, MarketRegime.VOLATILE]
        }
        
        # Try to find parameters from most similar regime
        for similar_regime in similarities.get(regime, []):
            if similar_regime in existing_params:
                logger.info(f"Using {similar_regime.value} parameters for {regime.value}")
                return existing_params[similar_regime]
        
        # If no similar regime has parameters, use average of all available parameters
        logger.info(f"Using average parameters for {regime.value}")
        avg_params = {}
        
        # Get all parameter keys
        all_keys = set()
        for params in existing_params.values():
            all_keys.update(params.keys())
            
        # Calculate average for each parameter
        for key in all_keys:
            values = [params[key] for params in existing_params.values() if key in params]
            if values:
                avg_params[key] = sum(values) / len(values)
                
        return avg_params


#################################################
# 4. Example Usage
#################################################

def run_regime_optimization(data_dir, symbols, start_date, end_date, timeframe):
    """
    Run regime-specific optimization on the moving average strategy.
    
    Args:
        data_dir: Directory containing data files
        symbols: Symbol or list of symbols to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        timeframe: Data timeframe
        
    Returns:
        dict: Optimization results with regime-specific parameters
    """
    print("\n=== Running Regime-Based Optimization ===")
    
    # Convert symbols to list if needed
    if isinstance(symbols, str):
        symbols = [symbols]
    
    # Define parameter grid
    param_grid = {
        'fast_window': [5, 10, 15, 20],
        'slow_window': [20, 30, 40, 50]
    }
    
    # Create data handler
    data_source = CSVDataSource(
        data_dir=data_dir,
        filename_pattern='{symbol}_{timeframe}.csv',
        date_column='timestamp',
        date_format=None,
        column_map={
            'open': ['Open'],
            'high': ['High'],
            'low': ['Low'],
            'close': ['Close'],
            'volume': ['Volume']
        }
    )
    
    # Create data handler (without event bus, just for data access)
    data_handler = HistoricalDataHandler(data_source, None)
    
    # Load data
    for symbol in symbols:
        data_handler.load_data(symbol, start_date=start_date, end_date=end_date, timeframe=timeframe)
    
    # Create regime detector
    regime_detector = RegimeDetector(
        lookback_window=20,
        trend_threshold=0.05,
        volatility_threshold=0.015,
        sideways_threshold=0.02
    )
    
    # Create grid search optimizer
    grid_optimizer = GridSearchOptimizer()
    
    # Create regime optimizer
    regime_optimizer = RegimeOptimizer(regime_detector, grid_optimizer)
    
    # Define evaluation function for a set of bars
    def evaluate_strategy_on_bars(params, bars):
        """
        Evaluate strategy performance on a specific set of bars.
        
        Args:
            params: Strategy parameters
            bars: List of bar events
            
        Returns:
            float: Performance score
        """
        # Create event system
        event_bus = EventBus(use_weak_refs=False)
        
        # Create strategy with given parameters
        strategy = MovingAverageCrossoverStrategy(
            name="ma_crossover",
            symbols=symbols,
            fast_window=params['fast_window'],
            slow_window=params['slow_window']
        )
        strategy.set_event_bus(event_bus)
        
        # Create portfolio
        portfolio = PortfolioManager(initial_cash=100000.0, event_bus=event_bus)
        
        # Create risk manager
        risk_manager = SimpleRiskManager(
            portfolio=portfolio,
            event_bus=event_bus,
            fixed_size=100
        )
        
        # Create execution components
        broker = SimulatedBroker(fill_emitter=event_bus)
        execution_engine = ExecutionEngine(broker_interface=broker, event_bus=event_bus)
        
        # Register components
        event_manager = EventManager(event_bus)
        event_manager.register_component('strategy', strategy, [EventType.BAR])
        event_manager.register_component('risk', risk_manager, [EventType.SIGNAL])
        event_manager.register_component('execution', execution_engine, [EventType.ORDER])
        event_manager.register_component('portfolio', portfolio, [EventType.FILL])
        
        # Process bars
        symbol = bars[0].get_symbol() if bars else symbols[0]
        equity_curve = []
        
        for bar in bars:
            # Update broker's market data
            broker.update_market_data(symbol, {"price": bar.get_close()})
            
            # Emit bar event
            event_bus.emit(bar)
            
            # Record equity
            equity_curve.append({
                'timestamp': bar.get_timestamp(),
                'equity': portfolio.get_equity({symbol: bar.get_close()})
            })
        
        # Calculate performance
        if len(equity_curve) < 2:
            return float('-inf')
            
        # Create DataFrame from equity curve
        df = pd.DataFrame(equity_curve)
        df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        df['returns'] = df['equity'].pct_change()
        
        # Calculate Sharpe ratio (or other metrics)
        risk_free_rate = 0.01 / 252
        excess_returns = df['returns'] - risk_free_rate
        sharpe_ratio = excess_returns.mean() / df['returns'].std() * np.sqrt(252) if df['returns'].std() > 0 else 0
        
        return sharpe_ratio
    
    # Run regime optimization
    regime_params = regime_optimizer.optimize(
        param_grid=param_grid,
        data_handler=data_handler,
        evaluation_func=evaluate_strategy_on_bars,
        start_date=start_date,
        end_date=end_date
    )
    
    # Print optimization results
    print("\n=== Regime-Based Optimization Results ===")
    for regime, params in regime_params.items():
        print(f"{regime.value.capitalize()} Regime: {params}")
    
    # Print regime distribution
    regime_periods = regime_optimizer.results['regime_periods']
    print("\nRegime Distribution:")
    for regime, periods in regime_periods.items():
        total_days = sum((end - start).days for start, end in periods)
        print(f"  {regime.capitalize()}: {len(periods)} periods, {total_days} days")
    
    return regime_optimizer.results


def run_regime_backtest(data_dir, symbols, start_date, end_date, timeframe, regime_params):
    """
    Run a backtest with regime-specific parameter switching.
    
    Args:
        data_dir: Directory containing data files
        symbols: Symbol or list of symbols to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        timeframe: Data timeframe
        regime_params: Dictionary of regime-specific parameters
        
    Returns:
        tuple: (results, event_tracker, portfolio)
    """
    print("\n=== Running Regime-Based Backtest ===")
    
    # Convert symbols to list if needed
    if isinstance(symbols, str):
        symbols = [symbols]
    
    # Expand tilde in path if present
    if data_dir.startswith('~'):
        data_dir = os.path.expanduser(data_dir)
    
    # --- Setup Event System ---
    
    # Create event system
    event_bus = EventBus(use_weak_refs=False)
    event_manager = EventManager(event_bus)
    
    # Create event tracker
    tracker = EventTracker()
    
    # Register event tracking
    for event_type in EventType:
        event_bus.register(event_type, tracker.track_event)
    
    # --- Setup Data Components ---
    
    # Create data source
    data_source = CSVDataSource(
        data_dir=data_dir,
        filename_pattern='{symbol}_{timeframe}.csv',
        date_column='timestamp',
        date_format=None,
        column_map={
            'open': ['Open'],
            'high': ['High'],
            'low': ['Low'],
            'close': ['Close'],
            'volume': ['Volume']
        }
    )
    
    # Create bar emitter
    bar_emitter = BarEmitter("backtest_bar_emitter", event_bus)
    bar_emitter.start()
    
    # Create data handler
    data_handler = HistoricalDataHandler(data_source, bar_emitter)
    
    # --- Setup Portfolio ---
    
    # Create portfolio with initial capital
    initial_capital = 100000.0
    portfolio = PortfolioManager(initial_cash=initial_capital, event_bus=event_bus)
    
    # --- Setup Risk Manager ---
    
    # Create simplified risk manager
    risk_manager = SimpleRiskManager(
        portfolio=portfolio,
        event_bus=event_bus,
        fixed_size=100
    )
    
    # --- Setup Execution Components ---
    
    # Create broker
    broker = SimulatedBroker(fill_emitter=event_bus)
    
    # Create execution engine
    execution_engine = ExecutionEngine(broker_interface=broker, event_bus=event_bus)
    
    # --- Setup Strategy ---
    
    # Create base strategy (with default parameters)
    base_strategy = MovingAverageCrossoverStrategy(
        name="ma_crossover",
        symbols=symbols,
        fast_window=10,  # Default parameters
        slow_window=30   # Will be overridden by regime-specific params
    )
    
    # Create regime detector
    regime_detector = RegimeDetector(
        lookback_window=20,
        trend_threshold=0.05,
        volatility_threshold=0.015,
        sideways_threshold=0.02
    )
    
    # Create regime-aware strategy wrapper
    strategy = RegimeAwareStrategy(base_strategy, regime_detector)
    
    # Set regime-specific parameters
    for regime, params in regime_params.items():
        strategy.set_regime_parameters(regime, params)
    
    strategy.set_event_bus(event_bus)
    
    # Initial market price setup
    for symbol in symbols:
        broker.update_market_data(symbol, {"price": 100.0})
    
    # --- Register Components with Event Manager ---
    
    # Register all components
    event_manager.register_component('strategy', strategy, [EventType.BAR])
    event_manager.register_component('risk', risk_manager, [EventType.SIGNAL])
    event_manager.register_component('execution', execution_engine, [EventType.ORDER])
    event_manager.register_component('portfolio', portfolio, [EventType.FILL])
    
    # --- Run Backtest ---
    
    # Load data
    for symbol in symbols:
        data_handler.load_data(symbol, start_date=start_date, end_date=end_date, timeframe=timeframe)
    
    # Process bars for each symbol
    equity_curve = []
    regime_transitions = []
    
    # Go through data chronologically
    for symbol in symbols:
        bar_count = 0
        current_regime = None
        
        # Process bars
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
            
            # Update broker's market data with current price
            broker.update_market_data(symbol, {"price": bar.get_close()})
            
            # Record equity
            equity_curve.append({
                'timestamp': bar.get_timestamp(),
                'equity': portfolio.get_equity({symbol: bar.get_close()})
            })
            
            # Track regime transitions
            new_regime = regime_detector.get_current_regime(symbol)
            if new_regime != current_regime:
                regime_transitions.append({
                    'timestamp': bar.get_timestamp(),
                    'symbol': symbol,
                    'from_regime': current_regime.value if current_regime else 'None',
                    'to_regime': new_regime.value
                })
                current_regime = new_regime
            
            bar_count += 1
            
            if bar_count % 100 == 0:
                logger.info(f"Processed {bar_count} bars for {symbol}, Portfolio equity: ${portfolio.get_equity():,.2f}")
        
        logger.info(f"Completed processing {bar_count} bars for {symbol}")
    
    # --- Calculate Results ---
    
    # Calculate performance metrics
    stats = calculate_portfolio_stats(portfolio, equity_curve)
    
    # Compile results
    results = {
        'initial_equity': stats.get('initial_equity', initial_capital),
        'final_equity': stats.get('final_equity', portfolio.get_equity()),
        'return': stats.get('total_return', 0),
        'return_pct': stats.get('total_return', 0) * 100,
        'annual_return': stats.get('annual_return', 0) * 100,
        'max_drawdown': stats.get('max_drawdown', 0) * 100,
        'sharpe_ratio': stats.get('sharpe_ratio', 0),
        'trade_count': stats.get('trades', 0),
        'signal_count': len(tracker.events[EventType.SIGNAL]),
        'order_count': len(tracker.events[EventType.ORDER]),
        'regime_transitions': regime_transitions
    }
    
    # --- Print Summary ---
    
    print("\n=== Regime-Based Backtest Summary ===")
    print(f"Initial Equity: ${results['initial_equity']:,.2f}")
    print(f"Final Equity: ${results['final_equity']:,.2f}")
    print(f"Return: {results['return_pct']:.2f}%")
    print(f"Annual Return: {results['annual_return']:.2f}%")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Signals Generated: {results['signal_count']}")
    print(f"Orders Placed: {results['order_count']}")
    print(f"Trades Executed: {results['trade_count']}")
    
    # Print regime statistics
    regime_counts = {}
    for transition in regime_transitions:
        regime = transition['to_regime']
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
    
    print("\n=== Regime Statistics ===")
    for regime, count in regime_counts.items():
        print(f"{regime.capitalize()}: {count} occurrences")
    
    print(f"Total Regime Transitions: {len(regime_transitions)}")
    
    return results, tracker, portfolio


def compare_regime_vs_standard(data_dir, symbols, start_date, end_date, timeframe):
    """
    Compare regime-based optimization with standard optimization.
    
    Args:
        data_dir: Directory containing data files
        symbols: Symbol or list of symbols to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        timeframe: Data timeframe
    """
    print("\n=== Comparing Regime-Based vs. Standard Optimization ===")
    
    # 1. Run standard optimization
    print("\n--- Running Standard Optimization ---")
    standard_results = run_walk_forward_optimization(
        data_dir=data_dir,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe
    )
    
    standard_params = standard_results['best_params']
    
    # 2. Run regime-based optimization
    print("\n--- Running Regime-Based Optimization ---")
    regime_results = run_regime_optimization(
        data_dir=data_dir,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe
    )
    
    regime_params = regime_results['regime_parameters']
    
    # 3. Run backtest with standard parameters
    print("\n--- Running Standard Backtest ---")
    standard_backtest, _, _ = run_backtest(
        data_dir=data_dir,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        fast_window=standard_params['fast_window'],
        slow_window=standard_params['slow_window'],
        fixed_position_size=100
    )
    
    # 4. Run backtest with regime-based parameters
    print("\n--- Running Regime-Based Backtest ---")
    regime_backtest, _, _ = run_regime_backtest(
        data_dir=data_dir,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        regime_params=regime_params
    )
    
    # 5. Compare results
    print("\n=== Performance Comparison ===")
    print(f"                     Standard      Regime-Based")
    print(f"Return:             {standard_backtest['return_pct']:>8.2f}%     {regime_backtest['return_pct']:>8.2f}%")
    print(f"Annual Return:      {standard_backtest['annual_return']:>8.2f}%     {regime_backtest['annual_return']:>8.2f}%")
    print(f"Max Drawdown:       {standard_backtest['max_drawdown']:>8.2f}%     {regime_backtest['max_drawdown']:>8.2f}%")
    print(f"Sharpe Ratio:       {standard_backtest['sharpe_ratio']:>8.2f}      {regime_backtest['sharpe_ratio']:>8.2f}")
    print(f"Trade Count:        {standard_backtest['trade_count']:>8d}      {regime_backtest['trade_count']:>8d}")
    print(f"Signal Count:       {standard_backtest['signal_count']:>8d}      {regime_backtest['signal_count']:>8d}")
    
    # Calculate improvement
    if standard_backtest['sharpe_ratio'] > 0:
        sharpe_improvement = (regime_backtest['sharpe_ratio'] / standard_backtest['sharpe_ratio'] - 1) * 100
        print(f"\nSharpe Ratio Improvement: {sharpe_improvement:.2f}%")
    
    return {
        'standard': {
            'params': standard_params,
            'results': standard_backtest
        },
        'regime_based': {
            'params': regime_params,
            'results': regime_backtest
        }
    }


if __name__ == "__main__":
    # Use absolute path with tilde expansion
    DATA_DIR = "~/adf/data"  # Path to your data directory
    DATA_DIR = os.path.expanduser(DATA_DIR)  # Expand tilde to full path
    
    print(f"Using data directory: {DATA_DIR}")
    
    # Define parameters for backtest
    SYMBOL = "SPY"
    START_DATE = "2024-03-26"
    END_DATE = "2024-04-10"
    TIMEFRAME = "1m"
    
    # Run demo
    
    compare_regime_vs_standard(
        data_dir=DATA_DIR,
        symbols=[SYMBOL],
        start_date=START_DATE,
        end_date=END_DATE,
        timeframe=TIMEFRAME
    )
