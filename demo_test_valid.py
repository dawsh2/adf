"""
Demo script for testing and validating optimization accuracy with synthetic data.

This script creates synthetic price data with known return characteristics and 
a simple, predictable trading strategy to verify that the calculation of returns
and other performance metrics is accurate.
"""

import logging
import numpy as np
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple

# Configure logging - Enhanced to show more detail
logging.basicConfig(
    level=logging.DEBUG,  # Using DEBUG level for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import necessary components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType, BarEvent
from src.execution.portfolio import PortfolioManager
from src.execution.brokers.simulated import SimulatedBroker
from src.strategy.risk.risk_manager import SimpleRiskManager
from src.execution.backtest.backtest import run_backtest
from src.models.optimization.validation import OptimizationValidator
from src.core.events.event_utils import create_bar_event

class SyntheticDataGenerator:
    """Generator for synthetic price data with known properties."""
    
    @staticmethod
    def create_linear_trend(days=30, start_price=100.0, daily_return=0.01):
        """
        Create a price series with a constant daily return.
        
        Args:
            days: Number of days of data
            start_price: Starting price
            daily_return: Fixed daily return (e.g., 0.01 = 1% daily)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Create date range
        start_date = datetime.datetime(2024, 1, 1)
        dates = [start_date + datetime.timedelta(days=i) for i in range(days)]
        
        # Calculate prices with compounding
        prices = [start_price * (1 + daily_return) ** i for i in range(days)]
        
        # Log expected values for debugging
        logger.info(f"Linear trend data - Start price: {start_price:.4f}, End price: {prices[-1]:.4f}")
        logger.info(f"Expected daily return: {daily_return:.4f} ({daily_return*100:.2f}%)")
        logger.info(f"Expected prices: First 5: {prices[:5]}, Last 5: {prices[-5:]}")
        
        # Create OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Simple OHLC based on close price
            open_price = prices[i-1] if i > 0 else close * 0.99
            high = max(open_price, close) * 1.01  # 1% above max of open/close
            low = min(open_price, close) * 0.99   # 1% below min of open/close
            volume = 1000000  # Constant volume
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # Calculate the expected total return for validation
        expected_total_return = (1 + daily_return) ** (days - 1) - 1
        logger.info(f"Expected total return after {days} days: {expected_total_return:.6f} ({expected_total_return*100:.2f}%)")
        
        return df, expected_total_return
    
    @staticmethod
    def create_sine_wave(days=30, start_price=100.0, amplitude=10.0, period_days=10):
        """
        Create a price series that follows a sine wave pattern.
        
        Args:
            days: Number of days of data
            start_price: Starting price
            amplitude: Price amplitude
            period_days: Days per complete cycle
            
        Returns:
            DataFrame with OHLCV data
        """
        # Create date range
        start_date = datetime.datetime(2024, 1, 1)
        dates = [start_date + datetime.timedelta(days=i) for i in range(days)]
        
        # Calculate prices using sine wave
        angular_freq = 2 * np.pi / period_days
        prices = [start_price + amplitude * np.sin(angular_freq * i) for i in range(days)]
        
        # Log expected values for debugging
        logger.info(f"Sine wave data - Start price: {prices[0]:.4f}, End price: {prices[-1]:.4f}")
        logger.info(f"Amplitude: {amplitude:.4f}, Period: {period_days} days")
        logger.info(f"Expected prices: First 5: {prices[:5]}, Last 5: {prices[-5:]}")
        
        # Create OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Simple OHLC based on close price
            open_price = prices[i-1] if i > 0 else close * 0.99
            high = max(open_price, close) * 1.01
            low = min(open_price, close) * 0.99
            volume = 1000000  # Constant volume
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # For a sine wave, the expected return over a full period is 0
        # But for partial periods, we can calculate it based on the actual data
        expected_total_return = (prices[-1] / prices[0]) - 1
        logger.info(f"Expected total return for sine wave: {expected_total_return:.6f} ({expected_total_return*100:.2f}%)")
        
        return df, expected_total_return
    
    @staticmethod
    def create_random_walk(days=30, start_price=100.0, daily_vol=0.01, drift=0.001):
        """
        Create a price series that follows a random walk with drift.
        
        Args:
            days: Number of days of data
            start_price: Starting price
            daily_vol: Daily volatility (standard deviation of returns)
            drift: Daily drift (mean of returns)
            
        Returns:
            DataFrame with OHLCV data and expected return
        """
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create date range
        start_date = datetime.datetime(2024, 1, 1)
        dates = [start_date + datetime.timedelta(days=i) for i in range(days)]
        
        # Generate daily returns from a normal distribution
        daily_returns = np.random.normal(drift, daily_vol, days - 1)
        
        # Convert returns to prices with compounding
        prices = [start_price]
        for ret in daily_returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Log expected values for debugging
        logger.info(f"Random walk data - Start price: {prices[0]:.4f}, End price: {prices[-1]:.4f}")
        logger.info(f"Drift: {drift:.4f}, Volatility: {daily_vol:.4f}")
        logger.info(f"Daily returns: Mean={np.mean(daily_returns):.6f}, Std={np.std(daily_returns):.6f}")
        logger.info(f"Expected prices: First 5: {prices[:5]}, Last 5: {prices[-5:]}")
        
        # Create OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Simple OHLC based on close price
            open_price = prices[i-1] if i > 0 else close * 0.99
            high = max(open_price, close) * (1 + daily_vol)
            low = min(open_price, close) * (1 - daily_vol)
            volume = 1000000  # Constant volume
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # The expected return includes the drift component
        expected_total_return = (prices[-1] / prices[0]) - 1
        theoretical_expected_return = (1 + drift) ** (days - 1) - 1
        
        logger.info(f"Random walk: Actual total return = {expected_total_return:.6f} ({expected_total_return*100:.2f}%), " +
                   f"Theoretical expected return = {theoretical_expected_return:.6f} ({theoretical_expected_return*100:.2f}%)")
        
        return df, expected_total_return

class PredictableStrategy:
    """
    A strategy with predictable behavior for validation testing.
    
    This strategy follows simple, deterministic rules to make
    its behavior easy to validate.
    """
    
    def __init__(self, symbols=None, threshold=0.0, position_mode='binary', 
                 lookback=1, name="predictable_strategy"):
        """
        Initialize the strategy.
        
        Args:
            symbols: List of symbols to trade
            threshold: Return threshold to trigger signals
            position_mode: How to determine position ('binary', 'proportional')
            lookback: Number of bars to look back for return calculation
            name: Strategy name
        """
        self.symbols = symbols or ['SYNTHETIC']
        self.threshold = threshold
        self.position_mode = position_mode
        self.lookback = lookback
        self.name = name
        self.event_bus = None
        
        # Internal state tracking
        self.last_bars = {}  # symbol -> list of last N bars
        self.signals_generated = 0
        self.positions = {}  # symbol -> current position (1=long, 0=neutral, -1=short)
        self.trade_log = []  # Detailed log of all trades
        
        # Initialize debug counter for tracking bar processing
        self.bars_processed = 0
        
        logger.info(f"Initialized {self.name}: threshold={self.threshold}, mode={self.position_mode}, lookback={self.lookback}")
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
        return self
    
    def get_parameters(self):
        """Get strategy parameters."""
        return {
            'threshold': self.threshold,
            'position_mode': self.position_mode,
            'lookback': self.lookback
        }
    
    def set_parameters(self, params):
        """Set strategy parameters."""
        if 'threshold' in params:
            self.threshold = params['threshold']
        if 'position_mode' in params:
            self.position_mode = params['position_mode']
        if 'lookback' in params:
            self.lookback = params['lookback']
        return self
    
    def on_bar(self, event):
        """Process a bar event and generate signals."""
        if not isinstance(event, BarEvent):
            logger.debug(f"Ignoring non-bar event: {event.get_type() if hasattr(event, 'get_type') else type(event)}")
            return
            
        symbol = event.get_symbol()
        self.bars_processed += 1
        
        # Only process symbols we're interested in
        if symbol not in self.symbols:
            logger.debug(f"Ignoring bar for untracked symbol: {symbol}")
            return
        
        # Get bar details for logging
        timestamp = event.get_timestamp()
        open_price = event.get_open()
        high_price = event.get_high()
        low_price = event.get_low() 
        close_price = event.get_close()
        
        logger.debug(f"Processing bar #{self.bars_processed}: {symbol} @ {timestamp}: " +
                   f"OHLC=[{open_price:.4f}, {high_price:.4f}, {low_price:.4f}, {close_price:.4f}]")
        
        # Store bars for this symbol
        if symbol not in self.last_bars:
            self.last_bars[symbol] = []
        
        # Add current bar to history
        self.last_bars[symbol].append(event)
        
        # Keep only the lookback window
        if len(self.last_bars[symbol]) > max(2, self.lookback + 1):
            self.last_bars[symbol].pop(0)
            
        # Need at least 2 bars to calculate return
        if len(self.last_bars[symbol]) < 2:
            logger.debug(f"Not enough bars to calculate return for {symbol}. Need at least 2, have {len(self.last_bars[symbol])}")
            return
            
        # Calculate return based on close prices
        if len(self.last_bars[symbol]) >= self.lookback + 1:
            # Get current price and price 'lookback' bars ago
            current_bar = self.last_bars[symbol][-1]
            previous_bar = self.last_bars[symbol][-(self.lookback + 1)]
            
            current_price = current_bar.get_close()
            previous_price = previous_bar.get_close()
            
            # Calculate return
            period_return = (current_price / previous_price) - 1
            
            # Log the return calculation
            logger.debug(f"Return calculation for {symbol}: " +
                       f"Current price={current_price:.4f} (at {current_bar.get_timestamp()}), " +
                       f"Previous price={previous_price:.4f} (at {previous_bar.get_timestamp()}), " +
                       f"Return={period_return:.6f} ({period_return*100:.2f}%)")
            
            # Generate trading signal based on predictable rules
            self._generate_signal(symbol, period_return, current_price, event.get_timestamp())
        else:
            logger.debug(f"Not enough bars to satisfy lookback={self.lookback} for {symbol}. " + 
                       f"Have {len(self.last_bars[symbol])} bars.")
    
    def _generate_signal(self, symbol, period_return, price, timestamp):
        """Generate a trading signal based on the calculated return."""
        # Default to neutral
        signal_value = 0
        
        # Determine signal based on threshold
        if self.position_mode == 'binary':
            # Binary position: fully long or fully short
            if period_return > self.threshold:
                signal_value = 1  # BUY signal
                logger.debug(f"Return {period_return:.6f} > threshold {self.threshold}, generating BUY signal")
            elif period_return < -self.threshold:
                signal_value = -1  # SELL signal
                logger.debug(f"Return {period_return:.6f} < -threshold {-self.threshold}, generating SELL signal")
            else:
                logger.debug(f"Return {period_return:.6f} within threshold bounds [-{self.threshold}:{self.threshold}], no signal")
        elif self.position_mode == 'proportional':
            # Position size proportional to return
            signal_strength = min(1.0, abs(period_return / self.threshold)) if self.threshold > 0 else 1.0
            if period_return > 0:
                signal_value = signal_strength  # Partial BUY signal
                logger.debug(f"Positive return {period_return:.6f}, generating partial BUY signal: {signal_strength:.4f}")
            elif period_return < 0:
                signal_value = -signal_strength  # Partial SELL signal
                logger.debug(f"Negative return {period_return:.6f}, generating partial SELL signal: {-signal_strength:.4f}")
            else:
                logger.debug(f"Zero return, no signal generated")
        
        # Only emit if signal is non-zero or position changed
        current_position = self.positions.get(symbol, 0)
        if signal_value != 0 and signal_value != current_position:
            # Create and emit signal event
            from src.core.events.event_utils import create_signal_event
            
            signal = create_signal_event(
                signal_value=signal_value,
                price=price,
                symbol=symbol,
                rule_id=f"{self.name}_{self.lookback}_{self.threshold}",
                timestamp=timestamp
            )
            
            # Log the signal
            signal_type = "BUY" if signal_value > 0 else "SELL"
            logger.info(f"Generating {signal_type} signal for {symbol} @ {price:.4f} " +
                      f"(position change: {current_position} -> {signal_value})")
            
            # Update current position
            self.positions[symbol] = signal_value
            
            # Record trade in log
            self.trade_log.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'price': price,
                'signal_value': signal_value,
                'position_change': (signal_value - current_position),
                'return': period_return
            })
            
            # Emit signal
            if self.event_bus:
                self.event_bus.emit(signal)
                self.signals_generated += 1
                logger.debug(f"Signal #{self.signals_generated} emitted to event bus")
            else:
                logger.warning("No event bus available to emit signal!")
        else:
            if signal_value == 0:
                logger.debug(f"No signal generated (signal_value = 0)")
            elif signal_value == current_position:
                logger.debug(f"No signal generated (already in position: {current_position})")
    
    def reset(self):
        """Reset internal state."""
        self.last_bars = {}
        self.signals_generated = 0
        self.positions = {}
        self.trade_log = []
        self.bars_processed = 0
        logger.info(f"Strategy {self.name} reset")

# Enhanced event collector for debugging
class EventCollector:
    """
    Collects and logs events for debugging purposes.
    """
    
    def __init__(self, name="event_collector"):
        self.name = name
        self.events = {event_type: [] for event_type in EventType}
        self.event_counts = {event_type: 0 for event_type in EventType}
        
    def collect_event(self, event):
        """Collect an event for analysis."""
        event_type = event.get_type()
        self.events[event_type].append(event)
        self.event_counts[event_type] += 1
        
        # Log different types of events with appropriate detail
        if event_type == EventType.BAR:
            if self.event_counts[event_type] <= 5 or self.event_counts[event_type] % 10 == 0:
                logger.debug(f"Bar #{self.event_counts[event_type]}: {event.get_symbol()} @ {event.get_timestamp()} - Close: {event.get_close():.4f}")
                
        elif event_type == EventType.SIGNAL:
            signal_value = event.data.get('signal_value', 0)
            signal_name = "BUY" if signal_value > 0 else "SELL" if signal_value < 0 else "NEUTRAL"
            logger.info(f"Signal #{self.event_counts[event_type]}: {event.get_symbol()} - {signal_name} @ {event.data.get('price', 0):.4f}")
            
        elif event_type == EventType.ORDER:
            direction = event.data.get('direction', 'Unknown')
            quantity = event.data.get('quantity', 0)
            price = event.data.get('price', 0)
            logger.info(f"Order #{self.event_counts[event_type]}: {event.get_symbol()} - {direction} {quantity} @ {price:.4f}")
            
        elif event_type == EventType.FILL:
            direction = event.data.get('direction', 'Unknown')
            quantity = event.data.get('quantity', 0)
            price = event.data.get('price', 0)
            logger.info(f"Fill #{self.event_counts[event_type]}: {event.get_symbol()} - {direction} {quantity} @ {price:.4f}")
    
    def get_summary(self):
        """Get a summary of collected events."""
        return {event_type.name: count for event_type, count in self.event_counts.items() if count > 0}

class EnhancedPortfolioManager(PortfolioManager):
    """
    Enhanced portfolio manager with detailed equity tracking.
    """
    
    def __init__(self, initial_cash=0.0, event_bus=None):
        """Initialize enhanced portfolio manager."""
        super().__init__(initial_cash, event_bus)
        # Add detailed equity tracking
        self.equity_history_detailed = []
        self.fill_details = []
    
    def on_fill(self, event):
        """Enhanced fill handler with detailed logging."""
        # First call the parent implementation
        super().on_fill(event)
        
        # Then add our enhanced logging
        symbol = event.get_symbol()
        direction = event.get_direction()
        quantity = event.get_quantity()
        price = event.get_price()
        commission = event.get_commission() if hasattr(event, 'get_commission') else 0.0
        timestamp = event.get_timestamp()
        
        # Calculate impact on portfolio
        fill_value = price * quantity
        action = "BUY" if direction == "BUY" else "SELL"
        cash_impact = -fill_value if action == "BUY" else fill_value
        cash_after = self.cash
        position_value = self.get_position_value()
        equity = cash_after + position_value
        
        # Record detailed fill information
        self.fill_details.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'fill_value': fill_value,
            'cash_impact': cash_impact,
            'cash_after': cash_after,
            'position_value': position_value,
            'equity': equity
        })
        
        # Update detailed equity history
        self.equity_history_detailed.append({
            'timestamp': timestamp,
            'event': 'FILL',
            'cash': cash_after,
            'position_value': position_value,
            'equity': equity
        })
        
        logger.info(f"Fill processed: {symbol} {action} {quantity} @ {price:.4f}, " +
                  f"Fill value: {fill_value:.2f}, Cash impact: {cash_impact:.2f}, " +
                  f"Cash after: {cash_after:.2f}, Position value: {position_value:.2f}, " +
                  f"Equity: {equity:.2f}")
    
    def update_equity_snapshot(self, timestamp, prices=None):
        """Take a snapshot of current equity state."""
        position_value = self.get_position_value(prices)
        equity = self.cash + position_value
        
        # Record the snapshot
        self.equity_history_detailed.append({
            'timestamp': timestamp,
            'event': 'SNAPSHOT',
            'cash': self.cash,
            'position_value': position_value,
            'equity': equity
        })
        
        logger.debug(f"Equity snapshot @ {timestamp}: " +
                   f"Cash: {self.cash:.2f}, Position value: {position_value:.2f}, " +
                   f"Equity: {equity:.2f}")
        
        return equity

class SyntheticDataHandler:
    """Data handler for synthetic data with bar event emission."""
    
    def __init__(self, bar_emitter):
        """Initialize the synthetic data handler."""
        self.bar_emitter = bar_emitter
        self.data_frames = {}  # symbol -> DataFrame
        self.current_idx = {}  # symbol -> current index
        self.equity_tracker = None  # For tracking equity after each bar
    
    def set_equity_tracker(self, portfolio):
        """Set the portfolio to track equity after each bar."""
        self.equity_tracker = portfolio
    
    def load_data(self, symbol, df):
        """
        Load synthetic data for a symbol.
        
        Args:
            symbol: Symbol to load data for
            df: DataFrame with OHLCV data
        """
        self.data_frames[symbol] = df
        self.current_idx[symbol] = 0
        
        logger.info(f"Loaded synthetic data for {symbol}: {len(df)} bars from " +
                  f"{df.index[0]} to {df.index[-1]}")
    
    def get_next_bar(self, symbol):
        """Get the next bar for a symbol."""
        # Check if we have data for this symbol
        if symbol not in self.data_frames:
            logger.warning(f"No data loaded for symbol: {symbol}")
            return None
            
        df = self.data_frames[symbol]
        idx = self.current_idx[symbol]
        
        # Check if we've reached the end of the data
        if idx >= len(df):
            logger.debug(f"End of data reached for {symbol} (idx={idx}, data_length={len(df)})")
            return None
            
        # Get the row
        row = df.iloc[idx]
        
        # Get timestamp from index
        timestamp = df.index[idx]
        
        # Create bar event
        bar = create_bar_event(
            symbol=symbol,
            timestamp=timestamp,
            open_price=float(row['open']),
            high_price=float(row['high']),
            low_price=float(row['low']),
            close_price=float(row['close']),
            volume=int(row['volume'])
        )
        
        # Log bar details
        logger.debug(f"Bar created for {symbol} @ {timestamp}: " +
                   f"OHLC=[{row['open']:.4f}, {row['high']:.4f}, {row['low']:.4f}, {row['close']:.4f}], " +
                   f"idx={idx}")
        
        # Increment index
        self.current_idx[symbol] = idx + 1
        
        # Emit the bar event
        if self.bar_emitter:
            self.bar_emitter.emit(bar)
            logger.debug(f"Bar emitted for {symbol} @ {timestamp}")
        else:
            logger.warning(f"No bar emitter available to emit bar for {symbol}")
            
        # Update equity tracker if available
        if self.equity_tracker:
            # Update market data in the portfolio's tracking
            market_prices = {symbol: row['close']}
            self.equity_tracker.update_equity_snapshot(timestamp, market_prices)
            
        return bar
    
    def reset(self):
        """Reset the data handler state."""
        prev_indices = {s: self.current_idx.get(s, 0) for s in self.data_frames}
        self.current_idx = {symbol: 0 for symbol in self.data_frames}
        logger.info(f"Data handler reset. Previous indices: {prev_indices}, New indices: {self.current_idx}")
    
    def get_symbols(self):
        """Get the list of available symbols."""
        return list(self.data_frames.keys())

class ReturnCalculationValidator:
    """
    Validator for return calculation accuracy.
    
    This class compares the calculated returns from the system
    with the expected returns based on synthetic data.
    """
    
    def __init__(self, tolerance=1e-6):
        """
        Initialize the validator.
        
        Args:
            tolerance: Tolerance for floating point comparisons
        """
        self.tolerance = tolerance
        self.validation_results = {}
    
    def validate_total_return(self, expected_return, calculated_return):
        """
        Validate the total return calculation.
        
        Args:
            expected_return: Expected return value
            calculated_return: Return calculated by the system
            
        Returns:
            dict: Validation results
        """
        # Calculate absolute and relative differences
        abs_diff = abs(expected_return - calculated_return)
        rel_diff = abs_diff / (abs(expected_return) if expected_return != 0 else 1.0)
        
        # Check if within tolerance
        is_valid = abs_diff <= self.tolerance
        
        result = {
            'expected_return': expected_return,
            'calculated_return': calculated_return,
            'absolute_difference': abs_diff,
            'relative_difference': rel_diff,
            'is_valid': is_valid
        }
        
        # Store result
        self.validation_results['total_return'] = result
        
        # Log result
        logger.info(f"Return validation: expected={expected_return:.6f} ({expected_return*100:.2f}%), " +
                   f"calculated={calculated_return:.6f} ({calculated_return*100:.2f}%), " +
                   f"diff={abs_diff:.6f} ({abs_diff*100:.2f}%), " +
                   f"valid={is_valid}")
        
        return result
    
    def validate_equity_curve(self, expected_prices, equity_curve):
        """
        Validate the equity curve against expected prices.
        
        Args:
            expected_prices: Expected price series
            equity_curve: Equity curve from the system
            
        Returns:
            dict: Validation results
        """
        # Log the shapes
        logger.info(f"Equity curve validation - Expected prices length: {len(expected_prices)}")
        logger.info(f"Equity curve shape: {equity_curve.shape}, columns: {list(equity_curve.columns)}")
        
        # Align indices
        if len(expected_prices) != len(equity_curve):
            logger.warning(f"Length mismatch: expected={len(expected_prices)}, " +
                          f"calculated={len(equity_curve)}")
            
            # Truncate to the shorter length
            length = min(len(expected_prices), len(equity_curve))
            expected_prices_subset = expected_prices[:length]
            equity_curve_subset = equity_curve.iloc[:length]
            
            logger.info(f"Using truncated series for comparison (length={length})")
        else:
            expected_prices_subset = expected_prices
            equity_curve_subset = equity_curve
        
        # Calculate differences
        if 'equity' in equity_curve.columns:
            equity_values = equity_curve_subset['equity'].values
            
            # Normalize both series to start at the same value
            start_price = expected_prices_subset[0]
            start_equity = equity_values[0]
            
            normalized_prices = [p / start_price for p in expected_prices_subset]
            normalized_equity = [e / start_equity for e in equity_values]
            
            # Log detailed comparison for the first few points
            logger.info("Detailed equity curve comparison (first 5 points):")
            for i in range(min(5, len(normalized_prices))):
                logger.info(f"  Point {i+1}: Price={normalized_prices[i]:.6f}, Equity={normalized_equity[i]:.6f}, " +
                         f"Diff={(normalized_prices[i]-normalized_equity[i]):.6f}")
            
            # Calculate differences
            diffs = [abs(p - e) for p, e in zip(normalized_prices, normalized_equity)]
            max_diff = max(diffs)
            max_diff_idx = diffs.index(max_diff)
            avg_diff = sum(diffs) / len(diffs)
            
            # Log where the maximum difference occurred
            logger.info(f"Maximum difference of {max_diff:.6f} at index {max_diff_idx} " +
                      f"(Price={normalized_prices[max_diff_idx]:.6f}, Equity={normalized_equity[max_diff_idx]:.6f})")
            
            # Check if within tolerance
            is_valid = max_diff <= self.tolerance
            
            result = {
                'max_difference': max_diff,
                'max_diff_idx': max_diff_idx,
                'avg_difference': avg_diff,
                'is_valid': is_valid
            }
            
            # Store result
            self.validation_results['equity_curve'] = result
            
            # Log result
            logger.info(f"Equity curve validation: max_diff={max_diff:.6f}, " +
                       f"avg_diff={avg_diff:.6f}, valid={is_valid}")
            
            return result
        else:
            logger.warning("Equity curve does not contain 'equity' column")
            logger.info(f"Available columns: {list(equity_curve.columns)}")
            return {'is_valid': False, 'error': "Missing 'equity' column"}
    
    def generate_report(self):
        """
        Generate a validation report.
        
        Returns:
            str: Formatted report
        """
        if not self.validation_results:
            return "No validation results available."
            
        report = ["# Return Calculation Validation Report\n"]
        
        # Total return validation
        if 'total_return' in self.validation_results:
            result = self.validation_results['total_return']
            report.append("## Total Return Validation\n")
            report.append(f"Expected return: {result['expected_return']:.6f} ({result['expected_return']*100:.2f}%)")
            report.append(f"Calculated return: {result['calculated_return']:.6f} ({result['calculated_return']*100:.2f}%)")
            report.append(f"Absolute difference: {result['absolute_difference']:.6f} ({result['absolute_difference']*100:.2f}%)")
            report.append(f"Relative difference: {result['relative_difference']:.6f} ({result['relative_difference']*100:.4f}x)")
            report.append(f"Valid: {'Yes' if result['is_valid'] else 'No'}\n")
        
        # Equity curve validation
        if 'equity_curve' in self.validation_results:
            result = self.validation_results['equity_curve']
            report.append("## Equity Curve Validation\n")
            report.append(f"Maximum difference: {result['max_difference']:.6f} at index {result.get('max_diff_idx', 'unknown')}")
            report.append(f"Average difference: {result['avg_difference']:.6f}")
            report.append(f"Valid: {'Yes' if result['is_valid'] else 'No'}\n")
        
        # Summary
        report.append("## Summary\n")
        all_valid = all(result.get('is_valid', False) for result in self.validation_results.values())
        report.append(f"All validations passed: {'Yes' if all_valid else 'No'}")
        
        if not all_valid:
            report.append("\n## Potential Issues\n")
            if 'total_return' in self.validation_results and not self.validation_results['total_return']['is_valid']:
                report.append("- The calculated return differs significantly from the expected return")
                if self.validation_results['total_return']['calculated_return'] < self.validation_results['total_return']['expected_return']:
                    report.append("  - The system is *understating* returns")
                else:
                    report.append("  - The system is *overstating* returns")
                    
            if 'equity_curve' in self.validation_results and not self.validation_results['equity_curve']['is_valid']:
                report.append("- The equity curve shape differs from expected price movements")
                report.append("  - This could indicate issues with position sizing or order execution")
                
            report.append("\n## Recommendations\n")
            report.append("1. Check signal generation logic to ensure all expected signals are created")
            report.append("2. Verify position sizing accurately reflects desired exposure")
            report.append("3. Check equity calculation after each trade to ensure it's properly updated")
            report.append("4. Examine fill events to ensure they correctly update portfolio state")
            report.append("5. Verify that the return calculation correctly accounts for all cash flows")
        
        return "\n".join(report)

def setup_backtest_environment():
    """Set up the backtest environment components."""
    # Create event system
    event_bus = EventBus()
    event_manager = EventManager(event_bus)
    
    # Create event collector for debugging
    event_collector = EventCollector()
    
    # Register event collector with all event types
    for event_type in EventType:
        event_bus.register(event_type, event_collector.collect_event)
    
    # Create bar emitter
    from src.core.events.event_emitters import BarEmitter
    bar_emitter = BarEmitter(name="bar_emitter", event_bus=event_bus)
    
    # Create synthetic data handler
    data_handler = SyntheticDataHandler(bar_emitter)
    
    # Create enhanced portfolio and risk components
    portfolio = EnhancedPortfolioManager(initial_cash=10000.0)
    portfolio.set_event_bus(event_bus)
    
    # Connect data handler to portfolio for equity tracking
    data_handler.set_equity_tracker(portfolio)
    
    # Create broker simulation
    broker = SimulatedBroker()
    broker.set_event_bus(event_bus)
    
    # Create risk manager
    risk_manager = SimpleRiskManager(portfolio)
    risk_manager.set_event_bus(event_bus)
    
    # Register all components with event manager
    event_manager.register_component('portfolio', portfolio, [EventType.FILL])
    event_manager.register_component('risk_manager', risk_manager, [EventType.SIGNAL])
    event_manager.register_component('broker', broker, [EventType.ORDER])
    
    return {
        'event_bus': event_bus,
        'event_manager': event_manager,
        'data_handler': data_handler,
        'bar_emitter': bar_emitter,
        'portfolio': portfolio,
        'broker': broker,
        'risk_manager': risk_manager,
        'event_collector': event_collector
    }

def run_return_validation_enhanced():
    """Run the enhanced return calculation validation using synthetic data."""
    logger.info("\n===== ENHANCED RETURN CALCULATION VALIDATION =====\n")
    
    # 1. Set up test environment
    env = setup_backtest_environment()
    data_handler = env['data_handler']
    portfolio = env['portfolio']
    event_collector = env['event_collector']
    
    # 2. Generate synthetic data with known returns
    symbol = 'SYNTHETIC'
    
    # Create linear trend data with 1% daily return
    daily_return = 0.01  # 1% daily
    days = 30
    df, expected_return = SyntheticDataGenerator.create_linear_trend(
        days=days, 
        start_price=100.0,
        daily_return=daily_return
    )
    
    # Store the price series for validation
    expected_prices = df['close'].tolist()
    
    # Load data into handler
    data_handler.load_data(symbol, df)
    
    # 3. Create predictable strategy with lower threshold to ensure trades
    strategy = PredictableStrategy(
        symbols=[symbol],
        threshold=0.001,  # Lower threshold to ensure more signals
        position_mode='binary',  # Full position only
        lookback=1  # Look at 1-day returns
    )
    
    # Connect to event bus
    strategy.set_event_bus(env['event_bus'])
    
    # Register with event manager
    env['event_manager'].register_component('strategy', strategy, [EventType.BAR])
    
    # 4. Run backtest with custom position size
    logger.info("Running backtest with synthetic data...")
    
    # Create a DataFrame to track equity after each bar
    detailed_equity_df = pd.DataFrame(columns=['timestamp', 'equity', 'price'])
    
    # Process each bar manually to track equity
    data_handler.reset()
    portfolio.reset()
    strategy.reset()
    
    # First, get all prices
    all_prices = df['close'].tolist()
    
    # Manual bar processing loop
    bar_count = 0
    equity_values = []
    timestamps = []
    
    while True:
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break
            
        bar_count += 1
        
        # Record current equity and price
        timestamp = bar.get_timestamp()
        price = bar.get_close()
        equity = portfolio.get_equity({symbol: price})
        
        # Store values
        equity_values.append(equity)
        timestamps.append(timestamp)
        
        # Add to tracking DataFrame
        detailed_equity_df = pd.concat([
            detailed_equity_df,
            pd.DataFrame({
                'timestamp': [timestamp],
                'equity': [equity],
                'price': [price]
            })
        ])
        
        logger.debug(f"Bar {bar_count}: {timestamp}, Price={price:.4f}, Equity={equity:.2f}")
    
    # Log event summary
    event_summary = event_collector.get_summary()
    logger.info(f"Event summary: {event_summary}")
    
    # Log strategy trades 
    logger.info(f"Strategy generated {len(strategy.trade_log)} trades:")
    for i, trade in enumerate(strategy.trade_log):
        logger.info(f"  Trade {i+1}: {trade['symbol']} @ {trade['timestamp']} - " +
                   f"Price={trade['price']:.4f}, Signal={trade['signal_value']}, " +
                   f"Return={trade['return']:.6f}")
    
    # 5. Calculate expected vs. actual returns
    if len(equity_values) > 0:
        initial_equity = equity_values[0]
        final_equity = equity_values[-1]
        calculated_return = (final_equity / initial_equity) - 1
        
        logger.info(f"Initial equity: {initial_equity:.2f}")
        logger.info(f"Final equity: {final_equity:.2f}")
        logger.info(f"Calculated return: {calculated_return:.6f}")
        logger.info(f"Expected return: {expected_return:.6f}")
        
        # 6. Get portfolio's detailed equity history
        detailed_equity_history = pd.DataFrame(portfolio.equity_history_detailed)
        if not detailed_equity_history.empty and 'equity' in detailed_equity_history.columns:
            logger.info(f"Portfolio recorded {len(detailed_equity_history)} equity history points")
            logger.info(f"First 5 equity history points:")
            for i, row in detailed_equity_history.head().iterrows():
                logger.info(f"  {i+1}: {row['timestamp']} - Event: {row['event']}, " +
                          f"Cash: {row['cash']:.2f}, Position: {row['position_value']:.2f}, " +
                          f"Equity: {row['equity']:.2f}")
        else:
            logger.warning("No detailed equity history available from portfolio")
            
        # 7. Validate returns
        validator = ReturnCalculationValidator()
        return_validation = validator.validate_total_return(expected_return, calculated_return)
        
        # Use the manually tracked equity curve for validation
        if not detailed_equity_df.empty:
            logger.info(f"Using {len(detailed_equity_df)} manually tracked equity points for validation")
            curve_validation = validator.validate_equity_curve(expected_prices, detailed_equity_df)
        else:
            logger.warning("No manually tracked equity data available")
        
        # 8. Generate report
        report = validator.generate_report()
        print("\nReturn Calculation Validation Report:")
        print(report)
        
        # 9. Plot comparison with trades
        try:
            plt.figure(figsize=(15, 10))
            
            # Plot equity curve
            ax1 = plt.subplot(3, 1, 1)
            ax1.plot(detailed_equity_df['timestamp'], detailed_equity_df['equity'], label='Equity', color='blue')
            ax1.set_title('Equity Curve')
            ax1.set_ylabel('Equity ($)')
            ax1.legend()
            
            # Plot expected prices
            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            ax2.plot(df.index, df['close'], label='Expected Prices', color='green')
            ax2.set_ylabel('Price ($)')
            ax2.legend()
            
            # Plot delta between normalized equity and price
            ax3 = plt.subplot(3, 1, 3, sharex=ax1)
            
            # Normalize both to starting value
            norm_equity = detailed_equity_df['equity'] / detailed_equity_df['equity'].iloc[0]
            norm_price = detailed_equity_df['price'] / detailed_equity_df['price'].iloc[0]
            
            # Calculate delta
            delta = norm_equity - norm_price
            
            ax3.plot(detailed_equity_df['timestamp'], delta, label='Equity-Price Delta', color='red')
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.set_ylabel('Delta (normalized)')
            ax3.set_xlabel('Date')
            ax3.legend()
            
            # Add annotations for trades
            if strategy.trade_log:
                # Add buy and sell markers on the price chart
                for trade in strategy.trade_log:
                    timestamp = trade['timestamp']
                    price = trade['price']
                    signal = trade['signal_value']
                    
                    if signal > 0:  # Buy signal
                        ax2.plot(timestamp, price, '^', color='green', markersize=10)
                    elif signal < 0:  # Sell signal
                        ax2.plot(timestamp, price, 'v', color='red', markersize=10)
            
            plt.tight_layout()
            plt.savefig('return_validation_enhanced.png')
            logger.info("Enhanced plot saved to return_validation_enhanced.png")
        except Exception as e:
            logger.warning(f"Could not generate enhanced plot: {e}")
        
        # Return validation results
        return {
            'strategy': strategy,
            'equity_df': detailed_equity_df,
            'portfolio': portfolio,
            'expected_return': expected_return,
            'calculated_return': calculated_return,
            'validation': return_validation,
            'validator': validator,
            'event_summary': event_summary
        }
    else:
        logger.error("No equity data available from backtest")
        return None

def run_multi_data_validation():
    """Run validation with multiple types of synthetic data."""
    logger.info("\n===== MULTI-DATA VALIDATION =====\n")
    
    # Set up test environment
    env = setup_backtest_environment()
    data_handler = env['data_handler']
    portfolio = env['portfolio']
    
    # Generate different types of synthetic data
    validation_results = {}
    
    # Test cases with different data patterns
    test_cases = [
        {
            'name': 'linear_trend',
            'generator': SyntheticDataGenerator.create_linear_trend,
            'params': {'days': 30, 'start_price': 100.0, 'daily_return': 0.01}
        },
        {
            'name': 'sine_wave',
            'generator': SyntheticDataGenerator.create_sine_wave,
            'params': {'days': 30, 'start_price': 100.0, 'amplitude': 10.0, 'period_days': 10}
        },
        {
            'name': 'random_walk',
            'generator': SyntheticDataGenerator.create_random_walk,
            'params': {'days': 30, 'start_price': 100.0, 'daily_vol': 0.01, 'drift': 0.001}
        }
    ]
    
    for case in test_cases:
        logger.info(f"\nTesting with {case['name']} data pattern...")
        
        # Generate data
        symbol = f"SYNTHETIC_{case['name']}"
        df, expected_return = case['generator'](**case['params'])
        
        # Store expected prices
        expected_prices = df['close'].tolist()
        
        # Reset components
        data_handler.reset()
        portfolio.reset()
        
        # Load new data
        data_handler.load_data(symbol, df)
        
        # Create strategy for this test with lower threshold to ensure trading
        strategy = PredictableStrategy(
            symbols=[symbol],
            threshold=0.001,  # Lower threshold to ensure more signals
            position_mode='binary',
            lookback=1
        )
        
        # Connect to event bus and register with event manager
        strategy.set_event_bus(env['event_bus'])
        env['event_manager'].register_component(f'strategy_{case["name"]}', strategy, [EventType.BAR])
        
        # Create DataFrame to track equity manually
        detailed_equity_df = pd.DataFrame(columns=['timestamp', 'equity', 'price'])
        
        # Process each bar manually
        bar_count = 0
        equity_values = []
        
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
                
            bar_count += 1
            
            # Record equity and price
            timestamp = bar.get_timestamp()
            price = bar.get_close()
            equity = portfolio.get_equity({symbol: price})
            
            # Store values
            equity_values.append(equity)
            
            # Add to tracking DataFrame
            detailed_equity_df = pd.concat([
                detailed_equity_df,
                pd.DataFrame({
                    'timestamp': [timestamp],
                    'equity': [equity],
                    'price': [price]
                })
            ])
            
            if bar_count <= 5 or bar_count % 10 == 0:
                logger.debug(f"Bar {bar_count}: {timestamp}, Price={price:.4f}, Equity={equity:.2f}")
        
        # Log strategy trades
        logger.info(f"Strategy generated {len(strategy.trade_log)} trades for {case['name']}:")
        for i, trade in enumerate(strategy.trade_log[:5]):  # Show first 5 trades
            logger.info(f"  Trade {i+1}: {trade['symbol']} @ {trade['timestamp']} - " +
                       f"Price={trade['price']:.4f}, Signal={trade['signal_value']}")
        
        # Calculate returns
        if len(equity_values) > 0:
            initial_equity = equity_values[0]
            final_equity = equity_values[-1]
            calculated_return = (final_equity / initial_equity) - 1
            
            logger.info(f"Initial equity: {initial_equity:.2f}")
            logger.info(f"Final equity: {final_equity:.2f}")
            logger.info(f"Calculated return: {calculated_return:.6f} ({calculated_return*100:.2f}%)")
            logger.info(f"Expected return: {expected_return:.6f} ({expected_return*100:.2f}%)")
            
            # Validate returns
            validator = ReturnCalculationValidator()
            return_validation = validator.validate_total_return(expected_return, calculated_return)
            curve_validation = validator.validate_equity_curve(expected_prices, detailed_equity_df)
            
            # Store results
            validation_results[case['name']] = {
                'expected_return': expected_return,
                'calculated_return': calculated_return,
                'equity_df': detailed_equity_df,
                'trades': strategy.trade_log,
                'validation': return_validation,
                'is_valid': return_validation['is_valid'],
                'bar_count': bar_count,
                'trade_count': len(strategy.trade_log)
            }
            
            # Plot comparison if not running multi-test
            try:
                plt.figure(figsize=(12, 8))
                
                # Plot equity and price
                ax = plt.subplot(2, 1, 1)
                ax.plot(detailed_equity_df['timestamp'], detailed_equity_df['equity'], 
                        label=f'Equity ({calculated_return*100:.2f}%)', color='blue')
                ax.plot(df.index, df['close'], 
                        label=f'Price ({expected_return*100:.2f}%)', color='green')
                
                # Add trade markers
                for trade in strategy.trade_log:
                    timestamp = trade['timestamp']
                    price = trade['price']
                    signal = trade['signal_value']
                    
                    if signal > 0:  # Buy signal
                        ax.plot(timestamp, price, '^', color='green', markersize=10)
                    elif signal < 0:  # Sell signal
                        ax.plot(timestamp, price, 'v', color='red', markersize=10)
                
                ax.set_title(f"{case['name']} - Equity vs Price")
                ax.legend()
                
                # Plot equity/price ratio
                ax2 = plt.subplot(2, 1, 2, sharex=ax)
                ratio = detailed_equity_df['equity'] / detailed_equity_df['equity'].iloc[0]
                price_ratio = detailed_equity_df['price'] / detailed_equity_df['price'].iloc[0]
                ax2.plot(detailed_equity_df['timestamp'], ratio, label='Normalized Equity', color='blue')
                ax2.plot(detailed_equity_df['timestamp'], price_ratio, label='Normalized Price', color='green')
                ax2.set_ylabel('Ratio to Initial Value')
                ax2.set_xlabel('Date')
                ax2.legend()
                
                plt.tight_layout()
                plt.savefig(f'validation_{case["name"]}.png')
                logger.info(f"Plot for {case['name']} saved to validation_{case['name']}.png")
            except Exception as e:
                logger.warning(f"Could not generate plot for {case['name']}: {e}")
    
    # Generate summary report
    print("\nMulti-Data Validation Summary:")
    print("="*60)
    
    print(f"{'Pattern'.ljust(15)} | {'Return'.ljust(20)} | {'Error'.ljust(10)} | {'Bars'.ljust(5)} | {'Trades'}")
    print("-"*60)
    
    all_valid = True
    for name, result in validation_results.items():
        is_valid = result['is_valid']
        all_valid &= is_valid
        expected = result['expected_return']
        calculated = result['calculated_return']
        error = abs(expected - calculated)
        error_pct = (error / abs(expected) if expected != 0 else error) * 100
        
        print(f"{name.ljust(15)} | " +
              f"Exp: {expected*100:+.2f}% Calc: {calculated*100:+.2f}% | " +
              f"{error_pct:.2f}% | " +
              f"{result['bar_count']:4d} | " +
              f"{result['trade_count']:3d}")
    
    print("-"*60)
    print(f"Overall validation: {'Passed' if all_valid else 'Failed'}")
    
    # Print common issues if validation failed
    if not all_valid:
        print("\nPotential Issues Identified:")
        
        # Check if calculated returns are consistently lower than expected
        underestimated = [name for name, result in validation_results.items() 
                         if result['calculated_return'] < result['expected_return']]
        
        if len(underestimated) == len(validation_results):
            print("- System consistently UNDERESTIMATES returns across all test cases")
            print("  → This suggests positions are not being sized properly or trades aren't executing as expected")
        
        # Check if some patterns generate very few trades
        low_trade_count = [name for name, result in validation_results.items() 
                          if result['trade_count'] < result['bar_count'] * 0.2]  # Less than 20% of bars
        
        if low_trade_count:
            print(f"- Low trade count in patterns: {', '.join(low_trade_count)}")
            print("  → Strategy may not be generating enough signals or threshold is too high")
        
        # Check if bar count matches expected
        incorrect_bar_count = [name for name, result in validation_results.items() 
                             if result['bar_count'] != len(validation_results[name]['equity_df'])]
        
        if incorrect_bar_count:
            print(f"- Mismatch between expected and actual bar count: {', '.join(incorrect_bar_count)}")
            print("  → Some bars may not be properly processed or equity not tracked for all bars")
    
    return validation_results

if __name__ == "__main__":
    print("\n" + "="*80)
    print("RETURN CALCULATION VALIDATION DEMO")
    print("="*80 + "\n")
    
    # Run enhanced validation with detailed logging for linear trend data
    validation_results = run_return_validation_enhanced()
    
    # Run validation with multiple data patterns
    multi_results = run_multi_data_validation()
    
    print("\n" + "="*80)
    print("VALIDATION DEMO COMPLETE")
    print("="*80 + "\n")
