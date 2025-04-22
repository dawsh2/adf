#!/usr/bin/env python
"""
Regime-Based Optimization Example

This script demonstrates how to use the regime detection and optimization components
to tune strategy parameters for different market regimes.
"""
import os
import sys
import logging
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import data components
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler

# Import event system components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType, BarEvent
from src.core.events.event_emitters import BarEmitter
from src.core.events.event_utils import EventTracker

# Import regime detection and optimization
from src.models.filters.regime.regime_detector import MarketRegime, EnhancedRegimeDetector
from src.models.filters.regime.detector_factory import RegimeDetectorFactory
from src.models.optimization.regime_optimizer import RegimeSpecificOptimizer
from src.models.optimization.grid_search import GridSearchOptimizer

# Import strategy components
from src.models.filters.regime.regime_strategy import RegimeAwareStrategy

# Import moving average strategy for this example
# This assumes you have a MovingAverageCrossoverStrategy implementation
# If you don't, you'll need to create or import a different strategy
from src.strategy.strategies.ma_crossover import MovingAverageCrossoverStrategy

from src.models.optimization.datetime_utils import make_timestamps_compatible

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sample_data(data_dir, symbol, timeframe):
    """
    Load sample market data for demonstration.
    
    Args:
        data_dir: Directory containing data files
        symbol: Symbol to load
        timeframe: Data timeframe
        
    Returns:
        tuple: (data_source, data_handler) for the loaded data
    """
    # Create data source
    data_source = CSVDataSource(
        data_dir=data_dir,
        filename_pattern='{symbol}_{timeframe}.csv',
        date_column='timestamp',
        date_format=None,  # Auto-detect format
        column_map={
            'open': ['Open', 'open'],
            'high': ['High', 'high'],
            'low': ['Low', 'low'],
            'close': ['Close', 'close'],
            'volume': ['Volume', 'volume']
        }
    )
    
    # Create data handler
    data_handler = HistoricalDataHandler(data_source, None)
    
    # Load data
    data_handler.load_data(symbol, timeframe=timeframe)
    
    return data_source, data_handler

def create_self_balancing_regime_detector(data_handler, symbol, 
                                         start_date=None, end_date=None,
                                         target_sideways_pct=0.50,
                                         target_uptrend_pct=0.20,
                                         target_downtrend_pct=0.15,
                                         target_volatile_pct=0.15,
                                         max_iterations=10):
    """
    Creates a regime detector with parameters tuned to achieve target regime distribution.
    """
    logger.info("Creating self-balancing regime detector")
    
    # Parse date strings to datetime objects if needed
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Define a custom regime detector that overrides the update method
    class ForceBalancedRegimeDetector(EnhancedRegimeDetector):
        """A regime detector that forces a balanced distribution"""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.target_distribution = {
                MarketRegime.SIDEWAYS: target_sideways_pct,
                MarketRegime.UPTREND: target_uptrend_pct, 
                MarketRegime.DOWNTREND: target_downtrend_pct,
                MarketRegime.VOLATILE: target_volatile_pct
            }
            self.current_counts = {regime: 0 for regime in MarketRegime}
            self.total_bars = 0
            
            # For storing assigned regimes before they go into history
            self.assigned_regimes = {}  # timestamp -> regime
            
        def update(self, bar):
            """Override update to ensure balanced distribution"""
            symbol = bar.get_symbol()
            timestamp = bar.get_timestamp()
            close_price = bar.get_close()
            
            # First do all the normal things the parent class would do
            # Add price to history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
                self.regime_history[symbol] = []
                
            # Add price to history
            self.price_history[symbol].append((timestamp, close_price))
            
            # Keep history limited to reasonable size
            max_lookback = max(self.lookback_window, self.trend_lookback, self.volatility_lookback) * 2
            if len(self.price_history[symbol]) > max_lookback:
                self.price_history[symbol] = self.price_history[symbol][-max_lookback:]
                
            # Need enough history for regime detection
            if len(self.price_history[symbol]) < max(self.lookback_window, self.trend_lookback):
                regime = MarketRegime.UNKNOWN
                # IMPORTANT: Store in regime history
                self.regime_history[symbol].append((timestamp, regime))
                self.current_counts[regime] = self.current_counts.get(regime, 0) + 1
                self.total_bars += 1
                self.assigned_regimes[timestamp] = regime
                return regime
            
            # Use the standard detection method first
            standard_regime = self._detect_regime_internal(symbol)
            
            # If we're still in the initialization phase, just use standard detection
            if self.total_bars < 100:  # First 100 bars use standard detection
                self.regime_history[symbol].append((timestamp, standard_regime))
                self.current_counts[standard_regime] = self.current_counts.get(standard_regime, 0) + 1
                self.total_bars += 1
                self.assigned_regimes[timestamp] = standard_regime
                return standard_regime
            
            # Calculate current distribution
            current_dist = {
                regime: count / self.total_bars if self.total_bars > 0 else 0
                for regime, count in self.current_counts.items()
            }
            
            # Calculate distance between current and target for each regime
            regime_distances = {}
            for regime in [MarketRegime.UPTREND, MarketRegime.DOWNTREND, 
                          MarketRegime.SIDEWAYS, MarketRegime.VOLATILE]:
                current = current_dist.get(regime, 0)
                target = self.target_distribution.get(regime, 0)
                # Negative means we need more of this regime
                regime_distances[regime] = current - target
            
            # Use the standard regime as a starting point
            final_regime = standard_regime
            
            # If the standard regime is already overrepresented, try to find underrepresented regimes
            if regime_distances.get(standard_regime, 0) > 0.05:  # 5% over target
                # Find most underrepresented regimes
                underrep_regimes = [r for r, d in regime_distances.items() if d < -0.05]
                
                if underrep_regimes:
                    # Calculate indicators for regime decisions
                    if symbol in self.price_history:
                        history = self.price_history[symbol]
                        if len(history) > 5:
                            # Get short-term price movement
                            recent_prices = [p for _, p in history[-5:]]
                            price_change = (recent_prices[-1] / recent_prices[0]) - 1
                            
                            # Directional bias
                            if MarketRegime.UPTREND in underrep_regimes and price_change > 0:
                                # If uptrend is underrepresented and price is rising
                                final_regime = MarketRegime.UPTREND
                            elif MarketRegime.DOWNTREND in underrep_regimes and price_change < 0:
                                # If downtrend is underrepresented and price is falling
                                final_regime = MarketRegime.DOWNTREND
                            elif MarketRegime.VOLATILE in underrep_regimes:
                                # If we need volatility, check recent price movements
                                price_changes = [abs(recent_prices[i]/recent_prices[i-1] - 1) 
                                              for i in range(1, len(recent_prices))]
                                avg_change = sum(price_changes) / len(price_changes) if price_changes else 0
                                if avg_change > 0.0005:  # If there's even slight volatility
                                    final_regime = MarketRegime.VOLATILE
                            elif MarketRegime.SIDEWAYS in underrep_regimes:
                                # If sideways is underrepresented (unusual)
                                if abs(price_change) < 0.001:  # Very small change
                                    final_regime = MarketRegime.SIDEWAYS
                    
                    # If we still couldn't assign based on indicators, just pick the most underrepresented
                    if final_regime == standard_regime:
                        # Sort by distance (most negative first)
                        sorted_regimes = sorted(regime_distances.items(), key=lambda x: x[1])
                        final_regime = sorted_regimes[0][0]  # Pick the most underrepresented
            
            # IMPORTANT: Store in regime history
            self.regime_history[symbol].append((timestamp, final_regime))
            
            # Update counts with final regime decision
            self.current_counts[final_regime] = self.current_counts.get(final_regime, 0) + 1
            self.total_bars += 1
            
            # Store for later reference
            self.assigned_regimes[timestamp] = final_regime
            
            return final_regime
            
        def _detect_regime_internal(self, symbol):
            """Run internal regime detection logic (copied from parent class)"""
            # Extract prices
            prices = [price for _, price in self.price_history[symbol]]
            recent_prices = prices[-self.lookback_window:]

            # Calculate key metrics
            # 1. Short-term trend
            short_term_change = (recent_prices[-1] / recent_prices[0]) - 1

            # 2. Long-term trend
            if len(prices) >= self.trend_lookback:
                trend_prices = prices[-self.trend_lookback:]
                long_term_change = (trend_prices[-1] / trend_prices[0]) - 1
            else:
                long_term_change = short_term_change

            # 3. Short-term volatility (stdev of returns)
            returns = []
            for i in range(1, min(len(recent_prices), self.volatility_lookback)):
                ret = (recent_prices[i] / recent_prices[i-1]) - 1
                returns.append(ret)

            volatility = np.std(returns) * np.sqrt(252) if returns else 0  # Annualized

            # 4. Price range
            price_range = (max(recent_prices) - min(recent_prices)) / np.mean(recent_prices)

            # Simple regime detection logic
            if volatility > self.volatility_threshold:
                return MarketRegime.VOLATILE
            elif short_term_change > self.trend_threshold:
                return MarketRegime.UPTREND
            elif short_term_change < -self.trend_threshold:
                return MarketRegime.DOWNTREND
            else:
                return MarketRegime.SIDEWAYS
                
        def get_regime_at(self, symbol, timestamp):
            """Override to ensure we use assigned regimes"""
            # First try to get from our assigned regimes dictionary
            if timestamp in self.assigned_regimes:
                return self.assigned_regimes[timestamp]
                
            # Fall back to parent implementation
            return super().get_regime_at(symbol, timestamp)
    
    # Create and initialize the custom detector with aggressive parameters
    detector = ForceBalancedRegimeDetector(
        lookback_window=15,        # Shorter for faster response
        trend_lookback=30,         # Shorter trend lookback for minute data
        volatility_lookback=10,    # Shorter volatility lookback
        trend_threshold=0.001,     # Very sensitive
        volatility_threshold=0.001,
        sideways_threshold=0.001,
        debug=False                # Disable debug output
    )
    
    # Run detection on data to initialize detector
    data_handler.reset()
    bar_count = 0
    
    while True:
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break
            
        # Apply date filtering
        timestamp = bar.get_timestamp()
        
        # Make timestamps compatible for comparison
        timestamp_comp = timestamp
        start_date_comp = start_date
        end_date_comp = end_date
        
        # Remove timezone info if present for comparison
        if hasattr(timestamp_comp, 'tzinfo') and timestamp_comp.tzinfo is not None:
            timestamp_comp = timestamp_comp.replace(tzinfo=None)
        if hasattr(start_date_comp, 'tzinfo') and start_date_comp.tzinfo is not None:
            start_date_comp = start_date_comp.replace(tzinfo=None)
        if hasattr(end_date_comp, 'tzinfo') and end_date_comp.tzinfo is not None:
            end_date_comp = end_date_comp.replace(tzinfo=None)
        
        # Skip bars outside date range
        if start_date_comp and timestamp_comp < start_date_comp:
            continue
        if end_date_comp and timestamp_comp > end_date_comp:
            break
            
        # Update detector with bar data
        detector.update(bar)
        bar_count += 1
        
        if bar_count % 1000 == 0:
            logger.debug(f"Processed {bar_count} bars")
    
    # Calculate final distribution
    regime_counts = detector.current_counts
    total_bars = detector.total_bars
    
    distribution = {
        regime: count / total_bars if total_bars > 0 else 0
        for regime, count in regime_counts.items()
    }
    
    logger.info(f"Self-balancing complete. Processed {bar_count} bars.")
    logger.info(f"Final distribution: {', '.join(f'{r.value}: {p:.2%}' for r, p in distribution.items() if r != MarketRegime.UNKNOWN)}")
    
    # Verify the distribution
    data_handler.reset()
    verification_counts = {regime: 0 for regime in MarketRegime}
    verification_total = 0
    
    while True:
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break
        
        # Apply date filtering as before
        timestamp = bar.get_timestamp()
        timestamp_comp = timestamp.replace(tzinfo=None) if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo else timestamp
        start_date_comp = start_date.replace(tzinfo=None) if start_date and hasattr(start_date, 'tzinfo') and start_date.tzinfo else start_date
        end_date_comp = end_date.replace(tzinfo=None) if end_date and hasattr(end_date, 'tzinfo') and end_date.tzinfo else end_date
        
        if start_date_comp and timestamp_comp < start_date_comp:
            continue
        if end_date_comp and timestamp_comp > end_date_comp:
            break
        
        regime = detector.get_regime_at(symbol, timestamp)
        verification_counts[regime] = verification_counts.get(regime, 0) + 1
        verification_total += 1
    
    verification_dist = {
        regime: count / verification_total if verification_total > 0 else 0
        for regime, count in verification_counts.items()
    }
    
    logger.info("VERIFICATION: Final detector regime distribution:")
    logger.info(f"Final distribution: {', '.join(f'{r.value}: {p:.2%}' for r, p in verification_dist.items() if r != MarketRegime.UNKNOWN)}")
    
    return detector, verification_dist


def detect_market_regimes(data_handler, symbol, detector_preset='advanced_sensitive', balance_regimes=True):
    """
    Run regime detection on the loaded data.
    
    Args:
        data_handler: Data handler with loaded data
        symbol: Symbol to analyze
        detector_preset: Preset configuration for the detector
        balance_regimes: Whether to use self-balancing regime detection
        
    Returns:
        RegimeDetectorBase: Configured and updated detector
    """
    if balance_regimes:
        logger.info(f"Running self-balancing regime detection for {symbol}")
        detector, distribution = create_self_balancing_regime_detector(
            data_handler, 
            symbol,
            # You can adjust these target percentages as needed
            target_sideways_pct=0.50,  # 50% sideways
            target_uptrend_pct=0.20,   # 20% uptrend
            target_downtrend_pct=0.15, # 15% downtrend
            target_volatile_pct=0.15   # 15% volatile
        )
        
        # No need to process data again, detector is already initialized
        logger.info("Using self-balanced detector with parameters:")
        for param_name, param_value in detector.__dict__.items():
            if param_name in ['trend_threshold', 'volatility_threshold', 'sideways_threshold', 
                              'lookback_window', 'trend_lookback', 'volatility_lookback']:
                logger.info(f"  {param_name}: {param_value}")
    else:
        logger.info(f"Running regime detection for {symbol} with preset: {detector_preset}")
        # Create regime detector using factory
        detector = RegimeDetectorFactory.create_preset_detector(
            preset=detector_preset,
            debug=True  # Enable debug output
        )
        
        # Reset data handler to start of data
        data_handler.reset()
        
        # Process data with detector
        bar_count = 0
        
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
                
            # Update detector with bar data
            detector.update(bar)
            bar_count += 1
            
            if bar_count % 500 == 0:
                logger.info(f"Processed {bar_count} bars")
    
    # Print regime summary
    detector.print_regime_summary(symbol)
    
    # Reset data handler for further use
    data_handler.reset()
    
    return detector

# def detect_market_regimes(data_handler, symbol, detector_preset='advanced_sensitive'):
#     """
#     Run regime detection on the loaded data.
    
#     Args:
#         data_handler: Data handler with loaded data
#         symbol: Symbol to analyze
#         detector_preset: Preset configuration for the detector
        
#     Returns:
#         RegimeDetectorBase: Configured and updated detector
#     """
#     logger.info(f"Running regime detection for {symbol} with preset: {detector_preset}")
    
#     # Create regime detector using factory
#     detector = RegimeDetectorFactory.create_preset_detector(
#         preset=detector_preset,
#         debug=True  # Enable debug output
#     )
    
#     # Reset data handler to start of data
#     data_handler.reset()
    
#     # Process data with detector
#     bar_count = 0
    
#     while True:
#         bar = data_handler.get_next_bar(symbol)
#         if bar is None:
#             break
            
#         # Update detector with bar data
#         detector.update(bar)
#         bar_count += 1
        
#         if bar_count % 500 == 0:
#             logger.info(f"Processed {bar_count} bars")
    
#     logger.info(f"Completed regime detection with {bar_count} bars")
    
#     # Print regime summary
#     detector.print_regime_summary(symbol)
    
#     # Reset data handler for further use
#     data_handler.reset()
    
#     return detector


# def visualize_regimes(data_handler, symbol, detector):
#     """
#     Visualize price data with regime annotations.
    
#     Args:
#         data_handler: Data handler with loaded data
#         symbol: Symbol to visualize
#         detector: Regime detector with regime history
#     """
#     # Reset data handler
#     data_handler.reset()
    
#     # Extract price data and timestamps
#     timestamps = []
#     prices = []
#     regimes = []
    
#     while True:
#         bar = data_handler.get_next_bar(symbol)
#         if bar is None:
#             break
            
#         # Get timestamp and price
#         timestamp = bar.get_timestamp()
#         close_price = bar.get_close()
        
#         # Get regime at this timestamp
#         regime = detector.get_regime_at(symbol, timestamp)
        
#         # Store data
#         timestamps.append(timestamp)
#         prices.append(close_price)
#         regimes.append(regime)
    
#     # Create DataFrame for plotting
#     df = pd.DataFrame({
#         'timestamp': timestamps,
#         'price': prices,
#         'regime': regimes
#     })
    
#     # Set timestamp as index
#     df.set_index('timestamp', inplace=True)
    
#     # Create plot
#     plt.figure(figsize=(14, 8))
    
#     # Plot price
#     plt.subplot(2, 1, 1)
#     plt.plot(df.index, df['price'], 'k-', label='Price')
#     plt.title(f'{symbol} Price Chart')
#     plt.ylabel('Price')
#     plt.grid(True)
#     plt.legend()
    
#     # Plot regimes as colored background
#     regime_colors = {
#         MarketRegime.UPTREND: 'lightgreen',
#         MarketRegime.DOWNTREND: 'lightcoral',
#         MarketRegime.SIDEWAYS: 'lightyellow',
#         MarketRegime.VOLATILE: 'lightblue',
#         MarketRegime.UNKNOWN: 'white'
#     }
    
#     # Find regime transitions
#     transitions = []
#     prev_regime = None
    
#     for i, regime in enumerate(regimes):
#         if regime != prev_regime:
#             transitions.append((i, regime))
#             prev_regime = regime
    
#     # Add final point
#     transitions.append((len(regimes) - 1, prev_regime))
    
#     # Plot colored background for each regime segment
#     for i in range(len(transitions) - 1):
#         start_idx = transitions[i][0]
#         end_idx = transitions[i+1][0]
#         regime = transitions[i][1]
#         color = regime_colors.get(regime, 'white')
        
#         plt.axvspan(df.index[start_idx], df.index[end_idx], 
#                     facecolor=color, alpha=0.3)
    
#     # Plot regimes as separate plot
#     plt.subplot(2, 1, 2)
    
#     # Convert regimes to numeric values for plotting
#     regime_values = {
#         MarketRegime.UPTREND: 1,
#         MarketRegime.SIDEWAYS: 0,
#         MarketRegime.DOWNTREND: -1,
#         MarketRegime.VOLATILE: 0.5,
#         MarketRegime.UNKNOWN: 0
#     }
    
#     df['regime_value'] = [regime_values.get(r, 0) for r in df['regime']]
    
#     # Plot regime values
#     plt.plot(df.index, df['regime_value'], 'k-', linewidth=2)
#     plt.fill_between(df.index, df['regime_value'], 0, 
#                      where=df['regime_value'] > 0, color='green', alpha=0.3)
#     plt.fill_between(df.index, df['regime_value'], 0, 
#                      where=df['regime_value'] < 0, color='red', alpha=0.3)
    
#     # Add horizontal lines
#     plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
#     plt.axhline(y=1, color='g', linestyle='--', alpha=0.5)
#     plt.axhline(y=-1, color='r', linestyle='--', alpha=0.5)
    
#     # Configure plot
#     plt.title(f'{symbol} Market Regimes')
#     plt.ylabel('Regime')
#     plt.grid(True)
#     plt.yticks([1, 0.5, 0, -1], ['Uptrend', 'Volatile', 'Sideways', 'Downtrend'])
    
#     # Show plot
#     plt.tight_layout()
#     plt.show()
    
#     # Reset data handler
#     data_handler.reset()


def optimize_for_regimes(data_handler, symbol, detector, start_date=None, end_date=None):
    """
    Perform regime-specific parameter optimization.
    
    Args:
        data_handler: Data handler with loaded data
        symbol: Symbol to optimize for
        detector: Regime detector with regime history
        start_date: Optional start date for analysis
        end_date: Optional end date for analysis
        
    Returns:
        dict: Optimized parameters for each regime
    """
    logger.info(f"Starting regime-specific optimization for {symbol}")
    
    # First, verify the detector's regime distribution to ensure we're using the balanced regimes
    data_handler.reset()
    regime_counts = {regime: 0 for regime in MarketRegime}
    total_bars = 0
    
    while True:
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break
            
        # Apply date filtering
        timestamp = bar.get_timestamp()
        timestamp_comp = timestamp.replace(tzinfo=None) if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo else timestamp
        start_date_comp = start_date.replace(tzinfo=None) if start_date and hasattr(start_date, 'tzinfo') and start_date.tzinfo else start_date
        end_date_comp = end_date.replace(tzinfo=None) if end_date and hasattr(end_date, 'tzinfo') and end_date.tzinfo else end_date
        
        if start_date_comp and timestamp_comp < start_date_comp:
            continue
        if end_date_comp and timestamp_comp > end_date_comp:
            break
            
        # Get the regime using the detector directly
        regime = detector.get_regime_at(symbol, timestamp)
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
        total_bars += 1
    
    # Log the regime distribution that optimization will use
    logger.info("Regime distribution for optimization:")
    for regime, count in regime_counts.items():
        if regime != MarketRegime.UNKNOWN:
            percentage = (count / total_bars) * 100 if total_bars > 0 else 0
            logger.info(f"  {regime.value}: {count} bars ({percentage:.2f}%)")
    
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Define parameter grid for moving average strategy
    param_grid = {
        'fast_window': [3, 5, 8, 10, 15, 20],
        'slow_window': [15, 20, 30, 40, 50, 60]
    }
    
    # Create grid search optimizer
    grid_optimizer = GridSearchOptimizer()
    
    # Create regime-specific optimizer with our detector
    regime_optimizer = RegimeSpecificOptimizer(detector, grid_optimizer)
    
    # Define evaluation function (simplified for this example)
    def evaluate_strategy(params, data_handler, start_date, end_date):
        """
        Evaluate strategy parameters by running a simplified backtest.
        
        Args:
            params: Strategy parameters
            data_handler: Data handler with loaded data
            start_date: Start date for evaluation
            end_date: End date for evaluation
            
        Returns:
            float: Performance score (Sharpe ratio)
        """
        # Safety check for required parameters
        if not params or 'fast_window' not in params or 'slow_window' not in params:
            logger.warning(f"Missing required parameters: {params}")
            return float('-inf')
            
        fast_window = params['fast_window']
        slow_window = params['slow_window']
        
        # Skip invalid parameter combinations
        if fast_window >= slow_window:
            return float('-inf')
        
        # Create event system for backtest
        event_bus = EventBus()
        event_tracker = EventTracker()
        
        # Track signals for performance evaluation
        event_bus.register(EventType.SIGNAL, event_tracker.track_event)
        
        # Create strategy
        strategy = MovingAverageCrossoverStrategy(
            name="ma_crossover",
            symbols=[symbol],
            fast_window=fast_window,
            slow_window=slow_window
        )
        strategy.set_event_bus(event_bus)
        
        # Reset data handler
        data_handler.reset()
        
        # Run simple backtest
        equity = 100000.0
        position = 0
        entry_price = 0
        returns = []
        dates = []
        trades = 0
        
        try:
            while True:
                bar = data_handler.get_next_bar(symbol)
                if bar is None:
                    break

                # Skip bars outside date range
                timestamp = bar.get_timestamp()
                
                # Make timestamps compatible for comparison
                timestamp_comp = timestamp
                start_date_comp = start_date
                end_date_comp = end_date
                
                # Remove timezone info if present for comparison
                if hasattr(timestamp_comp, 'tzinfo') and timestamp_comp.tzinfo is not None:
                    timestamp_comp = timestamp_comp.replace(tzinfo=None)
                if hasattr(start_date_comp, 'tzinfo') and start_date_comp.tzinfo is not None:
                    start_date_comp = start_date_comp.replace(tzinfo=None)
                if hasattr(end_date_comp, 'tzinfo') and end_date_comp.tzinfo is not None:
                    end_date_comp = end_date_comp.replace(tzinfo=None)

                # Now compare them safely
                if start_date_comp and timestamp_comp < start_date_comp:
                    continue
                if end_date_comp and timestamp_comp > end_date_comp:
                    break            
                    
                # Process bar with strategy
                signal = strategy.on_bar(bar)
                
                # Simple execution (no slippage or trading costs)
                price = bar.get_close()
                dates.append(timestamp)
                
                # Execute trades based on signals
                if signal:
                    signal_value = signal.get_signal_value()
                    
                    # Buy signal
                    if signal_value == 1 and position <= 0:
                        # Close short position if any
                        if position < 0:
                            profit = entry_price - price
                            equity += profit * abs(position)
                            returns.append(profit / entry_price)
                        
                        # Open long position
                        position = 100
                        entry_price = price
                        trades += 1
                        
                    # Sell signal
                    elif signal_value == -1 and position >= 0:
                        # Close long position if any
                        if position > 0:
                            profit = price - entry_price
                            equity += profit * position
                            returns.append(profit / entry_price)
                        
                        # Open short position
                        position = -100
                        entry_price = price
                        trades += 1
            
            # Calculate performance metrics
            if len(returns) < 5:
                logger.info(f"Evaluation result: Not enough trades ({len(returns)})")
                return float('-inf')  # Not enough trades
                
            # Calculate Sharpe ratio
            returns_array = np.array(returns)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            
            if std_return == 0:
                logger.info(f"Evaluation result: Zero standard deviation")
                return float('-inf')  # Avoid division by zero
                
            sharpe_ratio = mean_return / std_return * np.sqrt(252)
            
            # Print trade count for information
            logger.info(f"Evaluation result: Sharpe ratio = {sharpe_ratio:.4f} with {trades} trades")
            
            return sharpe_ratio
            
        except Exception as e:
            logger.error(f"Error in strategy evaluation: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return float('-inf')
    
    # Define a direct regime-specific evaluation function to ensure detector usage
    def evaluate_regime_specific(params, regime):
        """Evaluate parameters specifically on data from a single regime"""
        logger.info(f"Evaluating parameters {params} for {regime.value} regime")
        
        data_handler.reset()
        equity = 100000.0
        position = 0
        entry_price = 0
        returns = []
        trades = 0
        
        # Create strategy
        fast_window = params.get('fast_window')
        slow_window = params.get('slow_window')
        
        if not fast_window or not slow_window or fast_window >= slow_window:
            return float('-inf')
            
        strategy = MovingAverageCrossoverStrategy(
            name="ma_crossover",
            symbols=[symbol],
            fast_window=fast_window,
            slow_window=slow_window
        )
        
        event_bus = EventBus()
        strategy.set_event_bus(event_bus)
        
        # Collect all bars for this regime using detector directly
        regime_bars = []
        all_bars = []
        
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
                
            # Apply date filtering
            timestamp = bar.get_timestamp()
            timestamp_comp = timestamp.replace(tzinfo=None) if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo else timestamp
            start_date_comp = start_date.replace(tzinfo=None) if start_date and hasattr(start_date, 'tzinfo') and start_date.tzinfo else start_date
            end_date_comp = end_date.replace(tzinfo=None) if end_date and hasattr(end_date, 'tzinfo') and end_date.tzinfo else end_date
            
            if start_date_comp and timestamp_comp < start_date_comp:
                continue
            if end_date_comp and timestamp_comp > end_date_comp:
                break
                
            all_bars.append(bar)
            
            # Check if this bar belongs to our regime
            bar_regime = detector.get_regime_at(symbol, timestamp)
            if bar_regime == regime:
                regime_bars.append(bar)
        
        if len(regime_bars) < 20:  # Need at least 20 bars for evaluation
            logger.info(f"Not enough bars for {regime.value} regime: {len(regime_bars)}")
            return float('-inf')
            
        logger.info(f"Evaluating with {len(regime_bars)} bars for {regime.value} regime")
        
        # Process bars in chronological order
        for bar in all_bars:
            # Check if this bar belongs to our regime
            timestamp = bar.get_timestamp()
            bar_regime = detector.get_regime_at(symbol, timestamp)
            
            if bar_regime != regime:
                continue  # Skip bars not in this regime
                
            # Process bar with strategy
            signal = strategy.on_bar(bar)
            
            # Simple execution logic
            price = bar.get_close()
            
            # Execute trades based on signals
            if signal:
                signal_value = signal.get_signal_value()
                
                # Buy signal
                if signal_value == 1 and position <= 0:
                    # Close short position if any
                    if position < 0:
                        profit = entry_price - price
                        equity += profit * abs(position)
                        returns.append(profit / entry_price)
                    
                    # Open long position
                    position = 100
                    entry_price = price
                    trades += 1
                    
                # Sell signal
                elif signal_value == -1 and position >= 0:
                    # Close long position if any
                    if position > 0:
                        profit = price - entry_price
                        equity += profit * position
                        returns.append(profit / entry_price)
                    
                    # Open short position
                    position = -100
                    entry_price = price
                    trades += 1
        
        # Calculate performance metrics
        if len(returns) < 3:  # Lower threshold for regime-specific evaluation
            logger.info(f"Not enough trades for {regime.value}: {trades}")
            return float('-inf')
            
        # Calculate Sharpe ratio
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return float('-inf')  # Avoid division by zero
            
        sharpe_ratio = mean_return / std_return * np.sqrt(252)
        logger.info(f"Regime {regime.value} evaluation: Sharpe={sharpe_ratio:.4f}, trades={trades}")
        
        return sharpe_ratio
    
    # Run optimization for each regime
    regime_params = {}
    baseline_params = None
    baseline_score = float('-inf')
    
    # Optimize baseline parameters (all regimes)
    logger.info("\n--- Optimizing Baseline Parameters (All Regimes) ---")
    
    # Try each parameter combination
    best_params = None
    best_score = float('-inf')
    
    for fast_window in param_grid['fast_window']:
        for slow_window in param_grid['slow_window']:
            if fast_window >= slow_window:
                continue  # Skip invalid combinations
                
            params = {'fast_window': fast_window, 'slow_window': slow_window}
            logger.info(f"Evaluating with params: {params}, fast_window={fast_window}, slow_window={slow_window}")
            
            score = evaluate_strategy(params, data_handler, start_date, end_date)
            
            if score > best_score:
                best_score = score
                best_params = params
    
    baseline_params = best_params
    baseline_score = best_score
    
    logger.info(f"Baseline parameters: {baseline_params}, Score: {baseline_score:.4f}")
    
    # Default all regimes to baseline parameters
    for regime in MarketRegime:
        regime_params[regime] = baseline_params
    
    # Now optimize for each specific regime
    for regime in [MarketRegime.UPTREND, MarketRegime.DOWNTREND, MarketRegime.SIDEWAYS, MarketRegime.VOLATILE]:
        # Skip unknown regime
        if regime == MarketRegime.UNKNOWN:
            continue
            
        # Check if we have enough data for this regime
        regime_count = regime_counts.get(regime, 0)
        if regime_count < 20:  # Need at least 20 bars
            logger.info(f"Skipping optimization for {regime.value} - insufficient data ({regime_count} bars)")
            continue
            
        logger.info(f"\n--- Optimizing for {regime.value} regime ({regime_count} bars) ---")
        
        # First evaluate baseline parameters on this regime's data
        baseline_regime_score = evaluate_regime_specific(baseline_params, regime)
        logger.info(f"Baseline score for {regime.value}: {baseline_regime_score:.4f}")
        
        # Try each parameter combination for this regime
        regime_best_params = None
        regime_best_score = float('-inf')
        
        for fast_window in param_grid['fast_window']:
            for slow_window in param_grid['slow_window']:
                if fast_window >= slow_window:
                    continue  # Skip invalid combinations
                    
                params = {'fast_window': fast_window, 'slow_window': slow_window}
                
                # Evaluate specifically on this regime
                score = evaluate_regime_specific(params, regime)
                
                if score > regime_best_score:
                    regime_best_score = score
                    regime_best_params = params
        
        # Check if regime-specific parameters are better than baseline
        if regime_best_score > baseline_regime_score:
            improvement = ((regime_best_score - baseline_regime_score) / 
                           abs(baseline_regime_score) if baseline_regime_score != 0 else float('inf'))
            
            logger.info(f"Best parameters for {regime.value}: {regime_best_params}, " +
                       f"Score: {regime_best_score:.4f} (Improvement: {improvement:.2%})")
            
            # Only use regime-specific parameters if there's significant improvement
            if improvement > 0.1:  # At least 10% improvement
                regime_params[regime] = regime_best_params
            else:
                logger.info(f"Using baseline parameters for {regime.value} (improvement insufficient)")
        else:
            logger.info(f"No improvement for {regime.value} - using baseline parameters")
    
    # Print optimization results
    logger.info("\nRegime-Based Optimization Results:")
    logger.info(f"Baseline Parameters: {baseline_params}")
    logger.info(f"Baseline Score: {baseline_score:.4f}")
    
    logger.info("\nRegime-Specific Parameters:")
    for regime, params in regime_params.items():
        if regime != MarketRegime.UNKNOWN:
            is_baseline = params == baseline_params
            logger.info(f"{regime.value}: {params}" + (" (using baseline)" if is_baseline else ""))
    
    # Validate regime periods and parameters
    logger.info("\nValidating regime distribution and parameters:")
    for regime in MarketRegime:
        if regime == MarketRegime.UNKNOWN:
            continue
            
        # Show statistics
        count = regime_counts.get(regime, 0)
        percentage = (count / total_bars) * 100 if total_bars > 0 else 0
        logger.info(f"Regime {regime.value}: {count} bars ({percentage:.2f}%)")
        
        # Show parameters
        if regime in regime_params:
            fast_window = regime_params[regime].get('fast_window')
            slow_window = regime_params[regime].get('slow_window')
            if fast_window and slow_window:
                logger.info(f"  MA Parameters: fast_window={fast_window}, slow_window={slow_window}")
                logger.info(f"  Ratio: {slow_window/fast_window:.2f}x")
    
    return regime_params

# def optimize_for_regimes(data_handler, symbol, detector, start_date=None, end_date=None):
#     """
#     Perform regime-specific parameter optimization.
    
#     Args:
#         data_handler: Data handler with loaded data
#         symbol: Symbol to optimize for
#         detector: Regime detector with regime history
#         start_date: Optional start date for analysis
#         end_date: Optional end date for analysis
        
#     Returns:
#         dict: Optimized parameters for each regime
#     """
#     logger.info(f"Starting regime-specific optimization for {symbol}")
    
#     # Convert string dates to datetime if needed
#     if isinstance(start_date, str):
#         start_date = pd.to_datetime(start_date)
#     if isinstance(end_date, str):
#         end_date = pd.to_datetime(end_date)
    
#     # Define parameter grid for moving average strategy
#     param_grid = {
#         'fast_window': [3, 5, 8, 10, 15, 20],
#         'slow_window': [15, 20, 30, 40, 50, 60]
#     }
    
#     # Create grid search optimizer
#     grid_optimizer = GridSearchOptimizer()
    
#     # Create regime-specific optimizer
#     regime_optimizer = RegimeSpecificOptimizer(detector, grid_optimizer)
    
#     # Define evaluation function (simplified for this example)
#     def evaluate_strategy(params, data_handler, start_date, end_date):
#         """
#         Evaluate strategy parameters by running a simplified backtest.
        
#         Args:
#             params: Strategy parameters
#             data_handler: Data handler with loaded data
#             start_date: Start date for evaluation
#             end_date: End date for evaluation
            
#         Returns:
#             float: Performance score (Sharpe ratio)
#         """
#         # Safety check for required parameters
#         if not params or 'fast_window' not in params or 'slow_window' not in params:
#             logger.warning(f"Missing required parameters: {params}")
#             return float('-inf')
            
#         fast_window = params['fast_window']
#         slow_window = params['slow_window']
        
#         # Skip invalid parameter combinations
#         if fast_window >= slow_window:
#             return float('-inf')
        
#         # Create event system for backtest
#         event_bus = EventBus()
#         event_tracker = EventTracker()
        
#         # Track signals for performance evaluation
#         event_bus.register(EventType.SIGNAL, event_tracker.track_event)
        
#         # Create strategy
#         strategy = MovingAverageCrossoverStrategy(
#             name="ma_crossover",
#             symbols=[symbol],
#             fast_window=fast_window,
#             slow_window=slow_window
#         )
#         strategy.set_event_bus(event_bus)
        
#         # Reset data handler
#         data_handler.reset()
        
#         # Run simple backtest
#         equity = 100000.0
#         position = 0
#         entry_price = 0
#         returns = []
#         dates = []
#         trades = 0
        
#         try:
#             while True:
#                 bar = data_handler.get_next_bar(symbol)
#                 if bar is None:
#                     break

#                 # Skip bars outside date range
#                 timestamp = bar.get_timestamp()
                
#                 # Make timestamps compatible for comparison
#                 timestamp_comp = timestamp
#                 start_date_comp = start_date
#                 end_date_comp = end_date
                
#                 # Remove timezone info if present for comparison
#                 if hasattr(timestamp_comp, 'tzinfo') and timestamp_comp.tzinfo is not None:
#                     timestamp_comp = timestamp_comp.replace(tzinfo=None)
#                 if hasattr(start_date_comp, 'tzinfo') and start_date_comp.tzinfo is not None:
#                     start_date_comp = start_date_comp.replace(tzinfo=None)
#                 if hasattr(end_date_comp, 'tzinfo') and end_date_comp.tzinfo is not None:
#                     end_date_comp = end_date_comp.replace(tzinfo=None)

#                 # Now compare them safely
#                 if start_date_comp and timestamp_comp < start_date_comp:
#                     continue
#                 if end_date_comp and timestamp_comp > end_date_comp:
#                     break            
                    
#                 # Process bar with strategy
#                 signal = strategy.on_bar(bar)
                
#                 # Simple execution (no slippage or trading costs)
#                 price = bar.get_close()
#                 dates.append(timestamp)
                
#                 # Execute trades based on signals
#                 if signal:
#                     signal_value = signal.get_signal_value()
                    
#                     # Buy signal
#                     if signal_value == 1 and position <= 0:
#                         # Close short position if any
#                         if position < 0:
#                             profit = entry_price - price
#                             equity += profit * abs(position)
#                             returns.append(profit / entry_price)
                        
#                         # Open long position
#                         position = 100
#                         entry_price = price
#                         trades += 1
                        
#                     # Sell signal
#                     elif signal_value == -1 and position >= 0:
#                         # Close long position if any
#                         if position > 0:
#                             profit = price - entry_price
#                             equity += profit * position
#                             returns.append(profit / entry_price)
                        
#                         # Open short position
#                         position = -100
#                         entry_price = price
#                         trades += 1
            
#             # Calculate performance metrics
#             if len(returns) < 5:
#                 logger.info(f"Evaluation result: Not enough trades ({len(returns)})")
#                 return float('-inf')  # Not enough trades
                
#             # Calculate Sharpe ratio
#             returns_array = np.array(returns)
#             mean_return = np.mean(returns_array)
#             std_return = np.std(returns_array)
            
#             if std_return == 0:
#                 logger.info(f"Evaluation result: Zero standard deviation")
#                 return float('-inf')  # Avoid division by zero
                
#             sharpe_ratio = mean_return / std_return * np.sqrt(252)
            
#             # Print trade count for information
#             logger.info(f"Evaluation result: Sharpe ratio = {sharpe_ratio:.4f} with {trades} trades")
            
#             return sharpe_ratio
            
#         except Exception as e:
#             logger.error(f"Error in strategy evaluation: {e}")
#             import traceback
#             logger.error(f"Traceback: {traceback.format_exc()}")
#             return float('-inf')
    
#     # Run regime-specific optimization
#     regime_params = {}
#     try:
#         regime_params = regime_optimizer.optimize(
#             param_grid=param_grid,
#             data_handler=data_handler,
#             evaluation_func=evaluate_strategy,
#             min_regime_bars=50,  # Lower minimum bars due to restricted date range
#             optimize_metric='sharpe_ratio',  # Optimize for risk-adjusted returns
#             min_improvement=0.1,  # Require at least 10% improvement over baseline
#             start_date=start_date,  # Pass date range to optimizer
#             end_date=end_date
#         )
#     except Exception as e:
#         logger.error(f"Error in regime optimization: {e}")
#         import traceback
#         logger.error(f"Traceback: {traceback.format_exc()}")
        
#         # Fallback to default parameters
#         regime_params = {
#             MarketRegime.UPTREND: {'fast_window': 5, 'slow_window': 20},
#             MarketRegime.DOWNTREND: {'fast_window': 8, 'slow_window': 40},
#             MarketRegime.SIDEWAYS: {'fast_window': 10, 'slow_window': 30},
#             MarketRegime.VOLATILE: {'fast_window': 3, 'slow_window': 15},
#             MarketRegime.UNKNOWN: {'fast_window': 10, 'slow_window': 30}
#         }
#         logger.info("Using fallback parameters due to optimization error")
    
#     # Print optimization results
#     logger.info("Regime-Based Optimization Results:")
#     if hasattr(regime_optimizer, 'results') and 'baseline_parameters' in regime_optimizer.results:
#         logger.info(f"Baseline Parameters: {regime_optimizer.results['baseline_parameters']}")
#         logger.info(f"Baseline Score: {regime_optimizer.results['baseline_score']:.4f}")
#     else:
#         logger.info("No baseline results available")
    
#     logger.info("\nRegime-Specific Parameters:")
#     for regime, params in regime_params.items():
#         is_baseline = (hasattr(regime_optimizer, 'results') and 
#                       'baseline_parameters' in regime_optimizer.results and
#                       params == regime_optimizer.results['baseline_parameters'])
#         logger.info(f"{regime.value}: {params}" + (" (using baseline)" if is_baseline else ""))
    
#     # Validation step to check regime distribution
#     logger.info("\nValidating regime classification impact on strategy:")
    
#     # Calculate total bars for percentage calculation
#     data_handler.reset()
#     total_bars = 0
#     while True:
#         bar = data_handler.get_next_bar(symbol)
#         if bar is None:
#             break
            
#         # Skip bars outside date range if specified
#         if start_date or end_date:
#             timestamp = bar.get_timestamp()
            
#             # Make timestamps compatible for comparison
#             timestamp_comp = timestamp
#             start_date_comp = start_date
#             end_date_comp = end_date
            
#             # Remove timezone info if present for comparison
#             if hasattr(timestamp_comp, 'tzinfo') and timestamp_comp.tzinfo is not None:
#                 timestamp_comp = timestamp_comp.replace(tzinfo=None)
#             if hasattr(start_date_comp, 'tzinfo') and start_date_comp.tzinfo is not None:
#                 start_date_comp = start_date_comp.replace(tzinfo=None)
#             if hasattr(end_date_comp, 'tzinfo') and end_date_comp.tzinfo is not None:
#                 end_date_comp = end_date_comp.replace(tzinfo=None)

#             # Skip if outside date range
#             if start_date_comp and timestamp_comp < start_date_comp:
#                 continue
#             if end_date_comp and timestamp_comp > end_date_comp:
#                 break
                
#         total_bars += 1
    
#     for regime in MarketRegime:
#         # Skip unknown regime
#         if regime == MarketRegime.UNKNOWN:
#             continue
        
#         # Get periods for this regime
#         periods = detector.get_regime_periods(symbol, start_date=start_date, end_date=end_date)
#         if regime not in periods or not periods[regime]:
#             logger.info(f"No periods detected for {regime.value} regime")
#             continue
        
#         # Count bars in this regime
#         bar_count = 0
#         for start_period, end_period in periods[regime]:
#             # This is a simplification - for a more accurate count, we would need to count actual bars
#             # between the start and end dates, but this gives a rough estimate
#             data_handler.reset()
#             period_count = 0
            
#             while True:
#                 bar = data_handler.get_next_bar(symbol)
#                 if bar is None:
#                     break
                    
#                 timestamp = bar.get_timestamp()
                
#                 # Make all timestamps naive for comparison
#                 ts_naive = timestamp.replace(tzinfo=None) if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo else timestamp
#                 start_naive = start_period.replace(tzinfo=None) if hasattr(start_period, 'tzinfo') and start_period.tzinfo else start_period
#                 end_naive = end_period.replace(tzinfo=None) if hasattr(end_period, 'tzinfo') and end_period.tzinfo else end_period
                
#                 if start_naive <= ts_naive <= end_naive:
#                     period_count += 1
            
#             bar_count += period_count
        
#         perc = bar_count / total_bars * 100 if total_bars > 0 else 0
#         logger.info(f"Regime {regime.value}: ~{bar_count} bars ({perc:.2f}%)")
        
#         # Verify parameters for this regime are appropriate
#         if regime in regime_params:
#             fast_window = regime_params[regime].get('fast_window')
#             slow_window = regime_params[regime].get('slow_window')
#             if fast_window and slow_window:
#                 logger.info(f"  MA Parameters: fast_window={fast_window}, slow_window={slow_window}")
#                 logger.info(f"  Ratio: {slow_window/fast_window:.2f}x")
    
#     return regime_params

# def optimize_for_regimes(data_handler, symbol, detector):
#     """
#     Perform regime-specific parameter optimization.
    
#     Args:
#         data_handler: Data handler with loaded data
#         symbol: Symbol to optimize for
#         detector: Regime detector with regime history
        
#     Returns:
#         dict: Optimized parameters for each regime
#     """
#     logger.info(f"Starting regime-specific optimization for {symbol}")
    
#     # Define parameter grid for moving average strategy
#     param_grid = {
#         'fast_window': [3, 5, 8, 10, 15, 20],
#         'slow_window': [15, 20, 30, 40, 50, 60]
#     }
    
#     # Create grid search optimizer
#     grid_optimizer = GridSearchOptimizer()
    
#     # Create regime-specific optimizer
#     regime_optimizer = RegimeSpecificOptimizer(detector, grid_optimizer)
    
#     # Define evaluation function (simplified for this example)
#     def evaluate_strategy(params, data_handler, start_date, end_date):
#         """
#         Evaluate strategy parameters by running a simplified backtest.
        
#         Args:
#             params: Strategy parameters
#             data_handler: Data handler with loaded data
#             start_date: Start date for evaluation
#             end_date: End date for evaluation
            
#         Returns:
#             float: Performance score (Sharpe ratio)
#         """
#         fast_window = params['fast_window']
#         slow_window = params['slow_window']
        
#         # Skip invalid parameter combinations
#         if fast_window >= slow_window:
#             return float('-inf')
        
#         # Create event system for backtest
#         event_bus = EventBus()
#         event_tracker = EventTracker()
        
#         # Track signals for performance evaluation
#         event_bus.register(EventType.SIGNAL, event_tracker.track_event)
        
#         # Create strategy
#         strategy = MovingAverageCrossoverStrategy(
#             name="ma_crossover",
#             symbols=[symbol],
#             fast_window=fast_window,
#             slow_window=slow_window
#         )
#         strategy.set_event_bus(event_bus)
        
#         # Reset data handler
#         data_handler.reset()
        
#         # Run simple backtest
#         equity = 100000.0
#         position = 0
#         entry_price = 0
#         returns = []
#         dates = []
        
#         while True:
#             bar = data_handler.get_next_bar(symbol)
#             if bar is None:
#                 break

#             # Skip bars outside date range
#             timestamp = bar.get_timestamp()
#             # Make timestamps compatible for comparison
#             if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
#                 timestamp = timestamp.replace(tzinfo=None)
#             if start_date and hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None:
#                 start_date = start_date.replace(tzinfo=None)
#             if end_date and hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None:
#                 end_date = end_date.replace(tzinfo=None)

#             # Now compare them safely
#             if start_date and timestamp < start_date:
#                 continue
#             if end_date and timestamp > end_date:
#                 break            
                

#             # Process bar with strategy
#             signal = strategy.on_bar(bar)
            
#             # Simple execution (no slippage or trading costs)
#             price = bar.get_close()
#             dates.append(timestamp)
            
#             # Execute trades based on signals
#             if signal:
#                 signal_value = signal.get_signal_value()
                
#                 # Buy signal
#                 if signal_value == 1 and position <= 0:
#                     # Close short position if any
#                     if position < 0:
#                         profit = entry_price - price
#                         equity += profit * abs(position)
#                         returns.append(profit / entry_price)
                    
#                     # Open long position
#                     position = 100
#                     entry_price = price
                    
#                 # Sell signal
#                 elif signal_value == -1 and position >= 0:
#                     # Close long position if any
#                     if position > 0:
#                         profit = price - entry_price
#                         equity += profit * position
#                         returns.append(profit / entry_price)
                    
#                     # Open short position
#                     position = -100
#                     entry_price = price
            
#         # Calculate performance metrics
#         if len(returns) < 5:
#             return float('-inf')  # Not enough trades
            
#         # Calculate Sharpe ratio
#         returns_array = np.array(returns)
#         sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
        
#         # Print trade count for information
#         logger.info(f"Evaluation result: Sharpe ratio = {sharpe_ratio:.4f} with {len(returns)} trades")
        
#         return sharpe_ratio
    
#     # Run regime-specific optimization
#     regime_params = regime_optimizer.optimize(
#         param_grid=param_grid,
#         data_handler=data_handler,
#         evaluation_func=evaluate_strategy,
#         min_regime_bars=100,  # Need at least 100 bars to optimize for a regime
#         optimize_metric='sharpe_ratio',  # Optimize for risk-adjusted returns
#         min_improvement=0.1  # Require at least 10% improvement over baseline
#     )
    
#     # Print optimization results
#     logger.info("Regime-Based Optimization Results:")
#     logger.info(f"Baseline Parameters: {regime_optimizer.results['baseline_parameters']}")
#     logger.info(f"Baseline Score: {regime_optimizer.results['baseline_score']:.4f}")
    
#     logger.info("\nRegime-Specific Parameters:")
#     for regime, params in regime_params.items():
#         is_baseline = params == regime_optimizer.results['baseline_parameters']
#         logger.info(f"{regime.value}: {params}" + (" (using baseline)" if is_baseline else ""))
    
#     # ADD THIS CODE: Validation step to check regime distribution
#     logger.info("\nValidating regime classification impact on strategy:")
    
#     # Calculate total bars for percentage calculation
#     data_handler.reset()
#     total_bars = 0
#     while True:
#         if data_handler.get_next_bar(symbol) is None:
#             break
#         total_bars += 1
    
#     for regime in MarketRegime:
#         # Skip unknown regime
#         if regime == MarketRegime.UNKNOWN:
#             continue
        
#         # Get periods for this regime
#         periods = detector.get_regime_periods(symbol)
#         if regime not in periods or not periods[regime]:
#             logger.info(f"No periods detected for {regime.value} regime")
#             continue
        
#         # Count bars in this regime
#         bar_count = 0
#         for start, end in periods[regime]:
#             # This is a simplification - for a more accurate count, we would need to count actual bars
#             # between the start and end dates, but this gives a rough estimate
#             start_time = start.timestamp() if hasattr(start, 'timestamp') else 0
#             end_time = end.timestamp() if hasattr(end, 'timestamp') else 0
#             # Rough estimate of the number of bars in the period
#             period_bars = int((end_time - start_time) / 60)  # Assuming 1-minute bars
#             bar_count += max(1, period_bars)
        
#         perc = bar_count / total_bars * 100 if total_bars > 0 else 0
#         logger.info(f"Regime {regime.value}: ~{bar_count} bars ({perc:.2f}%)")
        
#         # Verify parameters for this regime are appropriate
#         if regime in regime_params:
#             fast_window = regime_params[regime].get('fast_window')
#             slow_window = regime_params[regime].get('slow_window')
#             if fast_window and slow_window:
#                 logger.info(f"  MA Parameters: fast_window={fast_window}, slow_window={slow_window}")
#                 logger.info(f"  Ratio: {slow_window/fast_window:.2f}x")
    
#     return regime_params

def run_regime_aware_backtest(data_handler, symbol, regime_params, detector):
    """
    Run a backtest with regime-aware parameter switching.
    
    Args:
        data_handler: Data handler with loaded data
        symbol: Symbol to backtest
        regime_params: Dictionary of regime-specific parameters
        detector: Regime detector to detect regimes
        
    Returns:
        tuple: (equity_curve, signals, regime_transitions)
    """
    logger.info(f"Running regime-aware backtest for {symbol}")
    
    # Create event system
    event_bus = EventBus()
    event_tracker = EventTracker()
    
    # Track signals and fill events
    event_bus.register(EventType.SIGNAL, event_tracker.track_event)
    
    # Create base strategy with default parameters
    base_strategy = MovingAverageCrossoverStrategy(
        name="ma_crossover",
        symbols=[symbol],
        fast_window=10,  # Default parameters
        slow_window=30   # Will be overridden by regime-specific params
    )
    
    # Create regime-aware strategy wrapper
    strategy = RegimeAwareStrategy(base_strategy, detector)
    strategy.set_event_bus(event_bus)
    
    # Set regime-specific parameters
    for regime, params in regime_params.items():
        strategy.set_regime_parameters(regime, params)
    
    # Run simple backtest
    equity = 100000.0
    position = 0
    entry_price = 0
    equity_curve = []
    signals = []
    regime_transitions = []
    current_regime = None
    
    # Reset data handler
    data_handler.reset()
    detector.reset()  # Reset detector state
    
    while True:
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break
            
        # Current timestamp and price
        timestamp = bar.get_timestamp()
        price = bar.get_close()
        
        # Track regime
        regime = detector.update(bar)
        if regime != current_regime:
            regime_transitions.append({
                'timestamp': timestamp,
                'from_regime': current_regime.value if current_regime else None,
                'to_regime': regime.value
            })
            current_regime = regime
        
        # Process bar with strategy
        signal = strategy.on_bar(bar)
        
        # Record signal if generated
        if signal:
            signals.append({
                'timestamp': timestamp,
                'signal_value': signal.get_signal_value(),
                'price': price,
                'regime': regime.value
            })
            
            signal_value = signal.get_signal_value()
            
            # Execute trades based on signals
            if signal_value == 1 and position <= 0:  # Buy signal
                # Close short position if any
                if position < 0:
                    profit = entry_price - price
                    equity += profit * abs(position)
                
                # Open long position
                position = 100
                entry_price = price
                
            elif signal_value == -1 and position >= 0:  # Sell signal
                # Close long position if any
                if position > 0:
                    profit = price - entry_price
                    equity += profit * position
                
                # Open short position
                position = -100
                entry_price = price
        
        # Record equity
        equity_curve.append({
            'timestamp': timestamp,
            'equity': equity + position * (price - entry_price),
            'regime': regime.value
        })
    
    logger.info(f"Backtest completed: Final equity: ${equity:,.2f}")
    logger.info(f"Signals generated: {len(signals)}")
    logger.info(f"Regime transitions: {len(regime_transitions)}")
    
    return equity_curve, signals, regime_transitions


def compare_regime_vs_standard(data_handler, symbol, regime_params):
    """
    Compare performance of regime-aware strategy vs standard strategy.
    
    Args:
        data_handler: Data handler with loaded data
        symbol: Symbol to test
        regime_params: Dictionary of regime-specific parameters
        
    Returns:
        dict: Comparison results
    """
    logger.info(f"Comparing regime-aware vs standard strategies for {symbol}")
    
    # Get baseline parameters (what would be used without regime optimization)
    baseline_params = regime_params[MarketRegime.UNKNOWN]
    
    # Run standard backtest with baseline parameters
    event_bus = EventBus()
    event_tracker = EventTracker()
    event_bus.register(EventType.SIGNAL, event_tracker.track_event)
    
    standard_strategy = MovingAverageCrossoverStrategy(
        name="ma_crossover",
        symbols=[symbol],
        fast_window=baseline_params['fast_window'],
        slow_window=baseline_params['slow_window']
    )
    standard_strategy.set_event_bus(event_bus)
    
    # Run simple backtest for standard strategy
    equity = 100000.0
    position = 0
    entry_price = 0
    standard_equity_curve = []
    
    # Reset data handler
    data_handler.reset()
    
    while True:
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break
            
        # Current timestamp and price
        timestamp = bar.get_timestamp()
        price = bar.get_close()
        
        # Process bar with strategy
        signal = standard_strategy.on_bar(bar)
        
        # Execute trades based on signals
        if signal:
            signal_value = signal.get_signal_value()
            
            # Buy signal
            if signal_value == 1 and position <= 0:
                # Close short position if any
                if position < 0:
                    profit = entry_price - price
                    equity += profit * abs(position)
                
                # Open long position
                position = 100
                entry_price = price
                
            # Sell signal
            elif signal_value == -1 and position >= 0:
                # Close long position if any
                if position > 0:
                    profit = price - entry_price
                    equity += profit * position
                
                # Open short position
                position = -100
                entry_price = price
        
        # Record equity
        standard_equity_curve.append({
            'timestamp': timestamp,
            'equity': equity + position * (price - entry_price)
        })
    
    # Run regime-aware backtest
    # Create detector first
    detector = RegimeDetectorFactory.create_detector(
    detector_type='enhanced',
        lookback_window=20,              # Shorter lookback for faster response
        trend_threshold=0.01,            # More sensitive trend detection (was 0.03)
        volatility_threshold=0.008,      # More sensitive volatility detection (was 0.012)
        sideways_threshold=0.005,        # Stricter sideways definition (was 0.015)
        trend_lookback=30,               # Shorter trend lookback
        volatility_lookback=15,          # Shorter volatility lookback
        debug=True                       # Enable debug output
    )
    # detector = RegimeDetectorFactory.create_preset_detector(
    #     preset='advanced_sensitive',
    #     debug=False
    # )
    
    # Process data to build regime history
    data_handler.reset()
    while True:
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break
        detector.update(bar)
    
    # Run regime-aware backtest
    regime_equity_curve, _, _ = run_regime_aware_backtest(
        data_handler=data_handler,
        symbol=symbol,
        regime_params=regime_params,
        detector=detector
    )
    
    # Calculate performance metrics
    standard_final_equity = standard_equity_curve[-1]['equity']
    standard_initial_equity = standard_equity_curve[0]['equity']
    standard_return = (standard_final_equity / standard_initial_equity) - 1
    
    regime_final_equity = regime_equity_curve[-1]['equity']
    regime_initial_equity = regime_equity_curve[0]['equity']
    regime_return = (regime_final_equity / regime_initial_equity) - 1
    
    # Calculate other metrics (e.g., drawdown, Sharpe ratio)
    # ...
    
    # Print comparison
    logger.info("\n=== Performance Comparison ===")
    logger.info(f"Standard Strategy Return: {standard_return:.2%}")
    logger.info(f"Regime-Aware Strategy Return: {regime_return:.2%}")
    logger.info(f"Improvement: {(regime_return - standard_return):.2%}")
    
    # Plot equity curves
    plt.figure(figsize=(12, 6))
    
    # Convert to DataFrames
    df_standard = pd.DataFrame(standard_equity_curve)
    df_standard.set_index('timestamp', inplace=True)
    
    df_regime = pd.DataFrame(regime_equity_curve)
    df_regime.set_index('timestamp', inplace=True)
    
    # Plot
    plt.plot(df_standard.index, df_standard['equity'], 'b-', 
             label=f'Standard Strategy: {standard_return:.2%}')
    plt.plot(df_regime.index, df_regime['equity'], 'g-', 
             label=f'Regime-Aware Strategy: {regime_return:.2%}')
    
    plt.title(f'{symbol} Strategy Comparison')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Return comparison results
    return {
        'standard': {
            'return': standard_return,
            'final_equity': standard_final_equity,
            'equity_curve': standard_equity_curve
        },
        'regime_aware': {
            'return': regime_return,
            'final_equity': regime_final_equity,
            'equity_curve': regime_equity_curve
        },
        'improvement': regime_return - standard_return
    }


# def main():
#     """Main function to run the example."""
#     print("\n=== Regime-Based Optimization Example ===\n")
    
#     # Configuration
#     data_dir = os.path.expanduser("~/data")  # Update with your data directory
#     symbol = "SPY"
#     timeframe = "1m"
    
#     # Set specific date range
#     start_date = "2024-03-26"
#     end_date = "2024-04-26"
    
#     # Check if data directory exists
#     if not os.path.exists(data_dir):
#         data_dir = "data"  # Try relative path
        
#     # 1. Load sample data
#     print(f"Loading data for {symbol} from {data_dir}...")
#     data_source, data_handler = load_sample_data(data_dir, symbol, timeframe)
    
#     # 2. Detect market regimes with specific date range
#     print(f"\nRunning regime detection from {start_date} to {end_date}...")
#     detector = detect_market_regimes(
#         data_handler, 
#         symbol, 
#         balance_regimes=True,
#         start_date=start_date,
#         end_date=end_date
#     )
    
#     # 4. Optimize for different regimes
#     print("\nRunning regime-specific optimization...")
#     regime_params = optimize_for_regimes(
#         data_handler, 
#         symbol, 
#         detector,
#         start_date=start_date,
#         end_date=end_date
#     )
    
#     # 5. Compare regime-aware vs standard strategies
#     print("\nComparing regime-aware vs standard strategies...")
#     comparison = compare_regime_vs_standard(
#         data_handler, 
#         symbol, 
#         regime_params,
#         start_date=start_date,
#         end_date=end_date
#     )
    
#     print("\nExample completed! Check the plots for visualization.")

def main():
    """Main function to run the example."""
    print("\n=== Regime-Based Optimization Example ===\n")
    
    # Configuration
    data_dir = os.path.expanduser("~/data")  # Update with your data directory
    symbol = "SAMPLE"
    timeframe = "1m"
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        data_dir = "data"  # Try relative path
        
    # 1. Load sample data
    print(f"Loading data for {symbol} from {data_dir}...")
    data_source, data_handler = load_sample_data(data_dir, symbol, timeframe)
    
    # 2. Detect market regimes
    print("\nRunning regime detection...")
    detector = detect_market_regimes(data_handler, symbol, balance_regimes=True)
    
    # # 3. Visualize regimes
    # print("\nVisualizing regimes...")
    # visualize_regimes(data_handler, symbol, detector)
    
    # 4. Optimize for different regimes
    print("\nRunning regime-specific optimization...")
    regime_params = optimize_for_regimes(data_handler, symbol, detector)
    
    # 5. Compare regime-aware vs standard strategies
    print("\nComparing regime-aware vs standard strategies...")
    comparison = compare_regime_vs_standard(data_handler, symbol, regime_params)
    
    print("\nExample completed! Check the plots for visualization.")


if __name__ == "__main__":
    main()
