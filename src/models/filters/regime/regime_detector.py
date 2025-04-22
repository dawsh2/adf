"""
Regime Detection Module

This module provides components for detecting market regimes from price data,
which can be used to adjust trading strategies based on market conditions.
"""
import logging
import numpy as np
import pandas as pd
import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Enumeration of market regime types."""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class RegimeDetectorBase(ABC):
    """
    Abstract base class for market regime detectors.
    
    Regime detectors analyze market data to identify different market conditions
    or "regimes" such as trends, ranges, or high-volatility periods.
    """
    
    def __init__(self, lookback_window=20, name=None):
        """
        Initialize the regime detector.
        
        Args:
            lookback_window: Period for regime analysis (in bars)
            name: Optional detector name
        """
        self.lookback_window = lookback_window
        self.name = name or self.__class__.__name__
        self.price_history = {}  # symbol -> list of (timestamp, price) tuples
        self.regime_history = {}  # symbol -> list of (timestamp, regime) tuples
    
    @abstractmethod
    def update(self, bar):
        """
        Update detector with new price data and detect current regime.
        
        Args:
            bar: Bar event with price data
            
        Returns:
            MarketRegime: Detected market regime
        """
        pass
    
    def get_current_regime(self, symbol):
        """
        Get current regime for a symbol.
        
        Args:
            symbol: Symbol to get regime for
            
        Returns:
            MarketRegime: Current market regime
        """
        if not symbol in self.regime_history or not self.regime_history[symbol]:
            return MarketRegime.UNKNOWN
        return self.regime_history[symbol][-1][1]
    
    def get_regime_at(self, symbol, timestamp):
        """
        Get regime for a symbol at specific timestamp.
        
        Args:
            symbol: Symbol to get regime for
            timestamp: Timestamp to get regime at
            
        Returns:
            MarketRegime: Market regime at timestamp
        """
        if not symbol in self.regime_history:
            return MarketRegime.UNKNOWN
            
        # Find the regime at or before the timestamp
        # Convert timestamp and all regime timestamps to naive datetimes for comparison
        target_ts = timestamp.replace(tzinfo=None) if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo else timestamp
        
        for ts, regime in reversed(self.regime_history[symbol]):
            # Convert to naive datetime for comparison
            regime_ts = ts.replace(tzinfo=None) if hasattr(ts, 'tzinfo') and ts.tzinfo else ts
            if regime_ts <= target_ts:
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

        # Convert all timestamps to naive datetimes to avoid timezone issues
        history_naive = []
        for ts, regime in history:
            # Remove timezone info if present
            ts_naive = ts.replace(tzinfo=None) if hasattr(ts, 'tzinfo') and ts.tzinfo else ts
            history_naive.append((ts_naive, regime))

        # Convert input dates to naive datetimes
        start_naive = None
        end_naive = None

        if start_date:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            start_naive = start_date.replace(tzinfo=None) if hasattr(start_date, 'tzinfo') and start_date.tzinfo else start_date

        if end_date:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            end_naive = end_date.replace(tzinfo=None) if hasattr(end_date, 'tzinfo') and end_date.tzinfo else end_date

        # Filter by date range if specified
        filtered_history = []
        for ts, regime in history_naive:
            if start_naive and ts < start_naive:
                continue
            if end_naive and ts > end_naive:
                continue
            filtered_history.append((ts, regime))

        if not filtered_history:
            return {}

        # Find continuous periods of same regime
        regime_periods = {}

        current_regime = filtered_history[0][1]
        period_start = filtered_history[0][0]

        for i in range(1, len(filtered_history)):
            timestamp, regime = filtered_history[i]

            # Regime change
            if regime != current_regime:
                # Store previous period
                if current_regime not in regime_periods:
                    regime_periods[current_regime] = []
                regime_periods[current_regime].append((period_start, timestamp))

                # Start new period
                current_regime = regime
                period_start = timestamp

        # Add final period
        if filtered_history:
            if current_regime not in regime_periods:
                regime_periods[current_regime] = []
            regime_periods[current_regime].append((period_start, filtered_history[-1][0]))

        return regime_periods
    
    def print_regime_summary(self, symbol):
        """
        Print a detailed summary of regime detection for a symbol.
        
        Args:
            symbol: Symbol to print summary for
        """
        if not symbol in self.regime_history or not self.regime_history[symbol]:
            print(f"No regime history for {symbol}")
            return

        # Count regimes
        regime_counts = {}
        for _, regime in self.regime_history[symbol]:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        total = len(self.regime_history[symbol])

        print(f"\n=== Regime Distribution Summary for {symbol} ===")
        print(f"Total bars analyzed: {total}")
        print("\nRegime Distribution:")

        # Print bar chart for visualization
        max_count = max(regime_counts.values()) if regime_counts else 0
        bar_length = 40  # Maximum bar length for visualization

        for regime, count in sorted(regime_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            bar = "█" * int((count / max_count) * bar_length) if max_count > 0 else ""
            print(f"  {regime.value:<10}: {count:>5} ({percentage:>6.2f}%) {bar}")

        # Calculate regime transitions
        transitions = 0
        previous_regime = None
        transition_counts = {}

        for _, regime in self.regime_history[symbol]:
            if previous_regime is not None and regime != previous_regime:
                transitions += 1
                transition_key = f"{previous_regime.value} → {regime.value}"
                transition_counts[transition_key] = transition_counts.get(transition_key, 0) + 1
            previous_regime = regime

        # Print regime stability metrics
        stability = 1 - (transitions / (total - 1)) if total > 1 else 1
        print(f"\nRegime Stability: {stability:.2f} (0-1 scale, higher is more stable)")
        print(f"Total Regime Transitions: {transitions}")

        # Print most common transitions
        if transition_counts:
            print("\nMost Common Transitions:")
            for transition, count in sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {transition}: {count} times")

        # Print current regime info
        if total > 0:
            current_regime = self.regime_history[symbol][-1][1]
            current_streak = 0
            for i in range(len(self.regime_history[symbol])-1, -1, -1):
                if self.regime_history[symbol][i][1] == current_regime:
                    current_streak += 1
                else:
                    break

            print(f"\nCurrent Regime: {current_regime.value} (streak: {current_streak} bars)")

        # Print periods summary
        regime_periods = self.get_regime_periods(symbol)
        if regime_periods:
            print("\nRegime Periods Summary:")
            for regime, periods in regime_periods.items():
                total_bars = sum(1 for _ in periods)
                avg_length = total_bars / len(periods) if periods else 0
                max_length = max((end - start).total_seconds() / 86400 for start, end in periods) if periods else 0

                print(f"  {regime.value}: {len(periods)} periods, avg length: {avg_length:.1f} bars, max: {max_length:.1f} days")
    
    def reset(self):
        """Reset the detector state."""
        self.price_history = {}
        self.regime_history = {}


class BasicRegimeDetector(RegimeDetectorBase):
    """
    Simple regime detector based on price trends and volatility.
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
        super().__init__(lookback_window)
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold
        self.sideways_threshold = sideways_threshold
    
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


class EnhancedRegimeDetector(RegimeDetectorBase):
    """
    Enhanced market regime detector that uses multiple indicators to identify regimes.
    """
    
    def __init__(self, lookback_window=30, trend_lookback=50, volatility_lookback=20,
                trend_threshold=0.03, volatility_threshold=0.012, 
                sideways_threshold=0.015, debug=False):
        """
        Initialize the enhanced regime detector.
        
        Args:
            lookback_window: Primary window for regime analysis
            trend_lookback: Window for trend strength calculation
            volatility_lookback: Window for volatility calculation
            trend_threshold: Minimum price change for trend detection
            volatility_threshold: Threshold for high volatility regime
            sideways_threshold: Max range for sideways market
            debug: Enable debug output
        """
        super().__init__(lookback_window)
        self.trend_lookback = trend_lookback
        self.volatility_lookback = volatility_lookback
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold
        self.sideways_threshold = sideways_threshold
        self.debug = debug
        
        # Additional state for enhanced analysis
        self.indicator_values = {}  # symbol -> dict of indicator values
        
        logger.info(f"Enhanced Regime Detector initialized with: lookback={lookback_window}, "
                   f"trend_lookback={trend_lookback}, volatility_lookback={volatility_lookback}")
        logger.info(f"Thresholds: trend={trend_threshold}, volatility={volatility_threshold}, "
                   f"sideways={sideways_threshold}")

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
            self.indicator_values[symbol] = {}

        # Add price to history
        self.price_history[symbol].append((timestamp, close_price))

        # Keep history limited to reasonable size
        max_lookback = max(self.lookback_window, self.trend_lookback, self.volatility_lookback) * 2
        if len(self.price_history[symbol]) > max_lookback:
            self.price_history[symbol] = self.price_history[symbol][-max_lookback:]

        # Need enough history for regime detection
        if len(self.price_history[symbol]) < max(self.lookback_window, self.trend_lookback):
            regime = MarketRegime.UNKNOWN
            self.regime_history[symbol].append((timestamp, regime))
            return regime

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

        # 5. Trend consistency - how consistently prices are moving in one direction
        price_changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
        ups = sum(1 for change in price_changes if change > 0)
        consistency = abs((ups / (len(recent_prices) - 1)) - 0.5) * 2  # 0 to 1 scale

        # Store indicator values for debugging
        self.indicator_values[symbol] = {
            'short_term_change': short_term_change,
            'long_term_change': long_term_change,
            'volatility': volatility,
            'price_range': price_range,
            'trend_consistency': consistency
        }

        # Print debug information
        if self.debug:
            print(f"Regime indicators for {symbol} at {timestamp}:")
            print(f"  Short-term change: {short_term_change:.2%}")
            print(f"  Long-term change: {long_term_change:.2%}")
            print(f"  Volatility: {volatility:.2%}")
            print(f"  Price range: {price_range:.2%}")
            print(f"  Trend consistency: {consistency:.2f}")

        # IMPROVED REGIME DETECTION LOGIC
        # Prioritize detection order: volatility → strong trends → weak trends → sideways

        # Score different regimes to select the most appropriate one
        regime_scores = {
            MarketRegime.VOLATILE: 0,
            MarketRegime.UPTREND: 0,
            MarketRegime.DOWNTREND: 0,
            MarketRegime.SIDEWAYS: 0
        }

        # Score volatility regime
        if volatility > self.volatility_threshold:
            regime_scores[MarketRegime.VOLATILE] += 2

        if price_range > self.sideways_threshold * 3:
            regime_scores[MarketRegime.VOLATILE] += 1

        # Score uptrend regime
        if short_term_change > self.trend_threshold:
            regime_scores[MarketRegime.UPTREND] += 1

        if long_term_change > 0:
            regime_scores[MarketRegime.UPTREND] += 1

        if consistency > 0.6 and short_term_change > 0:
            regime_scores[MarketRegime.UPTREND] += 1

        # Score downtrend regime
        if short_term_change < -self.trend_threshold:
            regime_scores[MarketRegime.DOWNTREND] += 1

        if long_term_change < 0:
            regime_scores[MarketRegime.DOWNTREND] += 1

        if consistency > 0.6 and short_term_change < 0:
            regime_scores[MarketRegime.DOWNTREND] += 1

        # Score sideways regime
        if abs(short_term_change) < self.sideways_threshold:
            regime_scores[MarketRegime.SIDEWAYS] += 1

        if abs(long_term_change) < self.sideways_threshold * 2:
            regime_scores[MarketRegime.SIDEWAYS] += 1

        if consistency < 0.4:  # Low trend consistency indicates sideways
            regime_scores[MarketRegime.SIDEWAYS] += 1

        # Select regime with highest score
        regime = max(regime_scores.items(), key=lambda x: x[1])[0]

        # If there's a tie or all scores are 0, use more explicit rules
        max_score = max(regime_scores.values())
        if max_score == 0 or len([r for r, s in regime_scores.items() if s == max_score]) > 1:
            # Fallback rules
            if volatility > self.volatility_threshold:
                regime = MarketRegime.VOLATILE
            elif abs(short_term_change) < self.sideways_threshold:
                regime = MarketRegime.SIDEWAYS
            elif short_term_change > 0:
                regime = MarketRegime.UPTREND
            else:
                regime = MarketRegime.DOWNTREND

        # Store regime
        self.regime_history[symbol].append((timestamp, regime))

        if self.debug:
            print(f"Detected regime for {symbol} at {timestamp}: {regime.value} (scores: {regime_scores})")

        return regime
    
    def get_dominant_regime(self, symbol, lookback=None):
        """
        Get the most common regime over a period.
        
        Args:
            symbol: Symbol to analyze
            lookback: Optional number of bars to look back
            
        Returns:
            MarketRegime: Most common regime
        """
        if not symbol in self.regime_history or not self.regime_history[symbol]:
            return MarketRegime.UNKNOWN
            
        history = self.regime_history[symbol]
        if lookback:
            history = history[-lookback:]
            
        # Count occurrences of each regime
        regime_counts = {}
        for _, regime in history:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
        # Find the most common regime
        dominant_regime = max(regime_counts.items(), key=lambda x: x[1])[0]
        return dominant_regime
    
    def get_regime_stability(self, symbol, lookback=None):
        """
        Calculate how stable the regime has been (0-1 scale).
        
        Args:
            symbol: Symbol to analyze
            lookback: Optional number of bars to look back
            
        Returns:
            float: Stability score (0-1, higher is more stable)
        """
        if not symbol in self.regime_history or not self.regime_history[symbol]:
            return 0
            
        history = self.regime_history[symbol]
        if lookback:
            history = history[-lookback:]
            
        if len(history) <= 1:
            return 1  # Only one regime, so it's stable
            
        # Count regime transitions
        transitions = 0
        for i in range(1, len(history)):
            if history[i][1] != history[i-1][1]:
                transitions += 1
                
        # Calculate stability (1 - transition rate)
        stability = 1 - (transitions / (len(history) - 1))
        return stability
