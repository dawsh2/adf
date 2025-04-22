#!/usr/bin/env python
"""
Simple Regime Detection Example

This is a minimal example showing how to use the regime detection components
and how to apply them with different trading strategies.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import regime detection components
from src.models.filters.regime.regime_detector import MarketRegime, EnhancedRegimeDetector
from src.models.filters.regime.detector_factory import RegimeDetectorFactory
from src.models.filters.regime.regime_strategy import RegimeAwareStrategy

# Import strategies
from src.strategy.strategies.ma_crossover import MovingAverageCrossoverStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_data(days=252, seed=42):
    """
    Generate sample price data with different regimes for testing.
    
    Args:
        days: Number of days to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame: Sample price data with date index
    """
    np.random.seed(seed)
    
    # Start with a base price
    base_price = 100.0
    
    # Define regime segments
    regimes = [
        ('uptrend', int(days * 0.25)),
        ('volatile', int(days * 0.15)),
        ('sideways', int(days * 0.2)),
        ('downtrend', int(days * 0.25)),
        ('uptrend', int(days * 0.15))
    ]
    
    # Generate price data
    dates = pd.date_range(start='2020-01-01', periods=days)
    prices = []
    true_regimes = []
    current_price = base_price
    
    for regime, length in regimes:
        if regime == 'uptrend':
            # Uptrend: consistent upward movement with some noise
            for _ in range(length):
                drift = np.random.normal(0.001, 0.005)
                current_price *= (1 + max(0.0005, drift))
                prices.append(current_price)
                true_regimes.append(MarketRegime.UPTREND.value)
                
        elif regime == 'downtrend':
            # Downtrend: consistent downward movement with some noise
            for _ in range(length):
                drift = np.random.normal(-0.001, 0.005)
                current_price *= (1 + min(-0.0005, drift))
                prices.append(current_price)
                true_regimes.append(MarketRegime.DOWNTREND.value)
                
        elif regime == 'sideways':
            # Sideways: mean-reverting with low volatility
            local_mean = current_price
            for _ in range(length):
                # Mean-reverting component
                reversion = (local_mean - current_price) * 0.05
                noise = np.random.normal(0, 0.003) * current_price
                current_price += reversion + noise
                prices.append(current_price)
                true_regimes.append(MarketRegime.SIDEWAYS.value)
                
        elif regime == 'volatile':
            # Volatile: higher volatility with some momentum
            for _ in range(length):
                noise = np.random.normal(0, 0.015) * current_price
                current_price += noise
                prices.append(current_price)
                true_regimes.append(MarketRegime.VOLATILE.value)
    
    # Create DataFrame
    df = pd.DataFrame({
        'price': prices,
        'true_regime': true_regimes
    }, index=dates)
    
    # Add OHLC data for more realistic simulation
    df['open'] = df['price'] * (1 + np.random.normal(0, 0.002, len(df)))
    df['high'] = df[['price', 'open']].max(axis=1) * (1 + abs(np.random.normal(0, 0.003, len(df))))
    df['low'] = df[['price', 'open']].min(axis=1) * (1 - abs(np.random.normal(0, 0.003, len(df))))
    df['close'] = df['price']
    df['volume'] = np.random.randint(100000, 1000000, len(df))
    
    return df

class BarEvent:
    """Simple bar event for testing."""
    
    def __init__(self, symbol, timestamp, open_price, high_price, low_price, close_price, volume):
        self.symbol = symbol
        self.timestamp = timestamp
        self.open = open_price
        self.high = high_price
        self.low = low_price
        self.close = close_price
        self.volume = volume
    
    def get_symbol(self):
        return self.symbol
        
    def get_timestamp(self):
        return self.timestamp
        
    def get_open(self):
        return self.open
        
    def get_high(self):
        return self.high
        
    def get_low(self):
        return self.low
        
    def get_close(self):
        return self.close
        
    def get_volume(self):
        return self.volume

def main():
    """Main function to run the example."""
    print("\n=== Simple Regime Detection Example ===\n")
    
    symbol = "SAMPLE"
    
    # 1. Generate sample data
    print(f"Generating sample data for {symbol}...")
    df = generate_sample_data(days=252)
    print(f"Generated {len(df)} days of sample data\n")
    
    # 2. Create regime detector
    print("Creating regime detector...")
    detector = RegimeDetectorFactory.create_preset_detector(
        preset='advanced_sensitive',
        debug=True
    )
    
    # 3. Process data to detect regimes
    print("Detecting regimes...")
    
    detected_regimes = []
    
    for idx, row in df.iterrows():
        # Create bar event from row data
        bar = BarEvent(
            symbol=symbol,
            timestamp=idx,
            open_price=row['open'],
            high_price=row['high'],
            low_price=row['low'],
            close_price=row['close'],
            volume=row['volume']
        )
        
        # Update detector
        regime = detector.update(bar)
        detected_regimes.append(regime.value)
    
    # Add detected regimes to DataFrame
    df['detected_regime'] = detected_regimes
    
    # 4. Create base strategy and regime-aware wrapper
    print("Creating strategies...")
    
    base_strategy = MovingAverageCrossoverStrategy(
        name="ma_crossover",
        symbols=[symbol],
        fast_window=10,
        slow_window=30
    )
    
    regime_strategy = RegimeAwareStrategy(base_strategy, detector)
    
    # Set regime-specific parameters
    regime_strategy.set_regime_parameters(MarketRegime.UPTREND, {'fast_window': 3, 'slow_window': 15})
    regime_strategy.set_regime_parameters(MarketRegime.DOWNTREND, {'fast_window': 8, 'slow_window': 40})
    regime_strategy.set_regime_parameters(MarketRegime.SIDEWAYS, {'fast_window': 10, 'slow_window': 30})
    regime_strategy.set_regime_parameters(MarketRegime.VOLATILE, {'fast_window': 5, 'slow_window': 25})
    
    # 5. Generate signals with standard and regime-aware strategies
    print("Generating trading signals...")
    
    standard_signals = []
    regime_signals = []
    
    # Reset detector and strategies
    detector.reset()
    base_strategy.reset()
    regime_strategy.reset()
    
    for idx, row in df.iterrows():
        # Create bar event from row data
        bar = BarEvent(
            symbol=symbol,
            timestamp=idx,
            open_price=row['open'],
            high_price=row['high'],
            low_price=row['low'],
            close_price=row['close'],
            volume=row['volume']
        )
        
        # Update detector (needed for regime strategy)
        detector.update(bar)
        
        # Get signals from both strategies
        standard_signal = base_strategy.on_bar(bar)
        regime_signal = regime_strategy.on_bar(bar)
        
        # Record signals (1 for buy, -1 for sell, 0 for no signal)
        standard_signals.append(1 if standard_signal and standard_signal.get_signal_value() == 1 
                             else -1 if standard_signal and standard_signal.get_signal_value() == -1 
                             else 0)
        
        regime_signals.append(1 if regime_signal and regime_signal.get_signal_value() == 1 
                           else -1 if regime_signal and regime_signal.get_signal_value() == -1 
                           else 0)
    
    # Add signals to DataFrame
    df['standard_signal'] = standard_signals
    df['regime_signal'] = regime_signals
    
    # 6. Visualize results
    print("Visualizing results...")
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Set title
    fig.suptitle(f'{symbol} Price, Regimes, and Signals', fontsize=16)
    
    # Plot 1: Price
    ax1.plot(df.index, df['close'], 'k-', linewidth=1)
    ax1.set_ylabel('Price')
    ax1.grid(True)
    
    # Color backgrounds by regime
    regimes = df['detected_regime'].unique()
    regime_colors = {
        MarketRegime.UPTREND.value: 'lightgreen',
        MarketRegime.DOWNTREND.value: 'lightcoral',
        MarketRegime.SIDEWAYS.value: 'lightyellow',
        MarketRegime.VOLATILE.value: 'lightblue',
        MarketRegime.UNKNOWN.value: 'white'
    }
    
    # Find regime change points
    regime_changes = []
    prev_regime = None
    
    for i, regime in enumerate(df['detected_regime']):
        if regime != prev_regime:
            regime_changes.append((i, regime))
            prev_regime = regime
    
    # Add colored background for each regime segment
    for i in range(len(regime_changes) - 1):
        start_idx = regime_changes[i][0]
        end_idx = regime_changes[i+1][0]
        regime = regime_changes[i][1]
        
        color = regime_colors.get(regime, 'white')
        ax1.axvspan(df.index[start_idx], df.index[end_idx], 
                   facecolor=color, alpha=0.3)
    
    # Add last segment
    if regime_changes:
        start_idx = regime_changes[-1][0]
        regime = regime_changes[-1][1]
        color = regime_colors.get(regime, 'white')
        ax1.axvspan(df.index[start_idx], df.index[-1], 
                   facecolor=color, alpha=0.3)
    
    # Plot 2: Regimes (true vs detected)
    # Convert regimes to numeric for easier plotting
    regime_values = {
        MarketRegime.UPTREND.value: 2,
        MarketRegime.VOLATILE.value: 1,
        MarketRegime.SIDEWAYS.value: 0,
        MarketRegime.DOWNTREND.value: -1,
        MarketRegime.UNKNOWN.value: -2
    }
    
    df['true_regime_value'] = df['true_regime'].map(regime_values)
    df['detected_regime_value'] = df['detected_regime'].map(regime_values)
    
    ax2.plot(df.index, df['true_regime_value'], 'b-', label='True Regime')
    ax2.plot(df.index, df['detected_regime_value'], 'r--', label='Detected Regime')
    ax2.set_ylabel('Regime')
    ax2.grid(True)
    ax2.legend()
    
    # Set y-ticks for regimes
    ax2.set_yticks(list(regime_values.values()))
    ax2.set_yticklabels([k for k in regime_values.keys()])
    
    # Plot 3: Signals
    ax3.plot(df.index, df['standard_signal'], 'b-', label='Standard Strategy')
    ax3.plot(df.index, df['regime_signal'], 'r-', label='Regime Strategy')
    ax3.set_ylabel('Signal')
    ax3.set_xlabel('Date')
    ax3.grid(True)
    ax3.legend()
    
    # Set y-ticks for signals
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(['Sell', 'Neutral', 'Buy'])
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.show()
    
    # Print regime detection statistics
    print("\nRegime Detection Statistics:")
    
    # Calculate accuracy
    accuracy = (df['true_regime'] == df['detected_regime']).mean()
    print(f"Detection Accuracy: {accuracy:.2%}")
    
    # Print regime distribution
    print("\nTrue Regime Distribution:")
    true_counts = df['true_regime'].value_counts()
    for regime, count in true_counts.items():
        print(f"  {regime}: {count} days ({count/len(df):.2%})")
    
    print("\nDetected Regime Distribution:")
    detected_counts = df['detected_regime'].value_counts()
    for regime, count in detected_counts.items():
        print(f"  {regime}: {count} days ({count/len(df):.2%})")
    
    # Print signal statistics
    print("\nSignal Statistics:")
    print(f"Standard Strategy Signals: {(df['standard_signal'] != 0).sum()} signals")
    print(f"Regime Strategy Signals: {(df['regime_signal'] != 0).sum()} signals")
    
    # Calculate hypothetical returns
    df['standard_position'] = df['standard_signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    df['regime_position'] = df['regime_signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    df['price_return'] = df['close'].pct_change()
    df['standard_return'] = df['standard_position'].shift(1) * df['price_return']
    df['regime_return'] = df['regime_position'].shift(1) * df['price_return']
    
    standard_total_return = (1 + df['standard_return'].fillna(0)).prod() - 1
    regime_total_return = (1 + df['regime_return'].fillna(0)).prod() - 1
    
    print("\nHypothetical Returns (no transaction costs):")
    print(f"Buy and Hold Return: {(df['close'].iloc[-1] / df['close'].iloc[0] - 1):.2%}")
    print(f"Standard Strategy Return: {standard_total_return:.2%}")
    print(f"Regime Strategy Return: {regime_total_return:.2%}")
    print(f"Improvement: {(regime_total_return - standard_total_return):.2%}")
    
    print("\nExample completed!")


if __name__ == "__main__":
    main()
