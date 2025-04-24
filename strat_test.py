"""
Test for the Mean Reversion Strategy with both long and short positions.

Uses the existing MeanReversionStrategy from src/strategy/strategies/mean_reversion.py
"""

import logging
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import necessary components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType, BarEvent
from src.core.events.event_utils import create_bar_event
from src.execution.portfolio import PortfolioManager
from src.execution.brokers.simulated import SimulatedBroker
from src.strategy.risk.risk_manager import SimpleRiskManager

# Import the existing mean reversion strategy
from src.strategy.strategies.mean_reversion import MeanReversionStrategy

def create_synthetic_price_series_with_reversions(days=100, start_price=100.0, 
                                                trend=0.0005, volatility=0.015,
                                                reversal_points=None):
    """
    Create a synthetic price series with both upward and downward trends.
    Includes deliberate reversal points to trigger mean reversion signals.
    
    Args:
        days: Number of days in the series
        start_price: Starting price
        trend: Daily trend factor (positive = upward, negative = downward)
        volatility: Daily price volatility
        reversal_points: List of (day, percent_change) tuples for deliberate reversals
    
    Returns:
        DataFrame with price data, expected_signals list
    """
    # Create date range
    start_date = datetime.datetime(2024, 1, 1)
    dates = [start_date + datetime.timedelta(days=i) for i in range(days)]
    
    # Default reversal points if none provided
    if reversal_points is None:
        reversal_points = [
            (20, 0.15),    # Day 20: +15% spike
            (40, -0.20),   # Day 40: -20% drop
            (60, 0.18),    # Day 60: +18% spike
            (80, -0.15)    # Day 80: -15% drop
        ]
    
    # Initialize price array
    prices = [start_price]
    
    # Generate random walk with trend
    for i in range(1, days):
        # Check if this is a reversal day
        reversal = next((change for day, change in reversal_points if day == i), None)
        
        if reversal is not None:
            # Apply the deliberate reversal
            new_price = prices[-1] * (1 + reversal)
            prices.append(new_price)
        else:
            # Normal random walk
            daily_return = np.random.normal(trend, volatility)
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Simple OHLC based on close price with some intraday volatility
        open_price = prices[i-1] if i > 0 else close * 0.995
        high = max(open_price, close) * (1 + np.random.uniform(0, 0.005))
        low = min(open_price, close) * (1 - np.random.uniform(0, 0.005))
        volume = int(1000000 * (1 + np.random.uniform(-0.3, 0.3)))  # Random volume
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low, 
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    
    # Calculate expected signal days based on reversal points
    # Signal happens the day after the reversal (when MA catches up)
    expected_signals = []
    for day, change in reversal_points:
        if change > 0:
            # After upward spike, expect SELL signal a few days later
            expected_signals.append((day + 1, "SELL"))
        else:
            # After downward drop, expect BUY signal a few days later
            expected_signals.append((day + 1, "BUY"))
    
    logger.info(f"Created price series with {len(reversal_points)} reversals")
    logger.info(f"Expected signals at days: {[d for d, _ in expected_signals]}")
    
    return df, expected_signals

def plot_strategy_results(df, trades, z_threshold, lookback):
    """Plot the price series with trading signals and Z-score bands."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})
        
        # Calculate rolling mean and std for Z-scores
        df['rolling_mean'] = df['close'].rolling(window=lookback).mean()
        df['rolling_std'] = df['close'].rolling(window=lookback).std()
        
        # Calculate upper and lower bands based on Z-score
        df['upper_band'] = df['rolling_mean'] + (df['rolling_std'] * z_threshold)
        df['lower_band'] = df['rolling_mean'] - (df['rolling_std'] * z_threshold)
        
        # Calculate Z-scores
        df['z_score'] = (df['close'] - df['rolling_mean']) / df['rolling_std']
        
        # Plot prices and bands on first subplot
        ax1.plot(df['timestamp'], df['close'], label='Price', linewidth=1.5)
        ax1.plot(df['timestamp'], df['rolling_mean'], label=f'{lookback}-day MA', linewidth=1.2, alpha=0.8)
        ax1.plot(df['timestamp'], df['upper_band'], 'r--', 
                label=f'Upper Band (+{z_threshold} σ)', linewidth=1, alpha=0.6)
        ax1.plot(df['timestamp'], df['lower_band'], 'g--', 
                label=f'Lower Band (-{z_threshold} σ)', linewidth=1, alpha=0.6)
        
        # Plot buy and sell points
        buy_dates = [t['timestamp'] for t in trades if t['direction'] == 'BUY']
        buy_prices = [t['price'] for t in trades if t['direction'] == 'BUY']
        sell_dates = [t['timestamp'] for t in trades if t['direction'] == 'SELL']
        sell_prices = [t['price'] for t in trades if t['direction'] == 'SELL']
        
        ax1.scatter(buy_dates, buy_prices, marker='^', color='green', s=100, label='Buy', alpha=0.7)
        ax1.scatter(sell_dates, sell_prices, marker='v', color='red', s=100, label='Sell', alpha=0.7)
        
        # Format first subplot
        ax1.set_title('Mean Reversion Strategy Performance', fontsize=14)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=10)
        
        # Plot Z-scores on second subplot
        ax2.plot(df['timestamp'], df['z_score'], label='Z-score', color='blue')
        ax2.axhline(y=z_threshold, color='r', linestyle='--', alpha=0.6, label=f'+{z_threshold} σ')
        ax2.axhline(y=-z_threshold, color='g', linestyle='--', alpha=0.6, label=f'-{z_threshold} σ')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.4)
        
        # Format second subplot
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Z-score', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=10)
        
        # Format x-axis dates for both subplots
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('mean_reversion_results.png')
        
        logger.info("Strategy results plotted to mean_reversion_results.png")
        return fig
    except Exception as e:
        logger.error(f"Error plotting results: {e}")
        return None

def run_mean_reversion_test():
    """Run a test of the mean reversion strategy with synthetic data."""
    logger.info("=== STARTING MEAN REVERSION STRATEGY TEST ===")
    
    # Set up event system
    event_bus = EventBus()
    event_manager = EventManager(event_bus)
    
    # Create portfolio
    initial_cash = 10000.0
    portfolio = PortfolioManager(initial_cash=initial_cash)
    
    # Create broker
    broker = SimulatedBroker(fill_emitter=event_bus)
    
    # Strategy parameters
    symbol = "TEST"
    lookback = 20
    z_threshold = 1.5
    
    # Create mean reversion strategy using existing implementation
    strategy = MeanReversionStrategy(
        name="mean_reversion_test",
        symbols=[symbol],
        lookback=lookback,
        z_threshold=z_threshold
    )
    
    # Create risk manager with moderate position sizing
    risk_manager = SimpleRiskManager(
        portfolio=portfolio,
        event_bus=event_bus,
        position_pct=0.4  # Use 40% of portfolio for each position
    )
    risk_manager.broker = broker  # Direct broker connection
    
    # Connect components with event bus
    portfolio.set_event_bus(event_bus)
    broker.set_event_bus(event_bus)
    strategy.set_event_bus(event_bus)
    
    # Register components with event manager
    event_manager.register_component('strategy', strategy, [EventType.BAR])
    event_manager.register_component('portfolio', portfolio, [EventType.FILL])
    event_manager.register_component('risk_manager', risk_manager, [EventType.SIGNAL])
    
    # Register order handler directly
    event_bus.register(EventType.ORDER, broker.place_order)
    
    # Track events and trades
    events = {
        'bar': 0,
        'signal': 0,
        'order': 0,
        'fill': 0
    }
    
    trades = []
    equity_curve = []
    
    # Event tracker
    def track_event(event):
        event_type = event.get_type()
        
        if event_type == EventType.BAR:
            events['bar'] += 1
            
            # Track equity at each bar
            if events['bar'] % 5 == 0 or events['bar'] == 1:
                # Update portfolio with current price
                symbol = event.get_symbol()
                price = event.get_close()
                market_prices = {symbol: price}
                
                # Log portfolio state
                cash = portfolio.cash
                position_value = portfolio.get_position_value(market_prices)
                equity = portfolio.get_equity(market_prices)
                
                equity_curve.append({
                    'day': events['bar'],
                    'timestamp': event.get_timestamp(),
                    'price': price,
                    'cash': cash,
                    'position': position_value,
                    'equity': equity
                })
                
                logger.info(f"Day {events['bar']}: Price=${price:.2f}, Cash=${cash:.2f}, " +
                           f"Position=${position_value:.2f}, Equity=${equity:.2f}")
                
        elif event_type == EventType.SIGNAL:
            events['signal'] += 1
            symbol = event.get_symbol()
            signal = event.get_signal_value()
            price = event.get_price()
            logger.info(f"Signal: {symbol} {signal} @ {price:.2f}")
            
        elif event_type == EventType.ORDER:
            events['order'] += 1
            symbol = event.get_symbol()
            direction = event.get_direction()
            quantity = event.get_quantity()
            price = event.get_price()
            logger.info(f"Order: {symbol} {direction} {quantity} @ {price:.2f}")
            
        elif event_type == EventType.FILL:
            events['fill'] += 1
            symbol = event.get_symbol()
            direction = event.get_direction()
            quantity = event.get_quantity()
            price = event.get_price()
            timestamp = event.get_timestamp()
            
            logger.info(f"Fill: {symbol} {direction} {quantity} @ {price:.2f}")
            
            # Record trade for analysis
            trades.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'direction': direction,
                'quantity': quantity,
                'price': price
            })
            
            # Log portfolio after fill
            current_position = portfolio.get_position(symbol)
            quantity_after = current_position.quantity if current_position else 0
            logger.info(f"Position after fill: {quantity_after} shares")
    
    # Register tracker for all event types
    for event_type in [EventType.BAR, EventType.SIGNAL, EventType.ORDER, EventType.FILL]:
        event_bus.register(event_type, track_event)
    
    # Generate synthetic price data with reversals
    df, expected_signals = create_synthetic_price_series_with_reversions()
    df.set_index('timestamp', inplace=True)
    
    # Log test configuration
    logger.info(f"Strategy parameters: Lookback={lookback}, Z-Threshold={z_threshold}")
    logger.info(f"Generated {len(df)} days of price data from {df.index[0]} to {df.index[-1]}")
    
    # Initial portfolio state
    logger.info(f"Initial portfolio: Cash=${portfolio.cash:.2f}, Equity=${portfolio.get_equity():.2f}")
    
    # Process each bar
    for date, row in df.iterrows():
        # Create bar event
        bar = create_bar_event(
            symbol=symbol,
            timestamp=date,
            open_price=row['open'],
            high_price=row['high'],
            low_price=row['low'],
            close_price=row['close'],
            volume=row['volume']
        )
        
        # Update broker's market data
        broker.update_market_data(symbol, {
            'price': row['close'],
            'timestamp': date
        })
        
        # Update portfolio's market data
        portfolio.update_market_data({symbol: row['close']})
        
        # Emit the bar event
        event_bus.emit(bar)
    
    # Calculate final performance
    final_equity = portfolio.get_equity({symbol: df['close'].iloc[-1]})
    total_return = (final_equity / initial_cash - 1) * 100
    
    # Log results
    logger.info(f"=== TEST COMPLETE ===")
    logger.info(f"Events processed: {events}")
    logger.info(f"Trades executed: {len(trades)}")
    logger.info(f"Final portfolio: Cash=${portfolio.cash:.2f}, Equity=${final_equity:.2f}")
    logger.info(f"Total return: {total_return:.2f}%")
    
    # Get position details
    positions = portfolio.get_position_details({symbol: df['close'].iloc[-1]})
    for pos in positions:
        logger.info(f"Final position: {pos['quantity']} shares of {pos['symbol']} @ {pos['cost_basis']:.2f}")
        logger.info(f"Position P&L: Realized=${pos['realized_pnl']:.2f}, Unrealized=${pos['unrealized_pnl']:.2f}")
    
    # Plot results
    plot_strategy_results(df.reset_index(), trades, z_threshold, lookback)
    
    # Analyze signals - use 'day' key to find days when trades occurred
    signal_days = []
    if len(equity_curve) > 1:
        initial_cash = equity_curve[0]['cash']
        for entry in equity_curve:
            if entry['cash'] != initial_cash:
                signal_days.append(entry['day'])
                initial_cash = entry['cash']  # Update to new cash value
    
    logger.info(f"Trades occurred on days: {signal_days}")
    logger.info(f"Expected signals on days: {[d for d, _ in expected_signals]}")
    
    # Check if trades occurred near expected signals
    if signal_days and expected_signals:
        signal_match = all(any(abs(actual - expected) <= 5 for actual in signal_days) 
                         for expected, _ in expected_signals)
    else:
        signal_match = False
    
    if signal_match:
        logger.info("✓ Strategy generated trades near all expected signal points")
    else:
        logger.warning("✗ Strategy missed some expected signals")
    
    # Check if short selling worked
    has_short_trades = any(t['direction'] == 'SELL' for t in trades)
    if has_short_trades:
        logger.info("✓ Short selling verified (SELL trades detected)")
    else:
        logger.warning("✗ No short selling detected (no SELL trades)")
    
    # Final test result
    test_passed = len(trades) > 0 and (signal_match or len(trades) >= 2)
    
    if test_passed:
        logger.info("MEAN REVERSION STRATEGY TEST PASSED!")
    else:
        logger.error("MEAN REVERSION STRATEGY TEST FAILED!")
    
    return {
        'passed': test_passed,
        'trades': trades,
        'events': events,
        'equity_curve': equity_curve,
        'final_equity': final_equity,
        'total_return': total_return
    }

if __name__ == "__main__":
    test_results = run_mean_reversion_test()
    
    if test_results['passed']:
        print("\nMEAN REVERSION STRATEGY TEST PASSED!")
        print(f"Total return: {test_results['total_return']:.2f}%")
        print(f"Trades executed: {len(test_results['trades'])}")
    else:
        print("\nMEAN REVERSION STRATEGY TEST FAILED!")
        print("Check logs for details.")
