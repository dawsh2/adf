"""
Test for the Mean Reversion Strategy using real market data.

Loads 1-minute data from 'data/SAMPLE_1m.csv' and tests the mean reversion strategy.
"""

import logging
import pandas as pd
import numpy as np
import datetime
import os
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

# Import existing data handling and processing components
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler

# Import the existing mean reversion strategy
from src.strategy.strategies.mean_reversion import MeanReversionStrategy

def analyze_mean_reversion_signals(df, lookback=20, z_threshold=1.5):
    """
    Analyze the data to predict where mean reversion signals should occur.
    This helps us validate that the strategy is working as expected.
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Lookback period for moving average
        z_threshold: Z-score threshold for signals
        
    Returns:
        List of expected signal points (dataframe indices)
    """
    # Calculate rolling mean and standard deviation
    df['rolling_mean'] = df['close'].rolling(window=lookback).mean()
    df['rolling_std'] = df['close'].rolling(window=lookback).std()
    
    # Calculate Z-scores
    df['z_score'] = (df['close'] - df['rolling_mean']) / df['rolling_std']
    
    # Find crossings of the threshold (potential signal points)
    buy_signals = []
    sell_signals = []
    
    # Skip the first lookback rows since they'll have NaN values
    for i in range(lookback+1, len(df)):
        # Check for buy signals (z-score crossing below negative threshold)
        if df['z_score'].iloc[i-1] >= -z_threshold and df['z_score'].iloc[i] < -z_threshold:
            buy_signals.append(i)
            
        # Check for sell signals (z-score crossing above positive threshold)
        if df['z_score'].iloc[i-1] <= z_threshold and df['z_score'].iloc[i] > z_threshold:
            sell_signals.append(i)
    
    logger.info(f"Analysis found {len(buy_signals)} potential buy signals and {len(sell_signals)} potential sell signals")
    
    return {
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'df_with_metrics': df  # Return the DataFrame with added metrics for plotting
    }

def plot_results(df, trades, lookback, z_threshold, save_path='mean_reversion_real_data_results.png'):
    """Plot the results of the mean reversion strategy on real data."""
    try:
        # Calculate strategy metrics if not already present
        if 'z_score' not in df.columns:
            df['rolling_mean'] = df['close'].rolling(window=lookback).mean()
            df['rolling_std'] = df['close'].rolling(window=lookback).std()
            df['z_score'] = (df['close'] - df['rolling_mean']) / df['rolling_std']
            df['upper_band'] = df['rolling_mean'] + (df['rolling_std'] * z_threshold)
            df['lower_band'] = df['rolling_mean'] - (df['rolling_std'] * z_threshold)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price and bands
        ax1.plot(df.index, df['close'], label='Price', linewidth=1.5)
        ax1.plot(df.index, df['rolling_mean'], label=f'{lookback}-period MA', linewidth=1.2, alpha=0.8)
        ax1.plot(df.index, df['upper_band'], 'r--', 
                label=f'Upper Band (+{z_threshold} σ)', linewidth=1, alpha=0.6)
        ax1.plot(df.index, df['lower_band'], 'g--', 
                label=f'Lower Band (-{z_threshold} σ)', linewidth=1, alpha=0.6)
        
        # Plot trades
        if trades:
            # Extract timestamps and prices
            buy_timestamps = [t['timestamp'] for t in trades if t['direction'] == 'BUY']
            buy_prices = [t['price'] for t in trades if t['direction'] == 'BUY']
            sell_timestamps = [t['timestamp'] for t in trades if t['direction'] == 'SELL']
            sell_prices = [t['price'] for t in trades if t['direction'] == 'SELL']
            
            # Plot trades
            if buy_timestamps:
                ax1.scatter(buy_timestamps, buy_prices, marker='^', color='green', s=80, label='Buy', alpha=0.7)
            if sell_timestamps:
                ax1.scatter(sell_timestamps, sell_prices, marker='v', color='red', s=80, label='Sell', alpha=0.7)
        
        # Format first subplot
        ax1.set_title('Mean Reversion Strategy on Real Market Data', fontsize=14)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=10)
        
        # Plot Z-scores
        ax2.plot(df.index, df['z_score'], label='Z-score', color='blue')
        ax2.axhline(y=z_threshold, color='r', linestyle='--', alpha=0.6, label=f'+{z_threshold} σ')
        ax2.axhline(y=-z_threshold, color='g', linestyle='--', alpha=0.6, label=f'-{z_threshold} σ')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.4)
        
        # Format second subplot
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Z-score', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=10)
        
        # Format datetime on x-axis
        for ax in [ax1, ax2]:
            if isinstance(df.index, pd.DatetimeIndex):
                ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(save_path)
        
        logger.info(f"Strategy results plotted to {save_path}")
        return fig
    except Exception as e:
        logger.error(f"Error plotting results: {e}")
        return None

def run_real_data_test():
    """Run mean reversion strategy test on real 1-minute data."""
    logger.info("=== STARTING MEAN REVERSION STRATEGY TEST ON REAL DATA ===")
    
    # Data file path
    csv_file = 'data/SAMPLE_1m.csv'
    if not os.path.exists(csv_file):
        logger.error(f"Data file not found: {csv_file}")
        return False
    
    # Load data from CSV
    logger.info(f"Loading data from {csv_file}")
    
    try:
        # Create data source and handler
        data_dir = os.path.dirname(csv_file)
        filename = os.path.basename(csv_file)
        
        # Extract symbol from filename (assuming format SYMBOL_timeframe.csv)
        symbol = filename.split('_')[0]
        
        # Create a CSV data source
        data_source = CSVDataSource(
            data_dir=data_dir, 
            filename_pattern=filename  # Use exact filename
        )
        
        # Create bar emitter (required by HistoricalDataHandler)
        event_bus = EventBus()
        from src.core.events.event_emitters import BarEmitter
        bar_emitter = BarEmitter(name="bar_emitter", event_bus=event_bus)
        bar_emitter.start()
        
        # Create historical data handler
        data_handler = HistoricalDataHandler(
            data_source=data_source,
            bar_emitter=bar_emitter
        )
        
        # Load data for the symbol
        data_handler.load_data(symbols=[symbol])
        
        # Check if data was loaded successfully
        if symbol not in data_handler.data_frames:
            logger.error(f"Failed to load data for symbol {symbol}")
            return False
            
        # Get the DataFrame for analysis
        df = data_handler.data_frames[symbol].copy()
        
        # Log data information
        logger.info(f"Loaded {len(df)} bars for {symbol} from {df.index.min()} to {df.index.max()}")
        
        # Set up event system
        event_manager = EventManager(event_bus)
        
        # Create portfolio
        initial_cash = 10000.0
        portfolio = PortfolioManager(initial_cash=initial_cash)
        
        # Create broker
        broker = SimulatedBroker(fill_emitter=event_bus)
        
        # Strategy parameters
        lookback = 20  # Use 20-period lookback
        z_threshold = 1.5  # Use 1.5 standard deviations
        
        # Create mean reversion strategy
        strategy = MeanReversionStrategy(
            name="mean_reversion_real",
            symbols=[symbol],
            lookback=lookback,
            z_threshold=z_threshold
        )
        
        # Create risk manager
        risk_manager = SimpleRiskManager(
            portfolio=portfolio,
            event_bus=event_bus,
            position_pct=0.3  # Use 30% of portfolio for each position
        )
        risk_manager.broker = broker
        
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
        
        # Analyze data to determine expected signals
        analysis_result = analyze_mean_reversion_signals(df, lookback, z_threshold)
        expected_buy_signals = analysis_result['buy_signals']
        expected_sell_signals = analysis_result['sell_signals']
        
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
                
                # Track equity at regular intervals
                if events['bar'] % 50 == 0 or events['bar'] == 1:
                    # Update portfolio with current price
                    symbol = event.get_symbol()
                    price = event.get_close()
                    market_prices = {symbol: price}
                    
                    # Log portfolio state
                    cash = portfolio.cash
                    position_value = portfolio.get_position_value(market_prices)
                    equity = portfolio.get_equity(market_prices)
                    
                    equity_curve.append({
                        'bar': events['bar'],
                        'timestamp': event.get_timestamp(),
                        'price': price,
                        'cash': cash,
                        'position': position_value,
                        'equity': equity
                    })
                    
                    logger.info(f"Bar {events['bar']}: Price=${price:.2f}, Cash=${cash:.2f}, " +
                               f"Position=${position_value:.2f}, Equity=${equity:.2f}")
                    
            elif event_type == EventType.SIGNAL:
                events['signal'] += 1
                symbol = event.get_symbol()
                signal = event.get_signal_value()
                price = event.get_price()
                logger.info(f"Signal {events['signal']}: {symbol} {signal} @ {price:.2f}")
                
            elif event_type == EventType.ORDER:
                events['order'] += 1
                symbol = event.get_symbol()
                direction = event.get_direction()
                quantity = event.get_quantity()
                price = event.get_price()
                logger.info(f"Order {events['order']}: {symbol} {direction} {quantity} @ {price:.2f}")
                
            elif event_type == EventType.FILL:
                events['fill'] += 1
                symbol = event.get_symbol()
                direction = event.get_direction()
                quantity = event.get_quantity()
                price = event.get_price()
                timestamp = event.get_timestamp()
                
                logger.info(f"Fill {events['fill']}: {symbol} {direction} {quantity} @ {price:.2f}")
                
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
        
        # Reset data handler
        data_handler.reset()
        
        # Process each bar
        logger.info("Starting backtest with real market data...")
        
        # Keep track of the current bar index for signal validation
        current_bar_index = 0
        
        # Process all bars
        while True:
            # Get next bar
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
                
            # Update broker's market data
            price = bar.get_close()
            timestamp = bar.get_timestamp()
            
            broker.update_market_data(symbol, {
                'price': price,
                'timestamp': timestamp
            })
            
            # Update portfolio's market data
            portfolio.update_market_data({symbol: price})
            
            # Emit the bar event
            event_bus.emit(bar)
            
            # Increment bar index
            current_bar_index += 1
        
        # Calculate final performance
        final_price = df['close'].iloc[-1]
        final_equity = portfolio.get_equity({symbol: final_price})
        total_return = (final_equity / initial_cash - 1) * 100
        
        # Log results
        logger.info(f"=== TEST COMPLETE ===")
        logger.info(f"Events processed: {events}")
        logger.info(f"Trades executed: {len(trades)}")
        logger.info(f"Final portfolio: Cash=${portfolio.cash:.2f}, Equity=${final_equity:.2f}")
        logger.info(f"Total return: {total_return:.2f}%")
        
        # Get position details
        positions = portfolio.get_position_details({symbol: final_price})
        for pos in positions:
            logger.info(f"Final position: {pos['quantity']} shares of {pos['symbol']} @ {pos['cost_basis']:.2f}")
            logger.info(f"Position P&L: Realized=${pos['realized_pnl']:.2f}, Unrealized=${pos['unrealized_pnl']:.2f}")
        
        # Plot results
        plot_results(df, trades, lookback, z_threshold)
        
        # Compare actual signals/trades with expected signals
        logger.info(f"Expected buy signals at bars: {expected_buy_signals}")
        logger.info(f"Expected sell signals at bars: {expected_sell_signals}")
        logger.info(f"Actual trades executed: {len(trades)}")
        
        # Classify trades
        buy_trades = [t for t in trades if t['direction'] == 'BUY']
        sell_trades = [t for t in trades if t['direction'] == 'SELL']
        
        logger.info(f"Buy trades: {len(buy_trades)}")
        logger.info(f"Sell trades: {len(sell_trades)}")
        
        # Determine test success
        # Consider successful if we got at least some trades and made a profit
        test_passed = len(trades) > 0 and total_return > 0
        
        if test_passed:
            logger.info("REAL DATA TEST PASSED!")
        else:
            logger.error("REAL DATA TEST FAILED!")
        
        return {
            'passed': test_passed,
            'trades': trades,
            'events': events,
            'equity_curve': equity_curve,
            'final_equity': final_equity,
            'total_return': total_return,
            'dataframe': df
        }
        
    except Exception as e:
        logger.exception(f"Error during real data test: {e}")
        return {'passed': False, 'error': str(e)}

if __name__ == "__main__":
    test_results = run_real_data_test()
    
    if isinstance(test_results, dict) and test_results.get('passed', False):
        print("\nREAL DATA TEST PASSED!")
        print(f"Total return: {test_results['total_return']:.2f}%")
        print(f"Trades executed: {len(test_results['trades'])}")
    else:
        print("\nREAL DATA TEST FAILED!")
        print("Check logs for details.")
