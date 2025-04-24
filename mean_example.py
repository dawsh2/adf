"""
Mean Reversion Strategy Diagnostic Script - Simplified

This script runs a basic mean reversion strategy with detailed debug logging.
"""
import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Import core components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType, SignalEvent
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler
from src.strategy.strategies.mean_reversion import MeanReversionStrategy
from src.execution.backtest.backtest import run_backtest

# Configure logging - increase level for more details
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_debug_backtest(symbol, timeframe="1m", lookback=20, z_threshold=1.5, initial_cash=10000.0):
    """
    Run a simplified debug backtest.
    """
    logger.info(f"=== RUNNING DEBUG BACKTEST FOR {symbol} ===")
    
    # Set up data
    data_path = f"./data/{symbol}_{timeframe}.csv"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return None, None, {"error": "Data file not found"}
    
    # Create data source and handler
    data_dir = os.path.dirname(data_path)
    filename = os.path.basename(data_path)
    
    data_source = CSVDataSource(
        data_dir=data_dir,
        filename_pattern=filename
    )
    
    # Create data handler
    data_handler = HistoricalDataHandler(
        data_source=data_source,
        bar_emitter=None  # We'll create this later
    )
    
    # Load data
    data_handler.load_data(symbols=[symbol])
    
    # Create a manual testing loop
    logger.info("Running manual verification of mean reversion strategy")
    
    # Create the strategy directly
    strategy = MeanReversionStrategy(
        name="mean_reversion_test",
        symbols=[symbol],
        lookback=lookback,
        z_threshold=z_threshold
    )
    
    # Process data manually to check signal generation
    event_bus = EventBus()
    bar_emitter = event_bus
    
    # Signal counter
    buy_signals = 0
    sell_signals = 0
    
    # Reset data handler
    data_handler.reset()
    data_handler.bar_emitter = bar_emitter
    
    # Process each bar manually
    while True:
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break
        
        # Process the bar through the strategy directly
        signal = strategy.on_bar(bar)
        
        # Check if signal was generated
        if signal:
            signal_type = signal.get_signal_value()
            if signal_type == SignalEvent.BUY:
                buy_signals += 1
                logger.info(f"BUY signal at {bar.get_timestamp()}, price: {bar.get_close()}")
            elif signal_type == SignalEvent.SELL:
                sell_signals += 1
                logger.info(f"SELL signal at {bar.get_timestamp()}, price: {bar.get_close()}")
    
    # Run the regular backtest
    from src.core.events.event_emitters import BarEmitter
    new_event_bus = EventBus()
    bar_emitter = BarEmitter(name="bar_emitter", event_bus=new_event_bus)
    bar_emitter.start()
    
    data_handler.reset()
    data_handler.bar_emitter = bar_emitter
    
    strategy.reset()
    
    # Now run the actual backtest
    logger.info(f"Running normal backtest with params: lookback={lookback}, z_threshold={z_threshold}")
    
    equity_curve, trades = run_backtest(
        component=strategy,
        data_handler=data_handler,
        initial_cash=initial_cash,
        debug=True  # Enable debug mode
    )
    
    # Calculate metrics
    metrics = {
        'manual_signals': {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'total_signals': buy_signals + sell_signals
        }
    }
    
    if equity_curve is not None and len(equity_curve) > 0:
        initial_equity = equity_curve['equity'].iloc[0]
        final_equity = equity_curve['equity'].iloc[-1]
        metrics['total_return'] = ((final_equity / initial_equity) - 1) * 100
        metrics['trade_count'] = len(trades)
        
        # Count winning and losing trades
        win_count = sum(1 for t in trades if t.get('pnl', 0) > 0)
        loss_count = sum(1 for t in trades if t.get('pnl', 0) <= 0)
        metrics['win_count'] = win_count
        metrics['loss_count'] = loss_count
        metrics['win_rate'] = (win_count / len(trades) * 100) if trades else 0
        
        logger.info(f"Backtest completed with {len(trades)} trades")
        logger.info(f"Total return: {metrics['total_return']:.2f}%")
        logger.info(f"Win rate: {metrics['win_rate']:.2f}%")
    else:
        logger.warning("Backtest returned no equity curve or trades")
        metrics['error'] = "No results returned from backtest"
    
    return equity_curve, trades, metrics

if __name__ == "__main__":
    # Run with your test data
    symbol = "SAMPLE"
    timeframe = "1m"
    
    # Adjust strategy parameters as needed
    lookback = 20
    z_threshold = 1.5
    
    # Run debug backtest
    equity_curve, trades, metrics = run_debug_backtest(
        symbol=symbol,
        timeframe=timeframe,
        lookback=lookback,
        z_threshold=z_threshold
    )
    
    # Display results
    print("\n=== DEBUG BACKTEST RESULTS ===")
    
    if 'error' in metrics:
        print(f"Error: {metrics['error']}")
    else:
        print(f"Symbol: {symbol}")
        
        # Print manual signal verification
        print(f"\nManual Signal Verification:")
        print(f"BUY signals: {metrics['manual_signals']['buy_signals']}")
        print(f"SELL signals: {metrics['manual_signals']['sell_signals']}")
        print(f"Total signals: {metrics['manual_signals']['total_signals']}")
        
        # Print backtest results
        print(f"\nBacktest Results:")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Trade Count: {metrics['trade_count']}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Win/Loss: {metrics['win_count']}/{metrics['loss_count']}")
