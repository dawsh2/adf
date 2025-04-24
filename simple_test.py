#!/usr/bin/env python3
"""
Simplified MA strategy test with appropriate parameters
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Import system components
from src.core.events.event_bus import EventBus
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler
from src.strategy.strategies.ma_crossover import MovingAverageCrossoverStrategy
from src.execution.backtest.backtest import run_backtest

def main():
    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Set up paths
    data_dir = "data"
    sample_file = "SAMPLE_1m.csv"  # Daily data file
    
    # Check if file exists
    sample_path = os.path.join(data_dir, sample_file)
    if not os.path.exists(sample_path):
        logger.error(f"Data file not found: {sample_path}")
        return
    
    # First, let's examine the data to understand its timeframe
    df = pd.read_csv(sample_path)
    logger.info(f"Data loaded. Shape: {df.shape}")
    
    # Determine the date column
    date_col = next((col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()), None)
    if not date_col:
        logger.error("Could not identify date column")
        return
        
    # Convert to datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Print data range
    date_range = df[date_col].max() - df[date_col].min()
    logger.info(f"Data spans from {df[date_col].min()} to {df[date_col].max()}")
    logger.info(f"Total timespan: {date_range}")
    
    # Determine appropriate MA parameters based on data timeframe
    # If we have short timeframe data, use shorter MAs
    if date_range.days < 60:  # Less than 2 months
        fast_window = 5    # Use 5-period MA instead of 10
        slow_window = 15   # Use 15-period MA instead of 30
        logger.info(f"Short timeframe detected. Using fast_window={fast_window}, slow_window={slow_window}")
    else:
        fast_window = 10   # Standard parameters
        slow_window = 30
        logger.info(f"Standard timeframe. Using fast_window={fast_window}, slow_window={slow_window}")
    
    # Set up for backtest
    event_bus = EventBus()
    
    class BarEmitter:
        def __init__(self, event_bus):
            self.event_bus = event_bus
        def emit(self, event):
            self.event_bus.emit(event)
    
    bar_emitter = BarEmitter(event_bus)
    
    # Initialize data components
    csv_source = CSVDataSource(data_dir=data_dir)
    data_handler = HistoricalDataHandler(csv_source, bar_emitter)
    
    # Load data
    data_handler.load_data(symbols='SAMPLE', timeframe='1m')
    
    # Create strategy with adaptive parameters
    strategy = MovingAverageCrossoverStrategy(
        name="ma_crossover_adaptive",
        symbols=['SAMPLE'],
        fast_window=fast_window,  # Adaptive fast MA
        slow_window=slow_window,  # Adaptive slow MA
        price_key='close'
    )
    strategy.set_event_bus(event_bus)
    
    # Run backtest
    logger.info("Running backtest with adaptive parameters...")
    equity_curve, trades = run_backtest(
        component=strategy,
        data_handler=data_handler,
        initial_cash=10000.0,
        position_size=100
    )
    
    # Simple analysis
    if equity_curve is not None and not equity_curve.empty:
        initial_equity = equity_curve['equity'].iloc[0]
        final_equity = equity_curve['equity'].iloc[-1]
        total_return = ((final_equity / initial_equity) - 1) * 100
        
        print("\n===== Strategy Results =====")
        print(f"Initial Equity: ${initial_equity:.2f}")
        print(f"Final Equity: ${final_equity:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Number of Trades: {len(trades)}")
        
        # Plot results
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(equity_curve['timestamp'], equity_curve['equity'])
        plt.title('Equity Curve - Adaptive MA Parameters')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        
        # Save equity curve to CSV for reference
        equity_curve.to_csv('equity_curve_adaptive.csv', index=False)
        
        # Save plot
        plt.tight_layout()
        plt.savefig('equity_curve_adaptive.png')
        print("Saved results to 'equity_curve_adaptive.png' and 'equity_curve_adaptive.csv'")
    else:
        logger.error("Backtest did not produce valid results")

if __name__ == "__main__":
    main()
