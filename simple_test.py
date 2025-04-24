#!/usr/bin/env python3
"""
Simplified validation script - core functionality only.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data():
    """Generate synthetic price data with clear trading signals."""
    # Create a date range for 100 days
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(100)]
    
    # Create a price series with multiple crossovers
    base_price = 100
    prices = []
    
    # Create alternating up and down trends to ensure crossovers
    for i in range(100):
        # Create 4 distinct trend periods
        if i < 25:
            # Uptrend
            trend = 0.5 * i
        elif i < 50:
            # Downtrend
            trend = 0.5 * (50 - i)
        elif i < 75:
            # Another uptrend
            trend = 0.5 * (i - 50)
        else:
            # Another downtrend
            trend = 0.5 * (100 - i)
            
        # Add oscillation to create crossovers
        oscillation = 5 * np.sin(i/3)
        
        # Combine base, trend and oscillation
        price = base_price + trend + oscillation
        prices.append(price)
    
    # Create DataFrame with OHLCV data
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.01 for p in prices],  # 1% higher than price
        'low': [p * 0.99 for p in prices],   # 1% lower than price
        'close': prices,
        'volume': 1000000 * np.ones(len(prices))  # Constant volume
    })
    
    # Ensure directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    df.to_csv('data/SYNTHETIC_1d.csv', index=False)
    print(f"Generated synthetic data with {len(df)} rows")
    
    return df

def manual_backtest_calculation(data, fast_window=5, slow_window=15):
    """
    Manually calculate expected performance using the MA crossover rule.
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Calculate moving averages
    df['fast_ma'] = df['close'].rolling(window=fast_window).mean()
    df['slow_ma'] = df['close'].rolling(window=slow_window).mean()
    
    # Calculate signals
    df['signal'] = 0
    df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1  # Buy
    df.loc[df['fast_ma'] < df['slow_ma'], 'signal'] = -1  # Sell
    
    # Calculate position changes
    df['signal_change'] = df['signal'].diff()
    
    # Initialize portfolio tracking
    initial_cash = 10000.0
    position_size = 100
    cash = initial_cash
    position = 0
    trades = []
    portfolio_values = [initial_cash]
    
    # Process each bar
    for i in range(1, len(df)):
        # Skip initial bars until we have both MAs
        if pd.isna(df['fast_ma'].iloc[i]) or pd.isna(df['slow_ma'].iloc[i]):
            portfolio_values.append(cash)
            continue
        
        # Check for signal change
        if df['signal_change'].iloc[i] != 0:
            price = df['close'].iloc[i]
            
            # Buy signal (0->1 or -1->1)
            if df['signal'].iloc[i] == 1 and df['signal'].iloc[i-1] != 1:
                # Calculate trade size
                shares_to_buy = position_size
                trade_value = shares_to_buy * price
                
                # Execute buy
                if cash >= trade_value:
                    cash -= trade_value
                    position += shares_to_buy
                    trades.append({
                        'date': df['timestamp'].iloc[i],
                        'action': 'BUY',
                        'price': price,
                        'shares': shares_to_buy,
                        'value': trade_value
                    })
                    print(f"BUY: {shares_to_buy} shares at ${price:.2f} on {df['timestamp'].iloc[i]}")
            
            # Sell signal (0->-1 or 1->-1)
            elif df['signal'].iloc[i] == -1 and df['signal'].iloc[i-1] != -1:
                # Calculate trade size
                shares_to_sell = min(position_size, position)
                
                # Execute sell
                if shares_to_sell > 0:
                    trade_value = shares_to_sell * price
                    cash += trade_value
                    position -= shares_to_sell
                    trades.append({
                        'date': df['timestamp'].iloc[i],
                        'action': 'SELL',
                        'price': price,
                        'shares': shares_to_sell,
                        'value': trade_value
                    })
                    print(f"SELL: {shares_to_sell} shares at ${price:.2f} on {df['timestamp'].iloc[i]}")
        
        # Calculate portfolio value
        portfolio_value = cash + (position * df['close'].iloc[i])
        portfolio_values.append(portfolio_value)
    
    # Liquidate final position
    if position > 0:
        final_price = df['close'].iloc[-1]
        trade_value = position * final_price
        cash += trade_value
        trades.append({
            'date': df['timestamp'].iloc[-1],
            'action': 'LIQUIDATE',
            'price': final_price,
            'shares': position,
            'value': trade_value
        })
        print(f"LIQUIDATE: {position} shares at ${final_price:.2f} on {df['timestamp'].iloc[-1]}")
    
    # Calculate final portfolio value
    final_value = cash
    
    # Calculate returns
    total_return = (final_value / initial_cash - 1) * 100
    
    # Create equity curve
    equity_curve = pd.DataFrame({
        'timestamp': df['timestamp'],
        'equity': portfolio_values
    })
    
    results = {
        'initial_cash': initial_cash,
        'final_value': final_value,
        'total_return': total_return,
        'trade_count': len(trades),
        'trades': trades,
        'equity_curve': equity_curve
    }
    
    return results

def main():
    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data()
    
    # Define strategy parameters
    fast_window = 5
    slow_window = 15
    
    # Run manual backtest calculation
    logger.info(f"Running manual backtest calculation with {fast_window}/{slow_window} MA parameters...")
    manual_results = manual_backtest_calculation(synthetic_data, fast_window, slow_window)
    
    print("\n===== MANUAL BACKTEST RESULTS =====")
    print(f"Initial Cash: ${manual_results['initial_cash']:.2f}")
    print(f"Final Value: ${manual_results['final_value']:.2f}")
    print(f"Total Return: {manual_results['total_return']:.2f}%")
    print(f"Number of Trades: {manual_results['trade_count']}")
    
    # Save results for reference
    manual_results['equity_curve'].to_csv('manual_equity_curve.csv', index=False)
    print("Saved manual equity curve to 'manual_equity_curve.csv'")

if __name__ == "__main__":
    main()
