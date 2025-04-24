"""
Integrated Regime Filter Test

This script runs an integrated test of the regime filter implementation with 
proper trade tracking to validate P&L calculation.
"""
import logging
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any

# Import core components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType, BarEvent, SignalEvent, OrderEvent
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler

# Import the strategies
from src.strategy.strategies.mean_reversion import MeanReversionStrategy
from src.models.filters.regime.simple import SimpleRegimeFilteredStrategy  # Our new implementation

# Import execution components
from src.execution.portfolio import PortfolioManager
from src.execution.brokers.simulated import SimulatedBroker
from src.strategy.risk.risk_manager import SimplePassthroughRiskManager

# Import trade tracking from validation code
from src.execution.trade_tracker import TradeTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_integrated_test(data_path, output_dir=None, initial_cash=10000.0):
    """
    Run an integrated test of regime filtering with proper trade tracking.
    
    Args:
        data_path: Path to price data CSV file
        output_dir: Directory for saving results
        initial_cash: Initial portfolio cash
        
    Returns:
        dict: Test results
    """
    # Set output directory
    if output_dir is None:
        output_dir = os.getcwd()
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract symbol from filename
    filename = os.path.basename(data_path)
    symbol = filename.split('_')[0]  # Assumes format SYMBOL_timeframe.csv
    
    # Set up data handler
    data_dir = os.path.dirname(data_path)
    
    # Create data source
    data_source = CSVDataSource(
        data_dir=data_dir, 
        filename_pattern=filename  # Use exact filename
    )
    
    # Create bar emitter
    event_bus = EventBus()
    from src.core.events.event_emitters import BarEmitter
    bar_emitter = BarEmitter(name="bar_emitter", event_bus=event_bus)
    bar_emitter.start()
    
    # Create data handler
    data_handler = HistoricalDataHandler(
        data_source=data_source,
        bar_emitter=bar_emitter
    )
    
    # Load data
    data_handler.load_data(symbols=[symbol])
    
    # Check if data was loaded
    if symbol not in data_handler.data_frames:
        raise ValueError(f"Failed to load data for {symbol}")
    
    # Log data summary
    df = data_handler.data_frames[symbol]
    logger.info(f"Loaded {len(df)} bars for {symbol} from {df.index.min()} to {df.index.max()}")
    
    # Set strategy parameters
    lookback = 20
    z_threshold = 1.5
    ma_window = 50  # For regime detection
    
    # Create base strategy for reference
    base_strategy = MeanReversionStrategy(
        name="mean_reversion",
        symbols=[symbol],
        lookback=lookback,
        z_threshold=z_threshold
    )
    
    # Create regime-filtered strategy
    regime_strategy = SimpleRegimeFilteredStrategy(
        base_strategy=MeanReversionStrategy(
            name="mean_reversion",
            symbols=[symbol],
            lookback=lookback,
            z_threshold=z_threshold
        ),
        ma_window=ma_window
    )
    
    # Set up event system
    event_manager = EventManager(event_bus)
    
    # Create trade tracker for accurate P&L tracking
    tracker = TradeTracker(initial_cash=initial_cash)
    
    # Create execution components
    portfolio = PortfolioManager(initial_cash=initial_cash)
    portfolio.set_event_bus(event_bus)
    
    broker = SimulatedBroker(fill_emitter=event_bus)
    broker.set_event_bus(event_bus)
    
    risk_manager = SimplePassthroughRiskManager(
        portfolio=portfolio,
        event_bus=event_bus
    )
    risk_manager.broker = broker
    
    # Reset and connect strategy
    regime_strategy.reset()
    regime_strategy.set_event_bus(event_bus)
    
    # Event tracking
    events = {
        'bar': 0,
        'signal': 0,
        'order': 0,
        'fill': 0
    }
    
    # Track market data for position valuation
    market_data = {}
    
    # Event handlers
    def on_bar(event):
        """Track bar events and update market data"""
        events['bar'] += 1
        symbol = event.get_symbol()
        price = event.get_close()
        timestamp = event.get_timestamp()
        
        # Update market data
        market_data[symbol] = price
        
        # Update broker's market data
        broker.update_market_data(symbol, {
            'price': price,
            'timestamp': timestamp
        })
        
        # Update trade tracker's equity
        tracker.update_equity(timestamp, market_data)
        
        # Log periodic updates
        if events['bar'] % 100 == 0:
            logger.info(f"Processing bar {events['bar']}: {symbol} @ {price:.2f}")
    
    def on_signal(event):
        """Track signal events"""
        events['signal'] += 1
        symbol = event.get_symbol()
        signal_value = event.get_signal_value()
        signal_name = "BUY" if signal_value == SignalEvent.BUY else "SELL" if signal_value == SignalEvent.SELL else "NEUTRAL"
        price = event.get_price()
        logger.debug(f"Signal: {symbol} {signal_name} @ {price:.2f}")
    
    def on_order(event):
        """Track order events"""
        events['order'] += 1
        symbol = event.get_symbol()
        direction = event.get_direction()
        quantity = event.get_quantity()
        price = event.get_price()
        logger.debug(f"Order: {symbol} {direction} {quantity} @ {price:.2f}")
    
    def on_fill(event):
        """Process fill events with trade tracker"""
        events['fill'] += 1
        symbol = event.get_symbol()
        direction = event.get_direction()
        quantity = event.get_quantity()
        price = event.get_price()
        commission = event.get_commission() if hasattr(event, 'get_commission') else 0.0
        timestamp = event.get_timestamp()
        
        logger.info(f"Fill: {symbol} {direction} {quantity} @ {price:.2f}")
        
        # Process with trade tracker
        fill_data = {
            'timestamp': timestamp,
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'price': price,
            'commission': commission
        }
        pnl = tracker.process_fill(fill_data)
        
        # Log P&L if significant
        if abs(pnl) > 0.01:
            logger.info(f"Trade P&L: {pnl:.2f}")
    
    # Register event handlers
    event_bus.register(EventType.BAR, on_bar)
    event_bus.register(EventType.SIGNAL, on_signal)
    event_bus.register(EventType.ORDER, on_order)
    event_bus.register(EventType.FILL, on_fill)
    
    # Register system components
    event_manager.register_component('strategy', regime_strategy, [EventType.BAR])
    event_manager.register_component('risk_manager', risk_manager, [EventType.SIGNAL])
    event_manager.register_component('portfolio', portfolio, [EventType.FILL])
    event_bus.register(EventType.ORDER, broker.place_order)
    
    # Reset data handler
    data_handler.reset()
    
    # Process each bar
    logger.info("Starting backtest with regime filtering...")
    
    last_bar = None
    
    while True:
        # Get next bar
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break
            
        # Keep reference to the last valid bar
        last_bar = bar
            
        # Process bar through event system
        event_bus.emit(bar)
    
    # Use the last bar's timestamp for liquidation
    last_timestamp = last_bar.get_timestamp() if last_bar else datetime.now()
    
    # Liquidate all positions
    logger.info("Liquidating all positions...")
    liquidation_pnl = tracker.liquidate_positions(last_timestamp, market_data)
    logger.info(f"Liquidation P&L: {liquidation_pnl:.2f}")
    
    # Get equity curve and trades
    equity_curve = tracker.get_equity_curve()
    trades = tracker.get_closed_trades()
    
    # Get trade statistics
    trade_stats = tracker.get_trade_statistics()
    logger.info(f"Trade Statistics: {trade_stats}")
    
    # Get regime filter stats
    regime_stats = regime_strategy.get_regime_stats()
    logger.info(f"Regime Filter Stats:")
    logger.info(f"- Signals Passed: {regime_stats['passed_signals']}")
    logger.info(f"- Signals Filtered: {regime_stats['filtered_signals']}")
    
    # Calculate filter rate
    if regime_stats['passed_signals'] + regime_stats['filtered_signals'] > 0:
        filter_rate = regime_stats['filtered_signals'] / (regime_stats['passed_signals'] + regime_stats['filtered_signals']) * 100
        logger.info(f"- Filter Rate: {filter_rate:.2f}%")
    else:
        filter_rate = 0
        logger.info("- Filter Rate: N/A (no signals generated)")
    
    # Create performance plot
    create_performance_plot(equity_curve, trades, regime_stats, symbol, output_dir)
    
    # Return results
    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'trade_stats': trade_stats,
        'regime_stats': regime_stats,
        'events': events,
        'filter_rate': filter_rate
    }

def create_performance_plot(equity_curve, trades, regime_stats, symbol, output_dir):
    """
    Create a performance plot with regime information.
    
    Args:
        equity_curve: DataFrame with equity curve data
        trades: List of completed trades
        regime_stats: Dictionary with regime filtering statistics
        symbol: Symbol traded
        output_dir: Directory to save plot
    """
    # Create figure with subplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot equity curve
    ax1.plot(equity_curve['timestamp'], equity_curve['equity'], 
            label='Regime-Filtered Strategy', color='green', linewidth=1.5)
    
    # Add initial cash line
    initial_cash = equity_curve['equity'].iloc[0] if not equity_curve.empty else 10000.0
    ax1.axhline(y=initial_cash, color='gray', linestyle='--', alpha=0.6, label='Initial Cash')
    
    # Add trades as markers
    buy_timestamps = [t['entry_time'] for t in trades if t['entry_direction'] == 'BUY']
    sell_timestamps = [t['entry_time'] for t in trades if t['entry_direction'] == 'SELL']
    
    # Get equity values at those timestamps
    buy_equities = []
    for ts in buy_timestamps:
        idx = equity_curve['timestamp'].searchsorted(ts)
        if idx < len(equity_curve):
            buy_equities.append(equity_curve['equity'].iloc[idx])
        else:
            buy_equities.append(None)
            
    buy_equities = [e for e in buy_equities if e is not None]
    buy_timestamps = buy_timestamps[:len(buy_equities)]
    
    sell_equities = []
    for ts in sell_timestamps:
        idx = equity_curve['timestamp'].searchsorted(ts)
        if idx < len(equity_curve):
            sell_equities.append(equity_curve['equity'].iloc[idx])
        else:
            sell_equities.append(None)
            
    sell_equities = [e for e in sell_equities if e is not None]
    sell_timestamps = sell_timestamps[:len(sell_equities)]
    
    # Plot trade markers
    if buy_timestamps:
        ax1.scatter(buy_timestamps, buy_equities, color='green', marker='^', 
                  s=50, alpha=0.7, label='Buy Entry')
                  
    if sell_timestamps:
        ax1.scatter(sell_timestamps, sell_equities, color='red', marker='v', 
                  s=50, alpha=0.7, label='Sell Entry')
    
    # Configure first subplot
    ax1.set_title(f'Regime-Filtered Strategy Performance: {symbol}', fontsize=14)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Calculate drawdown
    equity_values = equity_curve['equity'].values
    rolling_max = np.maximum.accumulate(equity_values)
    drawdown = (equity_values / rolling_max - 1) * 100
    
    # Plot drawdown
    ax2.fill_between(equity_curve['timestamp'], 0, drawdown, color='red', alpha=0.3)
    
    # Configure second subplot
    ax2.set_title('Drawdown', fontsize=14)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Format dates
    import matplotlib.dates as mdates
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax1.xaxis.set_major_formatter(date_format)
    ax2.xaxis.set_major_formatter(date_format)
    
    # Rotate date labels - FIXED: use get_xticklabels() instead of xticklabels()
    plt.setp(ax1.get_xticklabels(), rotation=45)
    plt.setp(ax2.get_xticklabels(), rotation=45)
    
    # Add performance statistics
    initial_equity = equity_curve['equity'].iloc[0] if not equity_curve.empty else initial_cash
    final_equity = equity_curve['equity'].iloc[-1] if not equity_curve.empty else initial_cash
    total_return = ((final_equity / initial_equity) - 1) * 100
    max_dd = drawdown.min() if len(drawdown) > 0 else 0
    
    # Add regime filter stats
    passed = regime_stats['passed_signals']
    filtered = regime_stats['filtered_signals']
    total = passed + filtered
    filter_rate = filtered / total * 100 if total > 0 else 0
    
    stats_text = (
        f"Total Return: {total_return:.2f}%\n"
        f"Max Drawdown: {max_dd:.2f}%\n"
        f"Trades: {len(trades)}\n\n"
        f"Regime Filter Stats:\n"
        f"Signals Passed: {passed}\n"
        f"Signals Filtered: {filtered}\n"
        f"Filter Rate: {filter_rate:.2f}%"
    )
    
    # Add text box with statistics
    ax1.text(0.01, 0.01, stats_text, transform=ax1.transAxes, fontsize=10,
           bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{symbol}_regime_filtered_performance.png"), dpi=150)
    logger.info(f"Saved performance chart to {os.path.join(output_dir, f'{symbol}_regime_filtered_performance.png')}")
    
    plt.close(fig)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Integrated Regime Filter Test')
    parser.add_argument('--data', type=str, required=True, help='Path to price data CSV file')
    parser.add_argument('--output', type=str, default=None, help='Directory to save results')
    parser.add_argument('--cash', type=float, default=10000.0, help='Initial portfolio cash')
    
    args = parser.parse_args()
    
    results = run_integrated_test(
        data_path=args.data,
        output_dir=args.output,
        initial_cash=args.cash
    )
    
    # Print summary
    print("\n=== INTEGRATED REGIME FILTER TEST RESULTS ===")
    
    # Performance summary
    initial_equity = results['equity_curve']['equity'].iloc[0]
    final_equity = results['equity_curve']['equity'].iloc[-1]
    total_return = ((final_equity / initial_equity) - 1) * 100
    
    print(f"Performance Summary:")
    print(f"- Total Return: {total_return:.2f}%")
    print(f"- Trade Count: {len(results['trades'])}")
    print(f"- Win Rate: {results['trade_stats']['win_rate']:.2f}%")
    
    # Regime filter stats
    print(f"\nRegime Filter Stats:")
    print(f"- Signals Passed: {results['regime_stats']['passed_signals']}")
    print(f"- Signals Filtered: {results['regime_stats']['filtered_signals']}")
    print(f"- Filter Rate: {results['filter_rate']:.2f}%")
    
    print("\nDetailed results have been saved, including performance chart.")

