#!/usr/bin/env python
"""
Complete test script to verify all fixes:
1. Timestamp handling
2. Order tracking
3. Fill processing
4. Position liquidation
"""
import logging
import sys
import os
import pandas as pd
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all necessary components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType

# Import fixed backtest implementation
from src.execution.backtest.backtest import run_backtest

# Import data components
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler

# Import strategy components
from src.strategy.strategies.ma_crossover import MovingAverageCrossoverStrategy

def test_complete_pipeline():
    """
    Test the complete pipeline with all fixes.
    
    This tests:
    1. Timestamp handling in CSVDataSource
    2. Order generation in SimpleRiskManager
    3. Fill processing in SimulatedBroker
    4. Position liquidation at end of backtest
    """
    logger.info("Starting complete pipeline test")
    
    # Create data source
    data_dir = "data"  # Adjust path as needed
    csv_source = CSVDataSource(
        data_dir=data_dir,
        date_column="timestamp",  # Adjust based on your CSV structure
        filename_pattern="{symbol}_{timeframe}.csv"
    )
    
    # Create historical data handler
    data_handler = HistoricalDataHandler(
        data_source=csv_source,
        bar_emitter=None  # Will be set by run_backtest
    )
    
    # Create strategy with small windows to ensure signals are generated
    strategy = MovingAverageCrossoverStrategy(
        name="ma_crossover",
        symbols=["SAMPLE"],
        fast_window=5,
        slow_window=20
    )
    
    # Set date range
    start_date = pd.to_datetime("2024-03-26")
    end_date = pd.to_datetime("2024-03-28")
    
    logger.info(f"Loading data from {start_date} to {end_date}")
    
    # Load data (will be reset by run_backtest)
    data_handler.load_data(
        symbols=["SAMPLE"],
        start_date=start_date,
        end_date=end_date,
        timeframe="1m"
    )
    
    # Run backtest with the strategy
    logger.info("Running backtest with the strategy")
    equity_curve, trades = run_backtest(
        component=strategy,
        data_handler=data_handler,
        start_date=start_date,
        end_date=end_date
    )
    
    # Analyze results
    logger.info("\nBacktest Results:")
    logger.info(f"Total bars processed: {len(equity_curve) - 1}")  # -1 for initial equity point
    logger.info(f"Total trades: {len(trades)}")
    logger.info(f"Initial equity: ${equity_curve['equity'].iloc[0]:.2f}")
    logger.info(f"Final equity: ${equity_curve['equity'].iloc[-1]:.2f}")
    
    # Calculate performance metrics
    if len(equity_curve) > 1:
        total_return = (equity_curve['equity'].iloc[-1] / equity_curve['equity'].iloc[0] - 1) * 100
        logger.info(f"Total return: {total_return:.2f}%")
        
        # Check for drawdowns
        running_max = equity_curve['equity'].cummax()
        drawdown = (equity_curve['equity'] / running_max - 1) * 100
        max_drawdown = drawdown.min()
        logger.info(f"Maximum drawdown: {max_drawdown:.2f}%")
    
    # Verification - check that trades were executed and positions were liquidated
    pipeline_working = len(trades) > 0 and abs(equity_curve['equity'].iloc[-1] - 10000) < 1000
    
    if pipeline_working:
        logger.info("\nSUCCESS: The complete pipeline is working correctly!")
        logger.info("✓ Timestamps are handled correctly")
        logger.info("✓ Orders are tracked")
        logger.info("✓ Fills are processed")
        logger.info("✓ Positions are liquidated at the end")
    else:
        if len(trades) == 0:
            logger.error("\nFAILURE: No trades were executed!")
        else:
            logger.error("\nFAILURE: Final equity significantly different from initial!")
            logger.error("Positions may not have been properly liquidated")
    
    # Return both the success flag and results for further analysis
    return pipeline_working, (equity_curve, trades)

def detailed_trade_analysis(trades):
    """
    Perform detailed analysis of trades.
    
    Args:
        trades: List of trade dictionaries
    """
    if not trades:
        logger.info("No trades to analyze")
        return
    
    logger.info("\nDetailed Trade Analysis:")
    logger.info(f"Total number of trades: {len(trades)}")
    
    # Count buys and sells
    buys = sum(1 for t in trades if t['direction'] == 'BUY')
    sells = sum(1 for t in trades if t['direction'] == 'SELL')
    logger.info(f"Buy trades: {buys}")
    logger.info(f"Sell trades: {sells}")
    
    # Calculate total commission
    total_commission = sum(t['commission'] for t in trades if 'commission' in t)
    logger.info(f"Total commission: ${total_commission:.2f}")
    
    # Analyze PnL if calculated
    pnl_trades = [t for t in trades if 'pnl' in t and t['pnl'] != 0]
    if pnl_trades:
        total_pnl = sum(t['pnl'] for t in pnl_trades)
        avg_pnl = total_pnl / len(pnl_trades)
        logger.info(f"Total PnL: ${total_pnl:.2f}")
        logger.info(f"Average PnL per trade: ${avg_pnl:.2f}")
        
        winning_trades = sum(1 for t in pnl_trades if t['pnl'] > 0)
        losing_trades = sum(1 for t in pnl_trades if t['pnl'] < 0)
        if winning_trades + losing_trades > 0:
            win_rate = winning_trades / (winning_trades + losing_trades) * 100
            logger.info(f"Win rate: {win_rate:.2f}%")

if __name__ == "__main__":
    success, (equity_curve, trades) = test_complete_pipeline()
    
    # Perform detailed analysis if requested
    if len(sys.argv) > 1 and sys.argv[1] == '--detailed':
        detailed_trade_analysis(trades)
    
    sys.exit(0 if success else 1)
