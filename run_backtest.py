import datetime
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import os

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import core components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType
from src.core.events.event_emitters import BarEmitter

# Import data components
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler

# Import execution components
from src.execution.brokers.simulated import SimulatedBroker
from src.execution.execution_base import ExecutionEngine
from src.execution.portfolio import PortfolioManager

# Import risk components
from src.strategy.risk.position_sizer import PositionSizer
from src.strategy.risk.risk_manager import RiskManager

# Import strategy
from ma_crossover_strategy import MovingAverageCrossoverStrategy

# Import backtest runner
from backtest_runner import BacktestRunner

def run_backtest(data_dir='./data', symbols=None, start_date='2024-03-25', end_date='2024-04-05',
                timeframe='1m'):
    """
    Run a moving average crossover backtest.
    
    Args:
        data_dir: Directory for data files
        symbols: List of symbols to trade
        start_date: Start date for the backtest
        end_date: End date for the backtest
        timeframe: Timeframe for the data ('1d', '1h', '1m', etc.)
        
    Returns:
        dict: Backtest results
    """
    if symbols is None:
        symbols = ['SPY']
    
    logger.info(f"Starting backtest for {symbols} from {start_date} to {end_date}")
    
    # Create event system
    event_bus = EventBus()
    event_manager = EventManager(event_bus)
    
    # Create data components with custom configuration for your data format
    data_source = CSVDataSource(
        data_dir=data_dir,
        filename_pattern='{symbol}_{timeframe}.csv',
        date_column='timestamp',  # Your date column is called 'timestamp'
        column_map={
            'open': ['Open'],     # Map 'Open' to 'open'
            'high': ['High'],     # Map 'High' to 'high'
            'low': ['Low'],       # Map 'Low' to 'low'
            'close': ['Close'],   # Map 'Close' to 'close'
            'volume': ['Volume']  # Map 'Volume' to 'volume'
        }
    )
    
    # Check if data exists
    for symbol in symbols:
        filename = f"{data_dir}/{symbol}_{timeframe}.csv"
        if not os.path.exists(filename):
            logger.error(f"Data file not found: {filename}")
            return None, None
    
    # Create bar emitter
    bar_emitter = BarEmitter("backtest_bar_emitter", event_bus)
    bar_emitter.start()
    
    # Create data handler with bar emitter
    data_handler = HistoricalDataHandler(data_source, bar_emitter)
    
    # Create portfolio components
    initial_capital = 100000.0
    portfolio = PortfolioManager(initial_cash=initial_capital)
    
    # Create risk components
    position_sizer = PositionSizer(method='percent_equity', params={'percent': 0.1})
    risk_manager = RiskManager(
        portfolio=portfolio,
        position_sizer=position_sizer,
        risk_limits={
            'max_position_size': 1000,
            'max_exposure': 0.3,
            'min_trade_size': 10
        }
    )
    
    # Create execution components
    broker = SimulatedBroker()
    execution_engine = ExecutionEngine(broker_interface=broker)
    
    # Create strategy with adjusted parameters for minute data
    # For 1-minute data, a 10-period MA is only 10 minutes, which might be too short
    # For 1-minute data, more appropriate windows might be 20 and 50
    strategy = MovingAverageCrossoverStrategy(
        name="ma_crossover",
        symbols=symbols,
        fast_window=20,   # 20 minutes for fast MA
        slow_window=50    # 50 minutes for slow MA
    )
    
    # Create backtest runner
    backtest = BacktestRunner(
        data_handler=data_handler,
        strategy=strategy,
        risk_manager=risk_manager,
        execution_engine=execution_engine,
        portfolio=portfolio,
        event_bus=event_bus,
        event_manager=event_manager
    )
    
    # Run backtest
    results = backtest.run(symbols, start_date, end_date, timeframe)
    
    # Display results
    if results:
        print("\nBacktest Results:")
        print(f"Initial Capital: ${results['initial_equity']:,.2f}")
        print(f"Final Equity: ${results['final_equity']:,.2f}")
        print(f"Return: {results['return_pct']:.2f}%")
        print(f"Maximum Drawdown: {results['max_drawdown_pct']:.2f}%")
        print(f"Number of Trades: {results['trade_count']}")
        
        # Plot equity curve
        plot_results(results)
    
    return results, backtest

def plot_results(results):
    """Plot backtest results."""
    equity_curve = results['equity_curve']
    
    plt.figure(figsize=(12, 8))
    
    # Equity curve
    plt.subplot(2, 1, 1)
    plt.plot(equity_curve.index, equity_curve['equity'])
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    
    # Drawdown
    plt.subplot(2, 1, 2)
    plt.plot(equity_curve.index, equity_curve['drawdown'] * 100)
    plt.title('Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run a backtest with SPY data
    results, backtest = run_backtest(
        symbols=['SPY'],
        start_date='2024-03-25',
        end_date='2024-04-05',
        timeframe='1m'
    )
