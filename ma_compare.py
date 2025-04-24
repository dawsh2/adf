# Test each parameter combination individually with proper data handling
import os
import pandas as pd
import logging
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler
from src.core.events.event_bus import EventBus
from src.core.events.event_emitters import BarEmitter
from src.strategy.strategies.ma_crossover import MovingAverageCrossoverStrategy
from src.execution.backtest.backtest import run_backtest
from src.analytics.performance import PerformanceAnalytics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define data path
data_path = 'data/SAMPLE_1m.csv'  # Replace with your actual data file path
data_dir = os.path.dirname(data_path)
filename = os.path.basename(data_path)

# Extract symbol from filename (assuming format SYMBOL_timeframe.csv)
symbol = filename.split('_')[0]

# Set up data source
data_source = CSVDataSource(
    data_dir=data_dir, 
    filename_pattern=filename  # Use exact filename
)

# Set up event bus and bar emitter
event_bus = EventBus()
bar_emitter = BarEmitter(name="bar_emitter", event_bus=event_bus)
bar_emitter.start()

# Create data handler
data_handler = HistoricalDataHandler(
    data_source=data_source,
    bar_emitter=bar_emitter
)

# Load data
data_handler.load_data(symbols=[symbol])

# Verify data was loaded successfully
if symbol not in data_handler.data_frames:
    raise ValueError(f"Failed to load data for {symbol}")

# Print data information
df = data_handler.data_frames[symbol]
logger.info(f"Loaded {len(df)} bars for {symbol} from {df.index.min()} to {df.index.max()}")

# Initial portfolio cash
initial_cash = 10000.0

# Test each parameter combination individually
results = []

for fast_window in [2, 3, 5, 8, 13, 21]:
    for slow_window in [34, 55, 89, 100]:
        if fast_window >= slow_window:
            continue  # Skip invalid combinations
            
        logger.info(f"\nTesting MA({fast_window}, {slow_window})")
        
        # Create strategy with these parameters
        strategy = MovingAverageCrossoverStrategy(
            name=f"ma_{fast_window}_{slow_window}",
            symbols=[symbol],
            fast_window=fast_window,
            slow_window=slow_window
        )
        
        # Reset data handler before each test
        data_handler.reset()
        
        # Run backtest directly
        equity_curve, trades = run_backtest(
            component=strategy,
            data_handler=data_handler,
            initial_cash=initial_cash,
            debug=True  # Enable debug logging
        )
        
        # Calculate metrics
        metrics = PerformanceAnalytics.calculate_metrics(equity_curve, trades)
        
        # Store results
        result = {
            'fast_window': fast_window,
            'slow_window': slow_window,
            'trade_count': len(trades),
            'return': metrics.get('total_return', 0),
            'sharpe': metrics.get('sharpe_ratio', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'equity_final': metrics.get('final_equity', initial_cash)
        }
        results.append(result)
        
        # Print detailed results
        logger.info(f"Trades: {len(trades)}")
        logger.info(f"Return: {result['return']:.2f}%")
        logger.info(f"Sharpe: {result['sharpe']:.2f}")
        logger.info(f"Max Drawdown: {result['max_drawdown']:.2f}%")
        
        # Print first few trades for debugging
        if trades:
            logger.info("First 3 trades:")
            for i, trade in enumerate(trades[:3]):
                logger.info(f"  Trade {i+1}: {trade}")

# Sort and display results
results.sort(key=lambda x: x['return'], reverse=True)
logger.info("\nTop 5 parameter combinations by return:")
for r in results[:5]:
    logger.info(f"MA({r['fast_window']}, {r['slow_window']}): "
               f"Return: {r['return']:.2f}%, "
               f"Sharpe: {r['sharpe']:.2f}, "
               f"Trades: {r['trade_count']}, "
               f"Max DD: {r['max_drawdown']:.2f}%")

# Also output best by Sharpe ratio
results.sort(key=lambda x: x['sharpe'], reverse=True)
logger.info("\nTop 5 parameter combinations by Sharpe ratio:")
for r in results[:5]:
    logger.info(f"MA({r['fast_window']}, {r['slow_window']}): "
               f"Sharpe: {r['sharpe']:.2f}, "
               f"Return: {r['return']:.2f}%, "
               f"Trades: {r['trade_count']}, "
               f"Max DD: {r['max_drawdown']:.2f}%")

# For comparison, create a buy-and-hold benchmark
from src.core.events.event_types import Event, EventType, SignalEvent, BarEvent
from src.core.events.event_utils import create_signal_event

class BuyAndHoldStrategy:
    """Simple buy-and-hold strategy for benchmark comparison."""
    
    def __init__(self, name="buy_and_hold", symbols=None):
        self.name = name
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.bought = {symbol: False for symbol in self.symbols}
        self.event_bus = None
    
    def set_event_bus(self, event_bus):
        self.event_bus = event_bus
        return self
    
    def on_bar(self, event):
        if not isinstance(event, BarEvent):
            return None
            
        symbol = event.get_symbol()
        
        if symbol not in self.symbols:
            return None
            
        # Buy once at the beginning and hold
        if not self.bought[symbol]:
            # Create buy signal
            signal = create_signal_event(
                signal_value=SignalEvent.BUY,
                price=event.get_close(),
                symbol=symbol,
                rule_id=self.name,
                timestamp=event.get_timestamp()
            )
            
            # Mark as bought
            self.bought[symbol] = True
            
            # Emit if we have an event bus
            if self.event_bus:
                self.event_bus.emit(signal)
                
            return signal
            
        return None
    
    def reset(self):
        self.bought = {symbol: False for symbol in self.symbols}
    
    def get_parameters(self):
        return {}
    
    def set_parameters(self, params):
        pass

# Run buy-and-hold benchmark
benchmark = BuyAndHoldStrategy(name="buy_and_hold", symbols=[symbol])
data_handler.reset()

equity_curve_bh, trades_bh = run_backtest(
    component=benchmark,
    data_handler=data_handler,
    initial_cash=initial_cash
)

metrics_bh = PerformanceAnalytics.calculate_metrics(equity_curve_bh, trades_bh)

logger.info("\nBuy and Hold Benchmark:")
logger.info(f"Return: {metrics_bh.get('total_return', 0):.2f}%")
logger.info(f"Sharpe: {metrics_bh.get('sharpe_ratio', 0):.2f}")
logger.info(f"Max Drawdown: {metrics_bh.get('max_drawdown', 0):.2f}%")
logger.info(f"Trades: {len(trades_bh)}")

# Compare with inverse signals for one promising parameter set
if results:
    # Get the best parameter set based on return
    best_params = results[0]
    fast = best_params['fast_window']
    slow = best_params['slow_window']
    
    logger.info(f"\nTesting inverse signals for MA({fast}, {slow})")
    
    # Create special inverse strategy class
    class InverseMACrossoverStrategy(MovingAverageCrossoverStrategy):
        """MA Crossover with inverted signals."""
        
        def _create_signal(self, symbol, signal_type, price, timestamp=None):
            # Invert the signal: BUY → SELL, SELL → BUY
            if signal_type == SignalEvent.BUY:
                inverted_signal_type = SignalEvent.SELL
            elif signal_type == SignalEvent.SELL:
                inverted_signal_type = SignalEvent.BUY
            else:
                inverted_signal_type = signal_type
                
            # Call parent method with inverted signal
            return super()._create_signal(symbol, inverted_signal_type, price, timestamp)
    
    # Create inverse strategy
    inverse_strategy = InverseMACrossoverStrategy(
        name=f"inverse_ma_{fast}_{slow}",
        symbols=[symbol],
        fast_window=fast,
        slow_window=slow
    )
    
    # Reset data handler
    data_handler.reset()
    
    # Run backtest with inverse strategy
    equity_curve_inv, trades_inv = run_backtest(
        component=inverse_strategy,
        data_handler=data_handler,
        initial_cash=initial_cash
    )
    
    # Calculate metrics
    metrics_inv = PerformanceAnalytics.calculate_metrics(equity_curve_inv, trades_inv)
    
    # Print results
    logger.info("\nInverse MA Crossover results:")
    logger.info(f"Return: {metrics_inv.get('total_return', 0):.2f}%")
    logger.info(f"Sharpe: {metrics_inv.get('sharpe_ratio', 0):.2f}")
    logger.info(f"Max Drawdown: {metrics_inv.get('max_drawdown', 0):.2f}%")
    logger.info(f"Trades: {len(trades_inv)}")
    
    # Compare with normal signals
    logger.info("\nComparison - Normal vs Inverse signals:")
    logger.info(f"Normal - Return: {best_params['return']:.2f}%, Sharpe: {best_params['sharpe']:.2f}")
    logger.info(f"Inverse - Return: {metrics_inv.get('total_return', 0):.2f}%, Sharpe: {metrics_inv.get('sharpe_ratio', 0):.2f}")
