"""
Regime Filter Diagnostic Test

This script provides diagnostic tools to verify that signal filtering
is working correctly in the algorithmic trading system.
"""
import logging
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Import core components
from src.core.events.event_bus import EventBus
from src.core.events.event_manager import EventManager
from src.core.events.event_types import EventType, BarEvent, SignalEvent, OrderEvent
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler

# Import strategies
from src.strategy.strategies.mean_reversion import MeanReversionStrategy

# Import execution components
from src.execution.backtest.backtest import run_backtest
from src.analytics.performance import PerformanceAnalytics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DiagnosticRegimeStrategy:
    """Test strategy that filters ALL signals with detailed logging."""
    
    def __init__(self, base_strategy):
        self.strategy = base_strategy
        self.name = f"diagnostic_{self.strategy.name}"
        self.symbols = self.strategy.symbols if hasattr(self.strategy, 'symbols') else []
        self.filtered_signals = 0
        self.passed_signals = 0
        self.event_bus = None
        self.debug = True
        
        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.DiagnosticStrategy")
        self.logger.setLevel(logging.DEBUG)
        
        print("Diagnostic Strategy initialized - FILTERING ALL SIGNALS")
    
    def set_event_bus(self, event_bus):
        self.event_bus = event_bus
        if hasattr(self.strategy, 'set_event_bus'):
            self.strategy.set_event_bus(None)  # Prevent base strategy from emitting directly
        return self
    
    def on_bar(self, event):
        # Get signal from base strategy
        signal = self.strategy.on_bar(event)
        
        # Log the signal
        if signal:
            symbol = signal.get_symbol()
            signal_value = signal.get_signal_value()
            signal_name = "BUY" if signal_value == 1 else "SELL" if signal_value == -1 else "NEUTRAL"
            price = signal.get_price()
            
            self.logger.info(f"FILTERING signal: {symbol} {signal_name} @ {price:.2f} (#{self.filtered_signals + 1})")
            self.filtered_signals += 1
            
            # Always return None to filter ALL signals
            return None
        return None
    
    def reset(self):
        if hasattr(self.strategy, 'reset'):
            self.strategy.reset()
        self.filtered_signals = 0
        self.passed_signals = 0
        
    def get_parameters(self):
        return self.strategy.get_parameters() if hasattr(self.strategy, 'get_parameters') else {}
    
    def set_parameters(self, params):
        if hasattr(self.strategy, 'set_parameters'):
            self.strategy.set_parameters(params)
            
    def get_regime_stats(self):
        return {
            'filtered_signals': self.filtered_signals,
            'passed_signals': self.passed_signals
        }

def run_diagnostic_test(data_path, output_dir=None, initial_cash=10000.0):
    """
    Run a diagnostic test to verify signal filtering is working properly.
    
    This test will:
    1. Create a strategy that filters ALL signals
    2. Run a backtest and verify no trades are executed
    3. Log each step of the signal → order → fill process
    
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
    
    # Create base strategy
    base_strategy = MeanReversionStrategy(
        name="mean_reversion",
        symbols=[symbol],
        lookback=lookback,
        z_threshold=z_threshold
    )
    
    # Create diagnostic strategy that filters ALL signals
    diagnostic_strategy = DiagnosticRegimeStrategy(base_strategy)
    
    # Create event tracker for detailed event monitoring
    from src.core.events.event_utils import EventTracker
    event_tracker = EventTracker(name="diagnostic_tracker", verbose=True)
    
    # Register event tracker to monitor all events
    event_bus.register(EventType.BAR, event_tracker.track_event)
    event_bus.register(EventType.SIGNAL, event_tracker.track_event)
    event_bus.register(EventType.ORDER, event_tracker.track_event)
    event_bus.register(EventType.FILL, event_tracker.track_event)
    
    # Add a direct event handler to log all events for debugging
    def log_all_events(event):
        event_type = event.get_type()
        if event_type == EventType.SIGNAL:
            symbol = event.get_symbol()
            signal_value = event.get_signal_value()
            signal_name = "BUY" if signal_value == 1 else "SELL" if signal_value == -1 else "NEUTRAL"
            logger.debug(f"SIGNAL EVENT: {symbol} {signal_name}")
        elif event_type == EventType.ORDER:
            symbol = event.get_symbol()
            direction = event.get_direction()
            quantity = event.get_quantity()
            logger.debug(f"ORDER EVENT: {symbol} {direction} {quantity}")
        elif event_type == EventType.FILL:
            symbol = event.get_symbol()
            direction = event.get_direction()
            quantity = event.get_quantity()
            logger.debug(f"FILL EVENT: {symbol} {direction} {quantity}")
    
    # Register direct event logging
    event_bus.register(EventType.SIGNAL, log_all_events)
    event_bus.register(EventType.ORDER, log_all_events)
    event_bus.register(EventType.FILL, log_all_events)
    
    # Run backtest with the diagnostic strategy
    logger.info("Running diagnostic backtest with ALL signals filtered...")
    
    equity_curve, trades = run_backtest(
        component=diagnostic_strategy,
        data_handler=data_handler,
        initial_cash=initial_cash,
        debug=True  # Enable debug mode for more verbose output
    )
    
    # Verify results
    logger.info("\n===== DIAGNOSTIC TEST RESULTS =====")
    logger.info(f"Filtered Signals: {diagnostic_strategy.filtered_signals}")
    logger.info(f"Trades Executed: {len(trades)}")
    
    # Check if filtering worked correctly
    if diagnostic_strategy.filtered_signals > 0 and len(trades) == 0:
        logger.info("SUCCESS: Signal filtering is working correctly!")
        logger.info("Signals were filtered and no trades were executed.")
    else:
        if diagnostic_strategy.filtered_signals == 0:
            logger.error("ERROR: Base strategy did not generate any signals.")
        else:
            logger.error("ERROR: Trades were executed despite signals being filtered!")
            logger.error("This indicates a problem in the signal -> order -> fill chain.")
            logger.error(f"Filtered signals: {diagnostic_strategy.filtered_signals}, Trades: {len(trades)}")
    
    # Get event statistics
    event_summary = event_tracker.get_summary()
    logger.info("\nEvent Summary:")
    for event_type, count in event_summary.items():
        logger.info(f"{event_type}: {count}")
    
    # Return diagnostic results
    return {
        'filtered_signals': diagnostic_strategy.filtered_signals,
        'trades': trades,
        'equity_curve': equity_curve,
        'event_summary': event_summary
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Regime Filter Diagnostic Test')
    parser.add_argument('--data', type=str, required=True, help='Path to price data CSV file')
    parser.add_argument('--output', type=str, default=None, help='Directory to save results')
    parser.add_argument('--cash', type=float, default=10000.0, help='Initial portfolio cash')
    
    args = parser.parse_args()
    
    results = run_diagnostic_test(
        data_path=args.data,
        output_dir=args.output,
        initial_cash=args.cash
    )
    
    # Print summary
    print("\n=== DIAGNOSTIC TEST SUMMARY ===")
    print(f"Signals generated and filtered: {results['filtered_signals']}")
    print(f"Trades executed: {len(results['trades'])}")
    
    if results['filtered_signals'] > 0 and len(results['trades']) == 0:
        print("\nDIAGNOSIS: Signal filtering is working correctly.")
        print("The issue may be in how your SimpleRegimeFilteredStrategy is implemented.")
    elif results['filtered_signals'] == 0:
        print("\nDIAGNOSIS: Base strategy did not generate any signals.")
        print("Check your data and strategy parameters.")
    else:
        print("\nDIAGNOSIS: Trades were executed despite signals being filtered!")
        print("This indicates a problem in the signal -> order -> fill chain.")
        print("Check how the backtest system processes filtered signals.")
