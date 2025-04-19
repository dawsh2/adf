#!/usr/bin/env python
"""
Clean Event Integration Example

This example demonstrates the proper integration of data handlers with the event system
using emitters.
"""

import os
import sys
import logging
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import core event system components
from core.events.event_types import EventType
from core.events.event_bus import EventBus
from core.events.event_manager import EventManager
from core.events.event_handlers import LoggingHandler
from core.events.event_emitters import BarEmitter  # Import our concrete implementation

# Import data module components
from data.sources.csv_handler import CSVDataSource
from data.historical_data_handler import HistoricalDataHandler

def run_example(data_dir, symbols, start_date=None, end_date=None):
    """
    Run the clean event integration example.
    
    Args:
        data_dir: Directory containing data files
        symbols: Symbols to process
        start_date: Start date
        end_date: End date
    """
    # Create the event system
    event_bus = EventBus()
    event_manager = EventManager(event_bus)
    
    # Create a bar emitter
    bar_emitter = BarEmitter("bar_emitter", event_bus)
    
    # Start the emitter
    bar_emitter.start()  # Need to start it since it's a requirement
    
    # Create monitoring handlers for demo
    bar_logger = LoggingHandler("bar_logger")
    event_bus.register(EventType.BAR, bar_logger.handle)
    
    # Create data components
    data_source = CSVDataSource(data_dir)
    data_handler = HistoricalDataHandler(
        data_source=data_source,
        bar_emitter=bar_emitter,
        max_bars_history=100
    )
    
    # Register components with event manager
    event_manager.register_component('data_handler', data_handler)
    
    # Load data
    logger.info(f"Loading data for {symbols}")
    data_handler.load_data(symbols, start_date, end_date)
    
    # Process data
    logger.info("Processing data...")
    for symbol in symbols:
        logger.info(f"Processing {symbol}")
        bar_count = 0
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
            bar_count += 1
            
            # Bar events are automatically emitted by the data handler
            
        logger.info(f"Processed {bar_count} bars for {symbol}")
    
    # Print stats
    logger.info(f"Bar events processed by logger: {bar_logger.stats['processed']}")
    logger.info(f"Bar emitter stats: {bar_emitter.stats}")
    
    # Stop the emitter
    bar_emitter.stop()


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Clean Event Integration Example')
    parser.add_argument('--data-dir', default='./data', help='Directory containing data files')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT'], help='Symbols to process')
    
    args = parser.parse_args()
    
    # Run example
    run_example(args.data_dir, args.symbols)
