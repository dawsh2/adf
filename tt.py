#!/usr/bin/env python
"""
Script to test timestamp handling in the data pipeline
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

# Import necessary components
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler
from src.core.events.event_emitters import BarEmitter
from src.core.events.event_bus import EventBus

def test_timestamp_handling():
    """Test that timestamps are properly handled throughout the data pipeline"""
    logger.info("Starting timestamp handling test")
    
    # Create event bus
    event_bus = EventBus()
    
    # Create bar emitter
    bar_emitter = BarEmitter(name="test_emitter", event_bus=event_bus)
    bar_emitter.start()
    
    # Create data source
    data_dir = "data"  # Adjust path as needed
    csv_source = CSVDataSource(
        data_dir=data_dir,
        date_column="timestamp",  # Adjust based on your CSV structure
        filename_pattern="{symbol}_{timeframe}.csv"
    )
    
    # Create data handler
    data_handler = HistoricalDataHandler(
        data_source=csv_source,
        bar_emitter=bar_emitter
    )
    
    # Set test dates
    start_date = pd.to_datetime("2024-03-26")
    end_date = pd.to_datetime("2024-04-26")
    
    logger.info(f"Loading data from {start_date} to {end_date}")
    
    # Load sample data
    symbol = "SAMPLE"
    data_handler.load_data(
        symbols=[symbol],
        start_date=start_date,
        end_date=end_date,
        timeframe="1m"
    )
    
    # Track timestamps
    timestamps = []
    bar_count = 0
    
    # Process bars and collect timestamps
    logger.info("Processing bars and collecting timestamps")
    data_handler.reset()
    
    while True:
        bar = data_handler.get_next_bar(symbol)
        if bar is None:
            break
            
        bar_count += 1
        timestamp = bar.get_timestamp()
        timestamps.append(timestamp)
        
        # Log some details
        if bar_count <= 5 or bar_count % 100 == 0:
            logger.info(f"Bar {bar_count}: timestamp={timestamp}, type={type(timestamp)}")
    
    # Analyze timestamps
    if timestamps:
        min_date = min(timestamps)
        max_date = max(timestamps)
        logger.info(f"Processed {bar_count} bars with timestamps from {min_date} to {max_date}")
        
        # Check if timestamps are within expected range
        if min_date >= start_date and max_date <= end_date:
            logger.info("PASSED: All timestamps are within the expected date range")
        else:
            logger.warning(f"FAILED: Timestamps outside expected range: min={min_date}, max={max_date}")
            logger.warning(f"Expected range: start={start_date}, end={end_date}")
    else:
        logger.error("No bars processed")
    
    return bar_count, timestamps

if __name__ == "__main__":
    bar_count, timestamps = test_timestamp_handling()
    
    # Print summary
    print("\nTest Summary:")
    print(f"Processed {bar_count} bars")
    
    if bar_count > 0:
        print(f"First timestamp: {timestamps[0]}")
        print(f"Last timestamp: {timestamps[-1]}")
        
        # Check for issues with Unix epoch timestamps (1970)
        epoch_count = sum(1 for ts in timestamps if ts.year < 2000)
        if epoch_count > 0:
            print(f"WARNING: Found {epoch_count} timestamps with year < 2000")
        else:
            print("All timestamps look good!")
