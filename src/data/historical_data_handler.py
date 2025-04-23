"""
Handler for historical data.
"""
import datetime
import pandas as pd
import logging
from typing import Dict, List, Optional, Union, Any, Deque
from collections import deque

from src.core.events.event_types import BarEvent
from src.core.events.event_utils import create_bar_event
from .data_handler_base import DataHandlerBase
from .data_source_base import DataSourceBase

logger = logging.getLogger(__name__)

class HistoricalDataHandler(DataHandlerBase):
    """Handler for historical data."""
    
    def __init__(self, data_source: DataSourceBase, bar_emitter, max_bars_history: int = 100):
        """
        Initialize the historical data handler.
        
        Args:
            data_source: Data source to use
            bar_emitter: Emitter for bar events
            max_bars_history: Maximum number of bars to keep in history
        """
        super().__init__(bar_emitter)
        self.data_source = data_source
        self.data_frames = {}  # symbol -> DataFrame
        self.current_idx = {}  # symbol -> current index
        self.bars_history = {}  # symbol -> deque of BarEvents
        self.max_bars_history = max_bars_history
    
    def load_data(self, symbols: Union[str, List[str]], start_date=None, 
                end_date=None, timeframe='1m') -> None:
        """Load data for the specified symbols."""
        # Convert single symbol to list
        if isinstance(symbols, str):
            symbols = [symbols]
            
        # Load data for each symbol
        for symbol in symbols:
            try:
                df = self.data_source.get_data(symbol, start_date, end_date, timeframe)
                if not df.empty:
                    # Ensure DataFrame has proper columns and types
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    if not all(col in df.columns for col in required_cols):
                        missing = [col for col in required_cols if col not in df.columns]
                        logger.warning(f"Symbol {symbol} missing columns: {missing}")
                        continue
                        
                    # Convert numeric columns to float
                    for col in ['open', 'high', 'low', 'close']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                    # Convert volume to int
                    if 'volume' in df.columns:
                        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
                    
                    # Store data
                    self.data_frames[symbol] = df
                    self.current_idx[symbol] = 0
                    self.bars_history[symbol] = deque(maxlen=self.max_bars_history)
                    
                    logger.info(f"Loaded {len(df)} bars for {symbol}")
                else:
                    logger.warning(f"No data found for {symbol}")
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
    
    def get_next_bar(self, symbol: str) -> Optional[BarEvent]:
        """Get the next bar for a symbol."""
        # Check if we have data for this symbol
        if symbol not in self.data_frames:
            logger.warning(f"No data loaded for symbol: {symbol}")
            return None
            
        df = self.data_frames[symbol]
        idx = self.current_idx[symbol]
        
        # Check if we've reached the end of the data
        if idx >= len(df):
            logger.debug(f"End of data reached for {symbol}")
            return None
            
        # Get the row
        row = df.iloc[idx]
        timestamp = row.name
        
        if not isinstance(timestamp, pd.Timestamp) and not isinstance(timestamp, datetime.datetime):
            try:
                timestamp = pd.to_datetime(timestamp)
            except Exception as e:
                logger.error(f"Invalid timestamp for {symbol} at index {idx}: {e}")
                self.current_idx[symbol] = idx + 1
                return None
        
        # Create bar event
        try:
            # Extract data from row
            open_price = float(row['open'])
            high_price = float(row['high'])
            low_price = float(row['low'])
            close_price = float(row['close'])
            volume = int(row['volume'])
            
            # Create bar event
            bar = create_bar_event(
                symbol=symbol,
                timestamp=timestamp,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price,
                volume=volume
            )
            
            # Store in history
            self.bars_history[symbol].append(bar)
            
            # Increment index
            self.current_idx[symbol] = idx + 1
            
            # Emit the bar event
            if self.bar_emitter:
                self.bar_emitter.emit(bar)
            
            return bar
        except Exception as e:
            logger.error(f"Error creating bar event for {symbol} at index {idx}: {e}")
            self.current_idx[symbol] = idx + 1
            return None
    
    def reset(self) -> None:
        """Reset the data handler state."""
        self.current_idx = {symbol: 0 for symbol in self.data_frames}
        self.bars_history = {symbol: deque(maxlen=self.max_bars_history) 
                            for symbol in self.data_frames}
    
    def get_symbols(self) -> List[str]:
        """Get the list of available symbols."""
        return list(self.data_frames.keys())
    
    def get_latest_bar(self, symbol: str) -> Optional[BarEvent]:
        """Get the latest bar for a symbol."""
        if symbol not in self.bars_history or not self.bars_history[symbol]:
            return None
        return self.bars_history[symbol][-1]
    
    def get_latest_bars(self, symbol: str, N: int = 1) -> List[BarEvent]:
        """Get the last N bars for a symbol."""
        if symbol not in self.bars_history:
            return []
            
        bars = list(self.bars_history[symbol])
        return bars[-N:] if N < len(bars) else bars
