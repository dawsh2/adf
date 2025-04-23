"""
Datetime Utility Module

This module provides utility functions to handle timezone issues consistently.
"""
import datetime
import pandas as pd
import pytz
from typing import Tuple, Optional, Union, Any, List

def standardize_dataframe_tz(df: pd.DataFrame, use_utc: bool = True) -> pd.DataFrame:
    """
    Ensure DataFrame has consistent timezone handling.
    
    Args:
        df: DataFrame with datetime index
        use_utc: If True, convert to UTC; if False, make timezone-naive
        
    Returns:
        DataFrame with standardized timezone
    """
    if df.index.dtype.kind != 'M':  # Not a datetime index
        return df
    
    df_copy = df.copy()
    
    if use_utc:
        # Convert index to UTC timezone
        if df.index.tz is None:
            # If naive, localize to UTC
            df_copy.index = df_copy.index.tz_localize('UTC')
        else:
            # If already has timezone, convert to UTC
            df_copy.index = df_copy.index.tz_convert('UTC')
    else:
        # Make timezone-naive by removing timezone info
        if df.index.tz is not None:
            # Remove timezone info
            df_copy.index = df_copy.index.tz_localize(None)
    
    return df_copy

def parse_timestamp(timestamp_str: str, default_tz: Optional[str] = None) -> datetime.datetime:
    """
    Parse timestamp string to datetime with consistent timezone handling.
    
    Args:
        timestamp_str: String timestamp to parse
        default_tz: Timezone to use if string has no timezone (None for naive)
        
    Returns:
        Datetime object with consistent timezone handling
    """
    # Parse the timestamp
    dt = pd.to_datetime(timestamp_str)
    
    # Apply timezone handling
    if default_tz:
        if dt.tzinfo is None:
            # Localize to the specified timezone
            dt = dt.tz_localize(default_tz)
    else:
        # Make timezone-naive
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
    
    return dt

def make_timestamps_compatible(ts1: datetime.datetime, ts2: datetime.datetime) -> Tuple[datetime.datetime, datetime.datetime]:
    """
    Make two timestamps compatible for comparison by standardizing timezone info.
    
    Args:
        ts1: First timestamp
        ts2: Second timestamp
        
    Returns:
        Tuple of compatible timestamps
    """
    # If both have timezone info or both don't, they're compatible
    ts1_has_tz = ts1.tzinfo is not None
    ts2_has_tz = ts2.tzinfo is not None
    
    if ts1_has_tz == ts2_has_tz:
        return ts1, ts2
    
    # If only one has timezone, standardize by making both naive
    if ts1_has_tz:
        ts1 = ts1.replace(tzinfo=None)
    else:
        ts2 = ts2.replace(tzinfo=None)
    
    return ts1, ts2

def read_csv_with_timezone(file_path: str, timestamp_col: str = 'timestamp', 
                         tz_handling: str = 'utc') -> pd.DataFrame:
    """
    Read CSV with proper timezone handling.
    
    Args:
        file_path: Path to CSV file
        timestamp_col: Name of timestamp column
        tz_handling: How to handle timezones ('utc', 'naive', or 'original')
        
    Returns:
        DataFrame with standardized timestamps
    """
    # Read the CSV
    df = pd.read_csv(file_path)
    
    # Convert timestamp column
    if timestamp_col in df.columns:
        # Parse timestamps with UTC to avoid mixing warning
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=(tz_handling == 'utc'))
        
        # Set as index
        df.set_index(timestamp_col, inplace=True)
        
        # Apply timezone handling
        if tz_handling == 'utc':
            # Already handled by utc=True in pd.to_datetime
            pass
        elif tz_handling == 'naive':
            # Remove timezone info
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
        # 'original' keeps whatever timezone was in the data
    
    return df

def get_date_range(start_date: Union[str, datetime.datetime], 
                  end_date: Union[str, datetime.datetime],
                  freq: str = 'D') -> List[datetime.datetime]:
    """
    Get a list of dates in a range with consistent timezone handling.
    
    Args:
        start_date: Start date
        end_date: End date
        freq: Frequency ('D' for daily, etc.)
        
    Returns:
        List of datetime objects
    """
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Make timezone info consistent
    start_date, end_date = make_timestamps_compatible(start_date, end_date)
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    return date_range.tolist()

def compare_datetimes(dt1: datetime.datetime, dt2: datetime.datetime) -> int:
    """
    Compare two datetime objects safely, handling timezone differences.
    
    Args:
        dt1: First datetime
        dt2: Second datetime
        
    Returns:
        -1 if dt1 < dt2, 0 if equal, 1 if dt1 > dt2
    """
    dt1, dt2 = make_timestamps_compatible(dt1, dt2)
    
    if dt1 < dt2:
        return -1
    elif dt1 > dt2:
        return 1
    else:
        return 0

# """
# Datetime Utility Module

# This module provides utility functions for handling date and time operations,
# particularly for dealing with timezone compatibility issues in time series data.
# """
# import datetime
# import logging
# import pytz
# from typing import Tuple, Optional, Any, Union

# import pandas as pd

# logger = logging.getLogger(__name__)

# def make_timestamps_compatible(ts1: Any, ts2: Any) -> Tuple[Any, Any]:
#     """
#     Make timestamps compatible for comparison by ensuring both are either
#     timezone-aware or timezone-naive.
    
#     Args:
#         ts1: First timestamp
#         ts2: Second timestamp
        
#     Returns:
#         tuple: (ts1_compatible, ts2_compatible)
#     """
#     # Check if inputs are None
#     if ts1 is None or ts2 is None:
#         return ts1, ts2
    
#     # Check if timestamps have timezone info
#     ts1_has_tz = hasattr(ts1, 'tzinfo') and ts1.tzinfo is not None
#     ts2_has_tz = hasattr(ts2, 'tzinfo') and ts2.tzinfo is not None
    
#     # If both have same timezone status, return as is
#     if ts1_has_tz == ts2_has_tz:
#         return ts1, ts2
    
#     # If only one has timezone, make both naive
#     if ts1_has_tz and not ts2_has_tz:
#         return ts1.replace(tzinfo=None), ts2
    
#     if not ts1_has_tz and ts2_has_tz:
#         return ts1, ts2.replace(tzinfo=None)
    
#     # Shouldn't reach here, but return original as fallback
#     return ts1, ts2

# def parse_timestamp(timestamp_str: str, date_format: Optional[str] = None, 
#                   default_timezone: str = 'UTC') -> datetime.datetime:
#     """
#     Parse timestamp string to datetime object with timezone handling.
    
#     Args:
#         timestamp_str: Timestamp string to parse
#         date_format: Optional format string for parsing
#         default_timezone: Timezone to use if the parsed result has no timezone
        
#     Returns:
#         datetime.datetime: Parsed datetime object
#     """
#     try:
#         # Try parsing with specified format
#         if date_format:
#             dt = datetime.datetime.strptime(timestamp_str, date_format)
#         else:
#             # Try pandas parsing which handles various formats
#             dt = pd.to_datetime(timestamp_str).to_pydatetime()
        
#         # Add timezone if not present
#         if dt.tzinfo is None:
#             dt = dt.replace(tzinfo=pytz.timezone(default_timezone))
            
#         return dt
    
#     except Exception as e:
#         logger.error(f"Error parsing timestamp {timestamp_str}: {e}")
#         # Return current time as fallback
#         return datetime.datetime.now(pytz.timezone(default_timezone))

# def ensure_timezone_aware(dt: datetime.datetime, 
#                         default_timezone: str = 'UTC') -> datetime.datetime:
#     """
#     Ensure a datetime object has timezone information.
    
#     Args:
#         dt: Datetime object to check
#         default_timezone: Timezone to use if dt has no timezone
        
#     Returns:
#         datetime.datetime: Timezone-aware datetime object
#     """
#     if dt.tzinfo is None:
#         return dt.replace(tzinfo=pytz.timezone(default_timezone))
#     return dt

# def ensure_timezone_naive(dt: datetime.datetime) -> datetime.datetime:
#     """
#     Ensure a datetime object has no timezone information.
    
#     Args:
#         dt: Datetime object to check
        
#     Returns:
#         datetime.datetime: Timezone-naive datetime object
#     """
#     if dt.tzinfo is not None:
#         return dt.replace(tzinfo=None)
#     return dt

# def date_range(start_date: Union[str, datetime.datetime], 
#               end_date: Union[str, datetime.datetime], 
#               freq: str = 'D') -> list:
#     """
#     Generate a list of dates within a range.
    
#     Args:
#         start_date: Start date (string or datetime)
#         end_date: End date (string or datetime)
#         freq: Frequency of dates ('D' for daily, 'M' for monthly, etc.)
        
#     Returns:
#         list: List of datetime objects
#     """
#     # Convert string dates to datetime objects if needed
#     if isinstance(start_date, str):
#         start_date = parse_timestamp(start_date)
#     if isinstance(end_date, str):
#         end_date = parse_timestamp(end_date)
    
#     # Make both dates timezone-compatible
#     start_date, end_date = make_timestamps_compatible(start_date, end_date)
    
#     # Generate date range
#     dates = pd.date_range(start=start_date, end=end_date, freq=freq)
#     return dates.tolist()

# def is_valid_timestamp(timestamp: Any) -> bool:
#     """
#     Check if an object is a valid timestamp.
    
#     Args:
#         timestamp: Object to check
        
#     Returns:
#         bool: True if valid timestamp, False otherwise
#     """
#     # Check if it's already a datetime
#     if isinstance(timestamp, (datetime.datetime, pd.Timestamp)):
#         return True
    
#     # Check if it's a string that can be parsed
#     if isinstance(timestamp, str):
#         try:
#             pd.to_datetime(timestamp)
#             return True
#         except:
#             return False
    
#     return False

# def convert_to_timestamp(obj: Any) -> Optional[datetime.datetime]:
#     """
#     Convert various object types to datetime.
    
#     Args:
#         obj: Object to convert
        
#     Returns:
#         datetime.datetime: Converted timestamp or None if conversion fails
#     """
#     try:
#         # Already a datetime
#         if isinstance(obj, datetime.datetime):
#             return obj
        
#         # Pandas Timestamp
#         if isinstance(obj, pd.Timestamp):
#             return obj.to_pydatetime()
        
#         # String
#         if isinstance(obj, str):
#             return pd.to_datetime(obj).to_pydatetime()
        
#         # Integer (assuming Unix timestamp)
#         if isinstance(obj, (int, float)):
#             return datetime.datetime.fromtimestamp(obj)
        
#         # Date object
#         if isinstance(obj, datetime.date):
#             return datetime.datetime.combine(obj, datetime.time())
        
#         return None
    
#     except Exception as e:
#         logger.error(f"Error converting to timestamp: {e}")
#         return None
