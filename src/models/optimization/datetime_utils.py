"""
Datetime Utility Module

This module provides utility functions for handling date and time operations,
particularly for dealing with timezone compatibility issues in time series data.
"""
import datetime
import logging
import pytz
from typing import Tuple, Optional, Any, Union

import pandas as pd

logger = logging.getLogger(__name__)

def make_timestamps_compatible(ts1: Any, ts2: Any) -> Tuple[Any, Any]:
    """
    Make timestamps compatible for comparison by ensuring both are either
    timezone-aware or timezone-naive.
    
    Args:
        ts1: First timestamp
        ts2: Second timestamp
        
    Returns:
        tuple: (ts1_compatible, ts2_compatible)
    """
    # Check if inputs are None
    if ts1 is None or ts2 is None:
        return ts1, ts2
    
    # Check if timestamps have timezone info
    ts1_has_tz = hasattr(ts1, 'tzinfo') and ts1.tzinfo is not None
    ts2_has_tz = hasattr(ts2, 'tzinfo') and ts2.tzinfo is not None
    
    # If both have same timezone status, return as is
    if ts1_has_tz == ts2_has_tz:
        return ts1, ts2
    
    # If only one has timezone, make both naive
    if ts1_has_tz and not ts2_has_tz:
        return ts1.replace(tzinfo=None), ts2
    
    if not ts1_has_tz and ts2_has_tz:
        return ts1, ts2.replace(tzinfo=None)
    
    # Shouldn't reach here, but return original as fallback
    return ts1, ts2

def parse_timestamp(timestamp_str: str, date_format: Optional[str] = None, 
                  default_timezone: str = 'UTC') -> datetime.datetime:
    """
    Parse timestamp string to datetime object with timezone handling.
    
    Args:
        timestamp_str: Timestamp string to parse
        date_format: Optional format string for parsing
        default_timezone: Timezone to use if the parsed result has no timezone
        
    Returns:
        datetime.datetime: Parsed datetime object
    """
    try:
        # Try parsing with specified format
        if date_format:
            dt = datetime.datetime.strptime(timestamp_str, date_format)
        else:
            # Try pandas parsing which handles various formats
            dt = pd.to_datetime(timestamp_str).to_pydatetime()
        
        # Add timezone if not present
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.timezone(default_timezone))
            
        return dt
    
    except Exception as e:
        logger.error(f"Error parsing timestamp {timestamp_str}: {e}")
        # Return current time as fallback
        return datetime.datetime.now(pytz.timezone(default_timezone))

def ensure_timezone_aware(dt: datetime.datetime, 
                        default_timezone: str = 'UTC') -> datetime.datetime:
    """
    Ensure a datetime object has timezone information.
    
    Args:
        dt: Datetime object to check
        default_timezone: Timezone to use if dt has no timezone
        
    Returns:
        datetime.datetime: Timezone-aware datetime object
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=pytz.timezone(default_timezone))
    return dt

def ensure_timezone_naive(dt: datetime.datetime) -> datetime.datetime:
    """
    Ensure a datetime object has no timezone information.
    
    Args:
        dt: Datetime object to check
        
    Returns:
        datetime.datetime: Timezone-naive datetime object
    """
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt

def date_range(start_date: Union[str, datetime.datetime], 
              end_date: Union[str, datetime.datetime], 
              freq: str = 'D') -> list:
    """
    Generate a list of dates within a range.
    
    Args:
        start_date: Start date (string or datetime)
        end_date: End date (string or datetime)
        freq: Frequency of dates ('D' for daily, 'M' for monthly, etc.)
        
    Returns:
        list: List of datetime objects
    """
    # Convert string dates to datetime objects if needed
    if isinstance(start_date, str):
        start_date = parse_timestamp(start_date)
    if isinstance(end_date, str):
        end_date = parse_timestamp(end_date)
    
    # Make both dates timezone-compatible
    start_date, end_date = make_timestamps_compatible(start_date, end_date)
    
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    return dates.tolist()

def is_valid_timestamp(timestamp: Any) -> bool:
    """
    Check if an object is a valid timestamp.
    
    Args:
        timestamp: Object to check
        
    Returns:
        bool: True if valid timestamp, False otherwise
    """
    # Check if it's already a datetime
    if isinstance(timestamp, (datetime.datetime, pd.Timestamp)):
        return True
    
    # Check if it's a string that can be parsed
    if isinstance(timestamp, str):
        try:
            pd.to_datetime(timestamp)
            return True
        except:
            return False
    
    return False

def convert_to_timestamp(obj: Any) -> Optional[datetime.datetime]:
    """
    Convert various object types to datetime.
    
    Args:
        obj: Object to convert
        
    Returns:
        datetime.datetime: Converted timestamp or None if conversion fails
    """
    try:
        # Already a datetime
        if isinstance(obj, datetime.datetime):
            return obj
        
        # Pandas Timestamp
        if isinstance(obj, pd.Timestamp):
            return obj.to_pydatetime()
        
        # String
        if isinstance(obj, str):
            return pd.to_datetime(obj).to_pydatetime()
        
        # Integer (assuming Unix timestamp)
        if isinstance(obj, (int, float)):
            return datetime.datetime.fromtimestamp(obj)
        
        # Date object
        if isinstance(obj, datetime.date):
            return datetime.datetime.combine(obj, datetime.time())
        
        return None
    
    except Exception as e:
        logger.error(f"Error converting to timestamp: {e}")
        return None
