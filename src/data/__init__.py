"""
Data handling module for the algorithmic trading system.
"""
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Import core classes
from .data_source_base import DataSourceBase
from .data_handler_base import DataHandlerBase
from .registry import ComponentRegistry
from .factory import ComponentFactory
from .historical_data_handler import HistoricalDataHandler

# Import data sources
from .sources.csv_handler import CSVDataSource

# Import transformers
from .transformers.resampler import Resampler
from .transformers.normalizer import Normalizer

# Create default registry
default_registry = ComponentRegistry()

# Register data sources
default_registry.register('csv', CSVDataSource)

# Register data handlers
default_registry.register('historical', HistoricalDataHandler)

# Create default factory
default_factory = ComponentFactory(default_registry)

__all__ = [
    'DataSourceBase',
    'DataHandlerBase',
    'ComponentRegistry',
    'ComponentFactory',
    'HistoricalDataHandler',
    'CSVDataSource',
    'Resampler',
    'Normalizer',
    'default_registry',
    'default_factory'
]
