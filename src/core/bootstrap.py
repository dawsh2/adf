# src/core/bootstrap.py

from core.config.config import Config
from core.di.container import Container
from core.events.event_bus import EventBus
# Import other components as needed

def bootstrap(config_files=None):
    """Bootstrap the application."""
    # Initialize configuration
    config = Config()
    
    # Register defaults
    register_default_configs(config)
    
    # Load configuration files
    if config_files:
        for file in config_files:
            config.load_file(file)
    
    # Load environment variables
    config.load_env(prefix='TRADING_')
    
    # Initialize container
    container = Container()
    
    # Register core components
    register_core_components(container, config)
    
    # Register data components
    register_data_components(container, config)
    
    # Register strategy components
    register_strategy_components(container, config)
    
    # Register execution components
    register_execution_components(container, config)
    
    return container, config

def register_default_configs(config):
    """Register default configurations."""
    config.register_defaults('core', {
        'log_level': 'INFO',
    })
    
    config.register_defaults('data', {
        'data_dir': './data',
        'date_format': '%Y-%m-%d',
    })
    
    # Add other default configurations
    
def register_core_components(container, config):
    """Register core components."""
    container.register_instance('config', config)
    container.register('event_bus', EventBus)
    # Register other core components

def register_data_components(container, config):
    """Register data components."""
    # Register data components with their dependencies
    # ...

# Define other registration functions
