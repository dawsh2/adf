# src/core/di/__init__.py

from .container import Container

# Create a default container instance
default_container = Container()

# Export classes and instance
__all__ = ['Container', 'default_container']
