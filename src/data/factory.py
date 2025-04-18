"""
Factory for creating data components.
"""
from typing import Dict, Any, Optional, Type

from .registry import ComponentRegistry

class ComponentFactory:
    """Factory for creating data components."""
    
    def __init__(self, registry: Optional[ComponentRegistry] = None):
        """
        Initialize the factory.
        
        Args:
            registry: Optional component registry
        """
        self.registry = registry or ComponentRegistry()
    
    def create(self, component_type: str, name: str, **kwargs) -> Any:
        """
        Create a component instance.
        
        Args:
            component_type: Type of component (e.g., 'source', 'handler')
            name: Name of the component class
            **kwargs: Parameters to pass to the component constructor
            
        Returns:
            Component instance
            
        Raises:
            ValueError: If component not found in registry
        """
        component_class = self.registry.get(name)
        if not component_class:
            raise ValueError(f"Unknown component: {name}")
            
        return component_class(**kwargs)
    
    def register(self, name: str, component_class: Type) -> None:
        """
        Register a component class with the registry.
        
        Args:
            name: Component name
            component_class: Component class
        """
        self.registry.register(name, component_class)
