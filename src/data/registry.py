"""
Registry for data components.
"""
from typing import Dict, Type, Any, Optional

class ComponentRegistry:
    """Registry for data components."""
    
    def __init__(self):
        """Initialize an empty registry."""
        self.components = {}
    
    def register(self, name: str, component_class: Type) -> None:
        """
        Register a component class.
        
        Args:
            name: Component name
            component_class: Component class
        """
        self.components[name] = component_class
    
    def get(self, name: str) -> Optional[Type]:
        """
        Get a component class by name.
        
        Args:
            name: Component name
            
        Returns:
            Component class or None if not found
        """
        return self.components.get(name)
    
    def list(self) -> list:
        """
        List all registered components.
        
        Returns:
            List of component names
        """
        return list(self.components.keys())
    
    def __contains__(self, name: str) -> bool:
        """
        Check if a component is registered.
        
        Args:
            name: Component name
            
        Returns:
            True if registered, False otherwise
        """
        return name in self.components
