from typing import Dict, Any, Optional, Callable, Type, TypeVar, List, Set, Union

T = TypeVar('T')

class Container:
    """Dependency injection container."""
    
    def __init__(self):
        self._components = {}  # name -> component info
        self._instances = {}   # name -> singleton instance
        self._factories = {}   # name -> factory function
    
    def register(self, name: str, component_class: Type[T], 
                dependencies: Dict[str, str] = None, singleton: bool = True) -> 'Container':
        """Register a component with the container."""
        self._components[name] = {
            'class': component_class,
            'dependencies': dependencies or {},
            'singleton': singleton
        }
        return self
    
    def register_instance(self, name: str, instance: Any) -> 'Container':
        """Register a pre-created instance."""
        self._instances[name] = instance
        return self
    
    def register_factory(self, name: str, factory: Callable[['Container'], T]) -> 'Container':
        """Register a factory function for complex initialization."""
        self._factories[name] = factory
        return self
    
    def get(self, name: str) -> Any:
        """Get a component instance by name."""
        # Check if it's already instantiated (singleton case)
        if name in self._instances:
            return self._instances[name]
        
        # Check if it has a factory
        if name in self._factories:
            instance = self._factories[name](self)
            if self._components.get(name, {}).get('singleton', True):
                self._instances[name] = instance
            return instance
        
        # Get component info
        if name not in self._components:
            raise ValueError(f"Component not registered: {name}")
            
        component_info = self._components[name]
        
        # Resolve dependencies
        resolved_deps = {}
        for dep_name, dep_key in component_info['dependencies'].items():
            resolved_deps[dep_name] = self.get(dep_key)
            
        # Create instance
        instance = component_info['class'](**resolved_deps)
        
        # Store if singleton
        if component_info['singleton']:
            self._instances[name] = instance
            
        return instance
    
    def inject(self, instance: Any) -> Any:
        """Inject dependencies into an existing instance."""
        # Find setter methods (set_X)
        for name in self._components:
            setter_name = f"set_{name}"
            if hasattr(instance, setter_name) and callable(getattr(instance, setter_name)):
                # Get dependency and inject
                dependency = self.get(name)
                getattr(instance, setter_name)(dependency)
        
        return instance
    
    def reset(self) -> None:
        """Reset all singleton instances."""
        self._instances.clear()
