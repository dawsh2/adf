# src/data/factory.py

class ComponentFactory:
    """Factory for creating system components."""
    
    def __init__(self, registry=None):
        """Initialize the factory."""
        self.registry = registry or ComponentRegistry()
    
    def create(self, component_type, name, **kwargs):
        """Create a component instance."""
        component_class = self.registry.get(name)
        if not component_class:
            raise ValueError(f"Unknown component: {name}")
            
        return component_class(**kwargs)
    
    def create_optimizer(self, name, **kwargs):
        """Create an optimizer instance."""
        optimizer_class = self.registry.get_optimizer(name)
        if not optimizer_class:
            raise ValueError(f"Unknown optimizer: {name}")
            
        return optimizer_class(**kwargs)
    
    def create_filter(self, name, **kwargs):
        """Create a filter instance."""
        filter_class = self.registry.get_filter(name)
        if not filter_class:
            raise ValueError(f"Unknown filter: {name}")
            
        return filter_class(**kwargs)
    
    def create_filter_combiner(self, name, **kwargs):
        """Create a filter combiner instance."""
        combiner_class = self.registry.get_filter_combiner(name)
        if not combiner_class:
            raise ValueError(f"Unknown filter combiner: {name}")
            
        return combiner_class(**kwargs)
    
    def register(self, name, component_class):
        """Register a component class with the registry."""
        self.registry.register(name, component_class)
