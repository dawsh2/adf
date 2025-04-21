# src/data/registry.py

class ComponentRegistry:
    """Registry for system components."""
    
    def __init__(self):
        """Initialize an empty registry."""
        self.components = {}
        self.optimizers = {}
        self.filters = {}
        self.filter_combiners = {}
    
    def register(self, name, component_class):
        """Register a component class."""
        self.components[name] = component_class
    
    def register_optimizer(self, name, optimizer_class):
        """Register an optimizer class."""
        self.optimizers[name] = optimizer_class
    
    def register_filter(self, name, filter_class):
        """Register a filter class."""
        self.filters[name] = filter_class
    
    def register_filter_combiner(self, name, combiner_class):
        """Register a filter combiner class."""
        self.filter_combiners[name] = combiner_class
    
    def get(self, name):
        """Get a component class by name."""
        return self.components.get(name)
    
    def get_optimizer(self, name):
        """Get an optimizer class by name."""
        return self.optimizers.get(name)
    
    def get_filter(self, name):
        """Get a filter class by name."""
        return self.filters.get(name)
    
    def get_filter_combiner(self, name):
        """Get a filter combiner class by name."""
        return self.filter_combiners.get(name)
    
    def list(self):
        """List all registered components."""
        return list(self.components.keys())
    
    def list_optimizers(self):
        """List all registered optimizers."""
        return list(self.optimizers.keys())
    
    def list_filters(self):
        """List all registered filters."""
        return list(self.filters.keys())
    
    def list_filter_combiners(self):
        """List all registered filter combiners."""
        return list(self.filter_combiners.keys())
