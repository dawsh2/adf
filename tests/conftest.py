import os
import sys
import importlib.util
import importlib.machinery
import types

# Get the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add src directory to Python path
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# Add project root to Python path
sys.path.insert(0, project_root)

# Add integration dir
sys.path.insert(0, os.path.join(project_root, 'tests/integration'))

# Create proper module aliases for 'core' and 'data'
def create_module_alias(source_module, alias_name):
    """Create a properly initialized module alias in sys.modules."""
    if source_module in sys.modules:
        # The source module is already imported, use it
        sys.modules[alias_name] = sys.modules[source_module]
    else:
        # Try to find and load the source module
        try:
            # Import the source module
            imported_module = importlib.import_module(source_module)
            # Create the alias
            sys.modules[alias_name] = imported_module
            
            # Also create parent modules if needed
            parts = alias_name.split('.')
            for i in range(1, len(parts)):
                parent_name = '.'.join(parts[:i])
                if parent_name not in sys.modules:
                    parent_module = type(sys)(parent_name)
                    setattr(parent_module, parts[i-1], sys.modules['.'.join(parts[:i+1])])
                    sys.modules[parent_name] = parent_module
                    
        except ImportError:
            # If source module can't be imported yet, create an empty module
            module = type(sys)(alias_name)
            sys.modules[alias_name] = module

# Create aliases for main modules and their submodules
create_module_alias('src.core', 'core')
create_module_alias('src.core.events', 'core.events') 
create_module_alias('src.data', 'data')
create_module_alias('src.models', 'models')
create_module_alias('src.strategy', 'strategy')
create_module_alias('src.execution', 'execution')

# Create data namespace if it doesn't exist
if 'data' not in sys.modules:
    data_module = types.ModuleType('data')
    sys.modules['data'] = data_module

# Directly import and alias the problematic test modules
try:
    # First try to use a direct module import
    test_data_event_integration = importlib.import_module('tests.integration.data.test_data_event_integration')
    sys.modules['data.test_data_event_integration'] = test_data_event_integration
except ImportError:
    # If that fails, try to load from the file path
    test_path = os.path.join(project_root, 'tests/integration/data/test_data_event_integration.py')
    if os.path.exists(test_path):
        spec = importlib.util.spec_from_file_location('data.test_data_event_integration', test_path)
        test_data_event_integration = importlib.util.module_from_spec(spec)
        sys.modules['data.test_data_event_integration'] = test_data_event_integration
        spec.loader.exec_module(test_data_event_integration)
    else:
        # If all else fails, create an empty module
        sys.modules['data.test_data_event_integration'] = types.ModuleType('data.test_data_event_integration')

try:
    # First try to use a direct module import
    test_transformers = importlib.import_module('tests.integration.data.test_transformers')
    sys.modules['data.test_transformers'] = test_transformers
except ImportError:
    # If that fails, try to load from the file path
    test_path = os.path.join(project_root, 'tests/integration/data/test_transformers.py')
    if os.path.exists(test_path):
        spec = importlib.util.spec_from_file_location('data.test_transformers', test_path)
        test_transformers = importlib.util.module_from_spec(spec)
        sys.modules['data.test_transformers'] = test_transformers
        spec.loader.exec_module(test_transformers)
    else:
        # If all else fails, create an empty module
        sys.modules['data.test_transformers'] = types.ModuleType('data.test_transformers')

# Print paths for debugging (only when run directly)
if __name__ == "__main__":
    print(f"Project root: {project_root}")
    print(f"Source path: {src_path}")
    print(f"Python path: {sys.path}")
    print(f"Module aliases: {['core' in sys.modules, 'core.events' in sys.modules, 'data' in sys.modules]}")
    print(f"Test modules: {['data.test_data_event_integration' in sys.modules, 'data.test_transformers' in sys.modules]}")
