"""
Configuration file for pytest and unittest discovery.

This file helps with properly setting up the Python path for tests.
"""
import os
import sys
import importlib.util

# Get the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add src directory to Python path
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# Add project root to Python path
sys.path.insert(0, project_root)

# This is critical: modify sys.modules to make the 'core' module importable directly
# by creating a module alias from 'src.core' to 'core'
if importlib.util.find_spec('src.core') and 'core' not in sys.modules:
    core_spec = importlib.util.find_spec('src.core')
    core_module = importlib.util.module_from_spec(core_spec)
    sys.modules['core'] = core_module
    
    # Also set up data module alias
    if importlib.util.find_spec('src.data') and 'data' not in sys.modules:
        data_spec = importlib.util.find_spec('src.data')
        data_module = importlib.util.module_from_spec(data_spec)
        sys.modules['data'] = data_module

# Print paths for debugging (only when run directly)
if __name__ == "__main__":
    print(f"Project root: {project_root}")
    print(f"Source path: {src_path}")
    print(f"Python path: {sys.path}")
    print(f"Module aliases: {['core' in sys.modules, 'data' in sys.modules]}")
