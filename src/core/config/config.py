import os
import yaml
import json
from typing import Dict, Any, Optional, Union, List, Type, TypeVar, cast

T = TypeVar('T')

class ConfigSection:
    """A section of configuration values."""
    
    def __init__(self, name: str, values: Dict[str, Any] = None):
        self.name = name
        self._values = values or {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._values.get(key, default)
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get a configuration value as an integer."""
        value = self.get(key, default)
        return int(value)
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get a configuration value as a float."""
        value = self.get(key, default)
        return float(value)
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get a configuration value as a boolean."""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', 'yes', 'y', '1')
        return bool(value)
    
    def get_list(self, key: str, default: List = None) -> List:
        """Get a configuration value as a list."""
        value = self.get(key, default or [])
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return value.split(',')
        return [value]
    
    def get_section(self, name: str) -> 'ConfigSection':
        """Get a nested configuration section."""
        value = self.get(name, {})
        if not isinstance(value, dict):
            value = {}
        return ConfigSection(f"{self.name}.{name}", value)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self._values[key] = value
    
    def update(self, values: Dict[str, Any]) -> None:
        """Update multiple configuration values."""
        self._values.update(values)
    
    def as_dict(self) -> Dict[str, Any]:
        """Get all values as a dictionary."""
        return dict(self._values)


class Config:
    """Hierarchical configuration system."""
    
    def __init__(self):
        self._sections = {}  # name -> ConfigSection
        self._defaults = {}  # name -> defaults dict
    
    def register_defaults(self, section_name: str, defaults: Dict[str, Any]) -> None:
        """Register default values for a section."""
        self._defaults[section_name] = defaults
        # Ensure section exists
        if section_name not in self._sections:
            self._sections[section_name] = ConfigSection(section_name, dict(defaults))
        else:
            # Update existing section with defaults for missing keys
            section = self._sections[section_name]
            for key, value in defaults.items():
                if key not in section.as_dict():
                    section.set(key, value)
    
    def load_file(self, filepath: str) -> None:
        """Load configuration from a file (YAML or JSON)."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        # Determine file type from extension
        _, ext = os.path.splitext(filepath)
        
        try:
            with open(filepath, 'r') as f:
                if ext.lower() in ('.yaml', '.yml'):
                    config_data = yaml.safe_load(f)
                elif ext.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {ext}")
            
            # Update sections with loaded data
            self._update_from_dict(config_data)
        except Exception as e:
            raise ValueError(f"Error loading configuration file: {e}")
    
    def load_env(self, prefix: str = 'APP_') -> None:
        """Load configuration from environment variables."""
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert APP_SECTION_KEY to section.key
                parts = key[len(prefix):].lower().split('_')
                if len(parts) > 1:
                    section_name = parts[0]
                    setting_key = '_'.join(parts[1:])
                    
                    # Ensure section exists
                    if section_name not in self._sections:
                        self._sections[section_name] = ConfigSection(section_name)
                    
                    # Update setting
                    self._sections[section_name].set(setting_key, value)
    
    def get_section(self, name: str) -> ConfigSection:
        """Get a configuration section."""
        if name not in self._sections:
            # Create section with defaults if available
            defaults = self._defaults.get(name, {})
            self._sections[name] = ConfigSection(name, dict(defaults))
        
        return self._sections[name]
    
    def _update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update configuration from a dictionary."""
        for section_name, section_data in data.items():
            if isinstance(section_data, dict):
                section = self.get_section(section_name)
                section.update(section_data)
