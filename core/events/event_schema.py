import json
from typing import Dict, Any, Optional, List, Callable
from .event_types import EventType

class SchemaValidator:
    """Validator for event data schemas."""
    
    def __init__(self):
        self.schemas = {}
        self._initialize_default_schemas()
    
    def _initialize_default_schemas(self):
        """Initialize default schemas for system events."""
        # Bar event schema
        self.register_schema(EventType.BAR, {
            'required_fields': [
                'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ],
            'field_types': {
                'symbol': str,
                'open': (int, float),
                'high': (int, float),
                'low': (int, float),
                'close': (int, float),
                'volume': (int, float)
            }
        })
        
        # Signal event schema
        self.register_schema(EventType.SIGNAL, {
            'required_fields': [
                'signal_value', 'price', 'symbol'
            ],
            'field_types': {
                'signal_value': int,
                'price': (int, float),
                'symbol': str,
                'confidence': (int, float),
                'rule_id': (str, type(None)),
                'metadata': dict
            },
            'custom_validations': [
                lambda data: data.get('signal_value') in (1, -1, 0),
                lambda data: 0.0 <= data.get('confidence', 1.0) <= 1.0
            ]
        })
        
        # Order event schema
        self.register_schema(EventType.ORDER, {
            'required_fields': [
                'symbol', 'order_type', 'direction', 'quantity'
            ],
            'field_types': {
                'symbol': str,
                'order_type': str,
                'direction': str,
                'quantity': (int, float),
                'price': (int, float, type(None))
            },
            'custom_validations': [
                lambda data: data.get('order_type') in ('MARKET', 'LIMIT', 'STOP'),
                lambda data: data.get('direction') in ('BUY', 'SELL'),
                lambda data: data.get('quantity', 0) > 0
            ]
        })
        
        # Fill event schema
        self.register_schema(EventType.FILL, {
            'required_fields': [
                'symbol', 'direction', 'quantity', 'price'
            ],
            'field_types': {
                'symbol': str,
                'direction': str,
                'quantity': (int, float),
                'price': (int, float),
                'commission': (int, float)
            },
            'custom_validations': [
                lambda data: data.get('direction') in ('BUY', 'SELL'),
                lambda data: data.get('quantity', 0) > 0,
                lambda data: data.get('price', 0) > 0
            ]
        })
    
    def register_schema(self, event_type, schema):
        """Register a schema for an event type."""
        self.schemas[event_type] = schema
    
    def validate(self, event_type, data):
        """
        Validate event data against schema.
        
        Args:
            event_type: EventType to validate against
            data: Dictionary of event data
            
        Returns:
            (bool, str): (is_valid, error_message)
        """
        if event_type not in self.schemas:
            return True, ""  # No schema defined for this event type
            
        schema = self.schemas[event_type]
        
        # Check required fields
        required_fields = schema.get('required_fields', [])
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Check field types
        field_types = schema.get('field_types', {})
        for field, expected_type in field_types.items():
            if field in data and data[field] is not None:
                if isinstance(expected_type, tuple):
                    if not isinstance(data[field], expected_type):
                        return False, f"Field {field} has wrong type: expected one of {expected_type}, got {type(data[field])}"
                elif not isinstance(data[field], expected_type):
                    return False, f"Field {field} has wrong type: expected {expected_type}, got {type(data[field])}"
        
        # Run custom validations
        custom_validations = schema.get('custom_validations', [])
        for validation in custom_validations:
            if not validation(data):
                return False, f"Custom validation failed"
        
        return True, ""
