import uuid
import datetime
from enum import Enum, auto
from typing import Dict, Any, Optional

class EventType(Enum):
    """Enum defining all event types in the system."""
    BAR = auto()            # Market data
    SIGNAL = auto()         # Trading signal
    ORDER = auto()          # Order request
    FILL = auto()           # Order fill
    POSITION = auto()       # Position update
    PORTFOLIO = auto()      # Portfolio state update
    STRATEGY = auto()       # Strategy state
    METRIC = auto()         # Performance metrics
    WEBSOCKET = auto()      # WebSocket events
    LIFECYCLE = auto()      # System lifecycle events
    ERROR = auto()          # Error events


    OPTIMIZATION = auto()   # Optimization events
    FILTER = auto()         # Filter state events
    REGIME = auto()         # Market regime events    

class Event:
    """Base class for all events in the system."""
    
    def __init__(self, event_type, data=None, timestamp=None):
        self.event_type = event_type
        self.data = data or {}
        self.timestamp = timestamp or datetime.datetime.now()
        self.id = str(uuid.uuid4())
    
    def get_type(self):
        """Get the event type."""
        return self.event_type
    
    def get_timestamp(self):
        """Get the event timestamp."""
        return self.timestamp
        
    def get_id(self):
        """Get the unique event ID."""
        return self.id


class BarEvent(Event):
    """Market data bar event."""
    
    def __init__(self, symbol, timestamp, open_price, high_price, 
                 low_price, close_price, volume):
        data = {
            'symbol': symbol,
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        }
        super().__init__(EventType.BAR, data, timestamp)
    
    # Accessor methods
    def get_symbol(self):
        return self.data['symbol']
        
    def get_open(self):
        return self.data['open']
        
    def get_high(self):
        return self.data['high']
        
    def get_low(self):
        return self.data['low']
        
    def get_close(self):
        return self.data['close']
        
    def get_volume(self):
        return self.data['volume']



class SignalEvent(Event):
    """Trading signal event."""
    
    # Signal constants
    BUY = 1
    SELL = -1
    NEUTRAL = 0
    
    def __init__(self, signal_value, price, symbol, rule_id=None, 
                 confidence=1.0, metadata=None, timestamp=None):
        # Validate signal value
        if signal_value not in (self.BUY, self.SELL, self.NEUTRAL):
            raise ValueError(f"Invalid signal value: {signal_value}")
            
        data = {
            'signal_value': signal_value,
            'price': price,
            'symbol': symbol,
            'rule_id': rule_id,
            'confidence': confidence,
            'metadata': metadata or {},
        }
        super().__init__(EventType.SIGNAL, data, timestamp)
    
    # Accessor methods
    def get_signal_value(self):
        return self.data['signal_value']
        
    def get_symbol(self):
        return self.data['symbol']
        
    def get_price(self):
        return self.data['price']
        
    def is_buy(self):
        return self.data['signal_value'] == self.BUY
        
    def is_sell(self):
        return self.data['signal_value'] == self.SELL


class OrderEvent(Event):
    """Order request event."""
    
    # Order types
    MARKET = 'MARKET'
    LIMIT = 'LIMIT'
    STOP = 'STOP'
    
    # Order directions
    BUY = 'BUY'
    SELL = 'SELL'
    
    def __init__(self, symbol, order_type, direction, quantity, 
                 price=None, timestamp=None):
        data = {
            'symbol': symbol,
            'order_type': order_type,
            'direction': direction,
            'quantity': quantity,
            'price': price,
        }
        super().__init__(EventType.ORDER, data, timestamp)
    
    # Accessor methods
    def get_symbol(self):
        return self.data['symbol']
        
    def get_order_type(self):
        return self.data['order_type']
        
    def get_direction(self):
        return self.data['direction']
        
    def get_quantity(self):
        return self.data['quantity']
        
    def get_price(self):
        return self.data['price']


class FillEvent(Event):
    """Order fill event."""
    
    def __init__(self, symbol, direction, quantity, price, 
                 commission=0.0, timestamp=None):
        data = {
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'price': price,
            'commission': commission,
        }
        super().__init__(EventType.FILL, data, timestamp)
    
    # Accessor methods
    def get_symbol(self):
        return self.data['symbol']
        
    def get_direction(self):
        return self.data['direction']
        
    def get_quantity(self):
        return self.data['quantity']
        
    def get_price(self):
        return self.data['price']
        
    def get_commission(self):
        return self.data['commission']


class WebSocketEvent(Event):
    """WebSocket connection event."""
    
    # Connection states
    CONNECTING = 'CONNECTING'
    CONNECTED = 'CONNECTED'
    DISCONNECTED = 'DISCONNECTED'
    MESSAGE = 'MESSAGE'
    ERROR = 'ERROR'
    
    def __init__(self, connection_id, state, data=None, timestamp=None):
        event_data = {
            'connection_id': connection_id,
            'state': state,
            'data': data or {},
        }
        super().__init__(EventType.WEBSOCKET, event_data, timestamp)
    
    # Accessor methods
    def get_connection_id(self):
        return self.data['connection_id']
        
    def get_state(self):
        return self.data['state']
        
    def get_data(self):
        return self.data['data']


class LifecycleEvent(Event):
    """System lifecycle event."""
    
    # Lifecycle states
    INITIALIZING = 'INITIALIZING'
    INITIALIZED = 'INITIALIZED'
    STARTING = 'STARTING'
    RUNNING = 'RUNNING'
    STOPPING = 'STOPPING'
    STOPPED = 'STOPPED'
    ERROR = 'ERROR'
    
    def __init__(self, state, component=None, data=None, timestamp=None):
        event_data = {
            'state': state,
            'component': component,
            'data': data or {},
        }
        super().__init__(EventType.LIFECYCLE, event_data, timestamp)
    
    # Accessor methods
    def get_state(self):
        return self.data['state']
        
    def get_component(self):
        return self.data['component']
        
    def get_data(self):
        return self.data['data']


class ErrorEvent(Event):
    """Error event."""
    
    def __init__(self, error_type, message, source=None, exception=None, timestamp=None):
        data = {
            'error_type': error_type,
            'message': message,
            'source': source,
            'exception': str(exception) if exception else None,
        }
        super().__init__(EventType.ERROR, data, timestamp)
    
    # Accessor methods
    def get_error_type(self):
        return self.data['error_type']
        
    def get_message(self):
        return self.data['message']
        
    def get_source(self):
        return self.data['source']
        
    def get_exception(self):
        return self.data['exception']


# Updates to src/core/events/event_types.py
class OptimizationEvent(Event):
    """Optimization result event."""
    
    def __init__(self, strategy, parameters, metrics, timestamp=None):
        data = {
            'strategy': strategy,
            'parameters': parameters,
            'metrics': metrics
        }
        super().__init__(EventType.OPTIMIZATION, data, timestamp)
    
    def get_strategy(self):
        return self.data['strategy']
        
    def get_parameters(self):
        return self.data['parameters']
        
    def get_metrics(self):
        return self.data['metrics']


class FilterEvent(Event):
    """Filter state change event."""
    
    def __init__(self, filter_name, symbol, state, reason=None, timestamp=None):
        data = {
            'filter_name': filter_name,
            'symbol': symbol,
            'state': state,  # True/False for active/inactive
            'reason': reason
        }
        super().__init__(EventType.FILTER, data, timestamp)
    
    def get_filter_name(self):
        return self.data['filter_name']
        
    def get_symbol(self):
        return self.data['symbol']
        
    def get_state(self):
        return self.data['state']
        
    def get_reason(self):
        return self.data['reason']


class RegimeEvent(Event):
    """Market regime change event."""
    
    def __init__(self, symbol, regime, confidence=1.0, timestamp=None):
        data = {
            'symbol': symbol,
            'regime': regime,
            'confidence': confidence
        }
        super().__init__(EventType.REGIME, data, timestamp)
    
    def get_symbol(self):
        return self.data['symbol']
        
    def get_regime(self):
        return self.data['regime']
        
    def get_confidence(self):
        return self.data['confidence']    
