import datetime
from src.core.events.event_types import EventType, BarEvent, SignalEvent
from src.models.components.base import StrategyBase

class MovingAverageCrossoverStrategy(StrategyBase):
    """Simple moving average crossover strategy for testing."""
    
    def __init__(self, name, symbols, fast_window=10, slow_window=30, 
                 config=None, container=None, signal_emitter=None, order_emitter=None):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            symbols: List of symbols to trade
            fast_window: Fast moving average window
            slow_window: Slow moving average window
        """
        self.name = name
        self._config = config
        self._container = container
        self.signal_emitter = signal_emitter
        self.order_emitter = order_emitter
        
        # Default parameters that will be used if not loaded from config
        self._default_params = {
            'fast_window': fast_window,
            'slow_window': slow_window
        }
        
        # Component collections (initialized empty as required by base class)
        self.indicators = {}
        self.features = {}
        self.rules = {}
        self.state = {}
        
        # Load parameters - don't call parent constructor, handle it directly
        self.params = self._load_parameters()
        
        # Settings
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.fast_window = self.params.get('fast_window', fast_window)
        self.slow_window = self.params.get('slow_window', slow_window)
        
        # Store price history
        self.price_history = {symbol: [] for symbol in self.symbols}
        
        # State
        self.last_ma_values = {symbol: {'fast': None, 'slow': None} for symbol in self.symbols}
        self.signals = []
    
    def _load_parameters(self):
        """
        Load parameters from configuration or use defaults.
        
        Returns:
            dict: Strategy parameters
        """
        params = {}
        
        # Try to load from config if available
        if self._config is not None:
            # Get configuration section for strategies
            strategy_config = self._config.get_section('strategies')
            
            # Get configuration specific to this strategy
            if hasattr(strategy_config, 'get_section'):
                my_config = strategy_config.get_section(self.name)
                params = my_config.as_dict() if hasattr(my_config, 'as_dict') else {}
        
        # Use default parameters for any missing values
        for key, value in self._default_params.items():
            if key not in params:
                params[key] = value
                
        return params
    
    def on_bar(self, event):
        """Process a bar event."""
        if not isinstance(event, BarEvent):
            return None
            
        symbol = event.get_symbol()
        if symbol not in self.symbols:
            return None
            
        # Add price to history
        close_price = event.get_close()
        self.price_history[symbol].append(close_price)
        
        # Keep history manageable
        if len(self.price_history[symbol]) > self.slow_window + 10:
            self.price_history[symbol] = self.price_history[symbol][-(self.slow_window + 10):]
        
        # Check if we have enough data
        if len(self.price_history[symbol]) < self.slow_window:
            return None
            
        # Calculate MAs
        fast_ma = sum(self.price_history[symbol][-self.fast_window:]) / self.fast_window
        slow_ma = sum(self.price_history[symbol][-self.slow_window:]) / self.slow_window
        
        # Get previous MA values
        prev_fast = self.last_ma_values[symbol]['fast']
        prev_slow = self.last_ma_values[symbol]['slow']
        
        # Update MA values
        self.last_ma_values[symbol]['fast'] = fast_ma
        self.last_ma_values[symbol]['slow'] = slow_ma
        
        # Skip if no previous values
        if prev_fast is None or prev_slow is None:
            return None
            
        # Check for crossover
        signal = None
        
        # Buy signal: fast MA crosses above slow MA
        if fast_ma > slow_ma and prev_fast <= prev_slow:
            signal = SignalEvent(
                signal_value=SignalEvent.BUY,
                price=close_price,
                symbol=symbol,
                rule_id=self.name,
                confidence=1.0,
                metadata={
                    'fast_ma': fast_ma,
                    'slow_ma': slow_ma
                },
                timestamp=event.get_timestamp()
            )
            
        # Sell signal: fast MA crosses below slow MA
        elif fast_ma < slow_ma and prev_fast >= prev_slow:
            signal = SignalEvent(
                signal_value=SignalEvent.SELL,
                price=close_price,
                symbol=symbol,
                rule_id=self.name,
                confidence=1.0,
                metadata={
                    'fast_ma': fast_ma,
                    'slow_ma': slow_ma
                },
                timestamp=event.get_timestamp()
            )
        
        # Emit signal if generated
        if signal:
            self.signals.append(signal)
            
            if self.signal_emitter:
                self.signal_emitter.emit(signal)
            elif self.event_bus:
                self.event_bus.emit(signal)
        
        return signal
    
    def reset(self):
        """Reset the strategy state."""
        # Clear price history
        self.price_history = {symbol: [] for symbol in self.symbols}
        
        # Reset MA values
        self.last_ma_values = {symbol: {'fast': None, 'slow': None} for symbol in self.symbols}
        
        # Clear signals
        self.signals = []
