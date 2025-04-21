# src/models/filters/manager.py
from typing import Dict, Any, Optional, List, Callable, Type
import logging

from .filter_base import FilterBase, FilterCombiner
from .interfaces import FilterTarget
from .result import FilterResult

logger = logging.getLogger(__name__)

class FilterManager:
    """
    Manager for coordinating filter evaluation.
    """
    
    def __init__(self, name: str = "filter_manager"):
        self.name = name
        self.filters = {}  # name -> FilterBase
        self.combiners = {}  # name -> FilterCombiner
        self.pipelines = {}  # name -> pipeline configuration
        self.results = {}  # symbol -> {pipeline -> result}
        self.event_bus = None
    
    def set_event_bus(self, event_bus):
        """Set the event bus."""
        self.event_bus = event_bus
        return self
    
    def register_filter(self, name: str, filter_obj: FilterBase) -> None:
        """
        Register a filter.
        
        Args:
            name: Filter name
            filter_obj: FilterBase instance
        """
        if not isinstance(filter_obj, FilterBase):
            raise TypeError("Filter must be a FilterBase instance")
        self.filters[name] = filter_obj
    
    def register_combiner(self, name: str, combiner: FilterCombiner) -> None:
        """
        Register a filter combiner.
        
        Args:
            name: Combiner name
            combiner: FilterCombiner instance
        """
        if not isinstance(combiner, FilterCombiner):
            raise TypeError("Combiner must be a FilterCombiner instance")
        self.combiners[name] = combiner
    
    def create_pipeline(self, name: str, config: Dict[str, Any]) -> None:
        """
        Create a filter pipeline.
        
        Args:
            name: Pipeline name
            config: Pipeline configuration
        """
        self.pipelines[name] = config
    
    def evaluate_pipeline(self, pipeline_name: str, context: Dict[str, Any], 
                        symbol: Optional[str] = None,
                        timestamp: Optional[Any] = None) -> FilterResult:
        """
        Evaluate a filter pipeline.
        
        Args:
            pipeline_name: Name of pipeline to evaluate
            context: Filter context
            symbol: Optional symbol
            timestamp: Optional timestamp
            
        Returns:
            FilterResult: Result of pipeline evaluation
        """
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")
            
        config = self.pipelines[pipeline_name]
        
        # Evaluate individual filters
        filter_results = []
        filter_names = []
        
        for filter_name in config.get('filters', []):
            if filter_name in self.filters:
                try:
                    result = self.filters[filter_name].evaluate(context)
                    filter_results.append(result)
                    filter_names.append(filter_name)
                except Exception as e:
                    logger.error(f"Error evaluating filter {filter_name}: {e}")
                    # Create a failed result
                    filter_results.append(FilterResult(
                        passed=False,
                        reason=f"Error: {e}"
                    ))
                    filter_names.append(filter_name)
        
        # Evaluate sub-pipelines
        for sub_config in config.get('sub_pipelines', []):
            sub_name = sub_config.get('name')
            if not sub_name:
                continue
                
            # Create temporary pipeline
            temp_pipeline_name = f"{pipeline_name}.{sub_name}"
            self.pipelines[temp_pipeline_name] = sub_config
            
            # Evaluate sub-pipeline
            sub_result = self.evaluate_pipeline(
                temp_pipeline_name, context, symbol, timestamp
            )
            
            # Add result
            filter_results.append(sub_result)
            filter_names.append(sub_name)
            
            # Clean up temporary pipeline
            del self.pipelines[temp_pipeline_name]
        
        # Combine results
        combiner_name = config.get('combiner', 'and')
        if combiner_name not in self.combiners:
            combiner_name = 'and'  # Default to AND
            
        combiner = self.combiners[combiner_name]
        combined_result = combiner.combine(filter_results, filter_names)
        
        # Store result
        if symbol:
            if symbol not in self.results:
                self.results[symbol] = {}
            self.results[symbol][pipeline_name] = combined_result
            
            # Emit filter event if event bus available
            if self.event_bus:
                self._emit_filter_event(pipeline_name, symbol, combined_result, timestamp)
        
        return combined_result
    
    def evaluate(self, pipeline_name: str, target: FilterTarget,
               symbol: Optional[str] = None,
               timestamp: Optional[Any] = None) -> FilterResult:
        """
        Evaluate target with filter pipeline.
        
        Args:
            pipeline_name: Name of pipeline to evaluate
            target: FilterTarget instance
            symbol: Optional symbol
            timestamp: Optional timestamp
            
        Returns:
            FilterResult: Result of evaluation
        """
        if not isinstance(target, FilterTarget):
            raise TypeError("Target must be a FilterTarget instance")
            
        # Get context from target
        context = target.get_filter_context(symbol, timestamp)
        
        # Evaluate pipeline
        result = self.evaluate_pipeline(pipeline_name, context, symbol, timestamp)
        
        # Apply result to target
        target.apply_filter_result(result, symbol, timestamp)
        
        return result
    
    def on_bar(self, event) -> Dict[str, Dict[str, FilterResult]]:
        """
        Process a bar event against all pipelines.
        
        Args:
            event: BarEvent to process
            
        Returns:
            dict: Results for each pipeline
        """
        # Extract event details
        symbol = event.get_symbol()
        timestamp = event.get_timestamp()
        
        # Create context from bar event
        context = {
            'event': event,
            'symbol': symbol,
            'timestamp': timestamp,
            'open': event.get_open(),
            'high': event.get_high(),
            'low': event.get_low(),
            'close': event.get_close(),
            'volume': event.get_volume()
        }
        
        # Evaluate all pipelines
        results = {}
        
        for pipeline_name in self.pipelines:
            results[pipeline_name] = self.evaluate_pipeline(
                pipeline_name, context, symbol, timestamp
            )
            
        return results
    
    def reset(self) -> None:
        """Reset all filter states."""
        self.results = {}
        
        # Reset filters if they support it
        for filter_obj in self.filters.values():
            if hasattr(filter_obj, 'reset'):
                filter_obj.reset()
                
    def _emit_filter_event(self, pipeline: str, symbol: str, 
                         result: FilterResult, timestamp: Any) -> None:
        """
        Emit a filter event.
        
        Args:
            pipeline: Pipeline name
            symbol: Symbol
            result: Filter result
            timestamp: Timestamp
        """
        # Implementation depends on event system
        pass
