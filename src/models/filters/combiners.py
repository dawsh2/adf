# src/models/filters/combiners.py
from .filter_base import FilterCombiner
from .result import FilterResult

class AndFilterCombiner(FilterCombiner):
    """Combines filters with AND logic (all must pass)."""
    
    def combine(self, results, names=None):
        """Combine multiple filter results."""
        if not results:
            return FilterResult(True, "No filters to evaluate")
        
        # Pass only if all pass
        passed = all(result.passed for result in results)
        
        # Get reason from first failing filter if any
        reason = None
        if not passed:
            for result in results:
                if not result.passed:
                    reason = result.reason
                    break
        else:
            reason = "All filters passed"
        
        # Combine metadata
        metadata = {}
        if names:
            for name, result in zip(names, results):
                metadata[name] = result.to_dict()
        else:
            for i, result in enumerate(results):
                metadata[f"filter_{i}"] = result.to_dict()
        
        return FilterResult(passed, reason, metadata)


class OrFilterCombiner(FilterCombiner):
    """Combines filters with OR logic (any can pass)."""
    
    def combine(self, results, names=None):
        """Combine multiple filter results."""
        if not results:
            return FilterResult(True, "No filters to evaluate")
        
        # Pass if any pass
        passed = any(result.passed for result in results)
        
        # Get reason
        if passed:
            # Find first passing filter
            for result in results:
                if result.passed:
                    reason = result.reason
                    break
        else:
            reason = "All filters failed"
        
        # Combine metadata
        metadata = {}
        if names:
            for name, result in zip(names, results):
                metadata[name] = result.to_dict()
        else:
            for i, result in enumerate(results):
                metadata[f"filter_{i}"] = result.to_dict()
        
        return FilterResult(passed, reason, metadata)
