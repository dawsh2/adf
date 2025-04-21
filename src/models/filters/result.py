# src/models/filters/filter_base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from .result import FilterResult


class FilterResult:
    """
    Result of a filter evaluation.
    Contains the result, reason, and additional metadata.
    """
    
    def __init__(self, passed: bool, reason: Optional[str] = None, 
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize filter result.
        
        Args:
            passed: Whether the filter passed
            reason: Optional explanation for result
            metadata: Optional additional data
        """
        self.passed = passed
        self.reason = reason or ("Passed" if passed else "Failed")
        self.metadata = metadata or {}
    
    def __bool__(self) -> bool:
        """
        Allow direct boolean evaluation.
        
        Returns:
            True if filter passed, False otherwise
        """
        return self.passed
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary.
        
        Returns:
            Dictionary representation of filter result
        """
        return {
            'passed': self.passed,
            'reason': self.reason,
            'metadata': self.metadata
        }    
