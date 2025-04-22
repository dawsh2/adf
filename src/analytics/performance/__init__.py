# src/analytics/performance/__init__.py
"""
Performance calculation module for analyzing trading results.
"""

from .base import PerformanceCalculatorBase
from .calculator import PerformanceCalculator

__all__ = [
    'PerformanceCalculatorBase',
    'PerformanceCalculator'
]
