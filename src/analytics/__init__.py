"""
Analytics module for performance measurement, event tracking, and reporting.
"""
# Import existing modules
from . import metrics
from . import visualize
# Import performance module
from . import performance
# Export key classes from performance
from .performance import PerformanceCalculator, PerformanceCalculatorBase, PerformanceAnalytics

__all__ = [
    'metrics',
    'visualize',
    'performance',
    'PerformanceCalculator',
    'PerformanceAnalytics',
    'PerformanceCalculatorBase'
]
