"""
Regime Detector Factory Module

This module provides factory methods for creating different regime detector implementations.
"""
import logging
from typing import Dict, Any, Optional, Type

from .regime_detector import (
    RegimeDetectorBase,
    BasicRegimeDetector,
    EnhancedRegimeDetector
)

logger = logging.getLogger(__name__)

class RegimeDetectorFactory:
    """
    Factory for creating regime detector instances.
    
    This factory centralizes the creation of different regime detector implementations
    with appropriate configuration.
    """
    
    @staticmethod
    def create_detector(detector_type: str = 'basic', **kwargs) -> RegimeDetectorBase:
        """
        Create a regime detector instance.
        
        Args:
            detector_type: Type of detector to create ('basic', 'enhanced')
            **kwargs: Configuration parameters for the detector
            
        Returns:
            RegimeDetectorBase: Configured detector instance
            
        Raises:
            ValueError: If detector_type is not recognized
        """
        # Normalize detector type
        detector_type = detector_type.lower()
        
        if detector_type == 'basic':
            return RegimeDetectorFactory._create_basic_detector(**kwargs)
        elif detector_type == 'enhanced':
            return RegimeDetectorFactory._create_enhanced_detector(**kwargs)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
    
    @staticmethod
    def _create_basic_detector(**kwargs) -> BasicRegimeDetector:
        """
        Create a basic regime detector instance.
        
        Args:
            **kwargs: Configuration parameters
            
        Returns:
            BasicRegimeDetector: Configured detector instance
        """
        # Extract parameters with defaults
        lookback_window = kwargs.get('lookback_window', 20)
        trend_threshold = kwargs.get('trend_threshold', 0.05)
        volatility_threshold = kwargs.get('volatility_threshold', 0.015)
        sideways_threshold = kwargs.get('sideways_threshold', 0.02)
        
        return BasicRegimeDetector(
            lookback_window=lookback_window, 
            trend_threshold=trend_threshold,
            volatility_threshold=volatility_threshold, 
            sideways_threshold=sideways_threshold
        )
    
    @staticmethod
    def _create_enhanced_detector(**kwargs) -> EnhancedRegimeDetector:
        """
        Create an enhanced regime detector instance.
        
        Args:
            **kwargs: Configuration parameters
            
        Returns:
            EnhancedRegimeDetector: Configured detector instance
        """
        # Extract parameters with defaults
        lookback_window = kwargs.get('lookback_window', 30)
        trend_lookback = kwargs.get('trend_lookback', 50)
        volatility_lookback = kwargs.get('volatility_lookback', 20)
        trend_threshold = kwargs.get('trend_threshold', 0.03)
        volatility_threshold = kwargs.get('volatility_threshold', 0.012)
        sideways_threshold = kwargs.get('sideways_threshold', 0.015)
        debug = kwargs.get('debug', False)
        
        return EnhancedRegimeDetector(
            lookback_window=lookback_window,
            trend_lookback=trend_lookback,
            volatility_lookback=volatility_lookback,
            trend_threshold=trend_threshold,
            volatility_threshold=volatility_threshold,
            sideways_threshold=sideways_threshold,
            debug=debug
        )
    
    @staticmethod
    def get_detector_types() -> Dict[str, Type[RegimeDetectorBase]]:
        """
        Get dictionary of available detector types.
        
        Returns:
            dict: Mapping from detector type name to detector class
        """
        return {
            'basic': BasicRegimeDetector,
            'enhanced': EnhancedRegimeDetector
        }
    
    @staticmethod
    def create_preset_detector(preset: str, **kwargs) -> RegimeDetectorBase:
        """
        Create a detector with a specific preset configuration.
        
        Args:
            preset: Preset configuration name ('default', 'sensitive', 'stable')
            **kwargs: Additional parameters to override preset values
            
        Returns:
            RegimeDetectorBase: Configured detector instance
            
        Raises:
            ValueError: If preset is not recognized
        """
        preset = preset.lower()
        
        if preset == 'default':
            # Default configuration
            config = {}
            detector_type = 'basic'
        elif preset == 'sensitive':
            # More responsive to short-term changes
            config = {
                'lookback_window': 15,
                'trend_threshold': 0.03,
                'volatility_threshold': 0.01,
                'sideways_threshold': 0.015
            }
            detector_type = 'basic'
        elif preset == 'stable':
            # Less sensitive, more stable regimes
            config = {
                'lookback_window': 40,
                'trend_threshold': 0.08,
                'volatility_threshold': 0.02,
                'sideways_threshold': 0.03
            }
            detector_type = 'basic'
        elif preset == 'advanced':
            # Enhanced detector with default settings
            config = {}
            detector_type = 'enhanced'
        elif preset == 'advanced_sensitive':
            # Enhanced detector tuned for sensitivity
            config = {
                'lookback_window': 15,
                'trend_lookback': 30,
                'volatility_lookback': 10,
                'trend_threshold': 0.01,
                'volatility_threshold': 0.008,
                'sideways_threshold': 0.01
            }
            detector_type = 'enhanced'
        elif preset == 'advanced_stable':
            # Enhanced detector tuned for stability
            config = {
                'lookback_window': 40,
                'trend_lookback': 80,
                'volatility_lookback': 30,
                'trend_threshold': 0.05,
                'volatility_threshold': 0.02,
                'sideways_threshold': 0.03
            }
            detector_type = 'enhanced'
        else:
            raise ValueError(f"Unknown preset: {preset}")
        
        # Override preset config with kwargs
        config.update(kwargs)
        
        # Create detector
        return RegimeDetectorFactory.create_detector(detector_type, **config)
