"""
Component for normalizing data.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class Normalizer:
    """Component for normalizing data."""
    
    def __init__(self, method: str = 'zscore', columns: Optional[List[str]] = None):
        """
        Initialize the normalizer.
        
        Args:
            method: Normalization method ('zscore', 'minmax', 'robust')
            columns: Columns to normalize (None for all)
        """
        self.method = method
        self.columns = columns
        self.stats = {}  # For storing normalization statistics
        
        # Validate method
        valid_methods = ['zscore', 'minmax', 'robust']
        if method not in valid_methods:
            logger.warning(f"Invalid normalization method: {method}. Using 'zscore' instead.")
            self.method = 'zscore'
    
    def fit(self, df: pd.DataFrame) -> 'Normalizer':
        """
        Fit the normalizer to data.
        
        Args:
            df: DataFrame to fit to
            
        Returns:
            Self for method chaining
        """
        columns = self.columns or list(df.columns)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            try:
                if self.method == 'zscore':
                    mean = df[col].mean()
                    std = df[col].std()
                    if std == 0:  # Avoid division by zero
                        std = 1.0
                    self.stats[col] = {'mean': mean, 'std': std}
                    
                elif self.method == 'minmax':
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if min_val == max_val:  # Avoid division by zero
                        max_val = min_val + 1.0
                    self.stats[col] = {'min': min_val, 'max': max_val}
                    
                elif self.method == 'robust':
                    median = df[col].median()
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    if iqr == 0:  # Avoid division by zero
                        iqr = 1.0
                    self.stats[col] = {'median': median, 'iqr': iqr}
            except Exception as e:
                logger.error(f"Error fitting column {col}: {e}")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted parameters.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Normalized DataFrame
        """
        result = df.copy()
        columns = self.columns or list(df.columns)
        
        for col in columns:
            if col not in df.columns or col not in self.stats:
                continue
                
            try:
                if self.method == 'zscore':
                    mean = self.stats[col]['mean']
                    std = self.stats[col]['std']
                    result[col] = (df[col] - mean) / std
                    
                elif self.method == 'minmax':
                    min_val = self.stats[col]['min']
                    max_val = self.stats[col]['max']
                    result[col] = (df[col] - min_val) / (max_val - min_val)
                    
                elif self.method == 'robust':
                    median = self.stats[col]['median']
                    iqr = self.stats[col]['iqr']
                    result[col] = (df[col] - median) / iqr
            except Exception as e:
                logger.error(f"Error transforming column {col}: {e}")
        
        return result
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame to fit and transform
            
        Returns:
            Normalized DataFrame
        """
        return self.fit(df).transform(df)
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform normalized data.
        
        Args:
            df: Normalized DataFrame
            
        Returns:
            Original scale DataFrame
        """
        result = df.copy()
        columns = self.columns or list(df.columns)
        
        for col in columns:
            if col not in df.columns or col not in self.stats:
                continue
                
            try:
                if self.method == 'zscore':
                    mean = self.stats[col]['mean']
                    std = self.stats[col]['std']
                    result[col] = df[col] * std + mean
                    
                elif self.method == 'minmax':
                    min_val = self.stats[col]['min']
                    max_val = self.stats[col]['max']
                    result[col] = df[col] * (max_val - min_val) + min_val
                    
                elif self.method == 'robust':
                    median = self.stats[col]['median']
                    iqr = self.stats[col]['iqr']
                    result[col] = df[col] * iqr + median
            except Exception as e:
                logger.error(f"Error inverse transforming column {col}: {e}")
        
        return result
    
    def save_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get normalization statistics.
        
        Returns:
            Dictionary of statistics by column
        """
        return self.stats
    
    def load_stats(self, stats: Dict[str, Dict[str, float]]) -> 'Normalizer':
        """
        Load normalization statistics.
        
        Args:
            stats: Dictionary of statistics by column
            
        Returns:
            Self for method chaining
        """
        self.stats = stats
        return self
