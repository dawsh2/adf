# src/analytics/performance/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

class PerformanceCalculatorBase(ABC):
    """Base class for performance calculation."""

    @abstractmethod
    def calculate(self, equity_curve: List[Dict[str, Any]], trades: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Calculate performance metrics from equity curve and trade data.
        
        Args:
            equity_curve: List of equity points with timestamp and value
            trades: Optional list of executed trades
            
        Returns:
            Dictionary of performance metrics
        """
        pass
    
    @abstractmethod
    def get_metric(self, name: str) -> Any:
        """
        Get a specific performance metric by name.
        
        Args:
            name: Name of the metric
            
        Returns:
            Value of the metric
        """
        pass


# src/analytics/performance/calculator.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import timedelta
from .base import PerformanceCalculatorBase

logger = logging.getLogger(__name__)

class PerformanceCalculator(PerformanceCalculatorBase):
    """Standard performance calculator with common metrics."""
    
    def __init__(self, risk_free_rate: float = 0.01, annualization_factor: int = 252):
        """
        Initialize the performance calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 1%)
            annualization_factor: Number of trading periods in a year
                (252 for daily, 52 for weekly, 12 for monthly)
        """
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor
        self.metrics = {}  # Calculated metrics
    
    def calculate(self, equity_curve: List[Dict[str, Any]], 
                trades: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Calculate performance metrics from equity curve and trade data.
        
        Args:
            equity_curve: List of equity points with timestamp and value
            trades: Optional list of executed trades
            
        Returns:
            Dictionary of performance metrics
        """
        if not equity_curve:
            logger.warning("Empty equity curve provided")
            return self._empty_metrics()
        
        # Create DataFrame from equity curve
        df = pd.DataFrame(equity_curve)
        
        # Ensure we have timestamp and equity columns
        if 'timestamp' not in df.columns or 'equity' not in df.columns:
            if 'timestamp' not in df.columns and 'date' in df.columns:
                df['timestamp'] = df['date']
            if 'equity' not in df.columns and 'value' in df.columns:
                df['equity'] = df['value']
            
            if 'timestamp' not in df.columns or 'equity' not in df.columns:
                logger.error("Equity curve must contain 'timestamp' and 'equity' columns")
                return self._empty_metrics()
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        df['returns'] = df['equity'].pct_change()
        
        # Get initial and final values
        initial_equity = df['equity'].iloc[0]
        final_equity = df['equity'].iloc[-1]
        
        # Calculate trading period
        start_date = df.index[0]
        end_date = df.index[-1]
        
        try:
            days = (end_date - start_date).days
            years = max(days / 365, 0.01)  # Avoid division by zero
        except Exception as e:
            logger.warning(f"Error calculating trading period: {e}")
            days = len(df)
            years = days / self.annualization_factor
        
        # Calculate total return
        total_return = (final_equity / initial_equity) - 1
        
        # Calculate annualized return
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        # Calculate drawdown
        df['cummax'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] / df['cummax']) - 1
        max_drawdown = df['drawdown'].min()
        
        # Calculate Sharpe ratio
        daily_risk_free = (1 + self.risk_free_rate) ** (1 / self.annualization_factor) - 1
        excess_returns = df['returns'] - daily_risk_free
        sharpe_ratio = 0
        
        if df['returns'].std() > 0:
            sharpe_ratio = excess_returns.mean() / df['returns'].std() * np.sqrt(self.annualization_factor)
        
        # Calculate Sortino ratio (downside risk only)
        downside_returns = df.loc[df['returns'] < 0, 'returns']
        sortino_ratio = 0
        
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = df['returns'].mean() / downside_returns.std() * np.sqrt(self.annualization_factor)
        
        # Calculate Calmar ratio (return / max drawdown)
        calmar_ratio = 0
        if max_drawdown != 0:
            calmar_ratio = annual_return / abs(max_drawdown)
        
        # Store and return metrics
        self.metrics = {
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annual_return': annual_return,
            'annual_return_pct': annual_return * 100,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'trading_days': days,
            'trading_years': years
        }
        
        # Add trade-based metrics if trades provided
        if trades:
            self._calculate_trade_metrics(trades)
        
        return self.metrics
    
    def _calculate_trade_metrics(self, trades: List[Dict[str, Any]]) -> None:
        """
        Calculate trade-based performance metrics.
        
        Args:
            trades: List of executed trades
        """
        if not trades:
            return
        
        # Count trades
        num_trades = len(trades)
        
        # Calculate win/loss
        profitable_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        num_winning = len(profitable_trades)
        num_losing = len(losing_trades)
        
        win_rate = num_winning / num_trades if num_trades > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(t.get('pnl', 0) for t in profitable_trades)
        gross_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate average trade metrics
        avg_profit = gross_profit / num_winning if num_winning > 0 else 0
        avg_loss = gross_loss / num_losing if num_losing > 0 else 0
        
        # Calculate expectancy
        expectancy = (win_rate * avg_profit) - ((1 - win_rate) * avg_loss)
        
        # Add to metrics
        self.metrics.update({
            'num_trades': num_trades,
            'num_winning': num_winning,
            'num_losing': num_losing,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'expectancy': expectancy
        })
    
    def get_metric(self, name: str) -> Any:
        """
        Get a specific performance metric by name.
        
        Args:
            name: Name of the metric
            
        Returns:
            Value of the metric or None if not found
        """
        return self.metrics.get(name)
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics dictionary with zero values."""
        return {
            'initial_equity': 0,
            'final_equity': 0,
            'total_return': 0,
            'total_return_pct': 0,
            'annual_return': 0,
            'annual_return_pct': 0,
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'trading_days': 0,
            'trading_years': 0,
            'num_trades': 0,
            'num_winning': 0,
            'num_losing': 0,
            'win_rate': 0,
            'win_rate_pct': 0,
            'profit_factor': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'expectancy': 0
        }


# src/analytics/performance/__init__.py
from .base import PerformanceCalculatorBase
# from .calculator import PerformanceCalculator

__all__ = [
    'PerformanceCalculatorBase',
    'PerformanceCalculator'
]
