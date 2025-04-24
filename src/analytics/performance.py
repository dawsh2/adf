"""
Enhanced performance analytics for trading strategies with improved metrics calculation.
"""
import pandas as pd
import numpy as np
import datetime
import logging
from typing import Dict, List, Any, Optional, Tuple
from tabulate import tabulate

class PerformanceAnalytics:
    """
    Calculate and display performance metrics for trading strategies.
    """
    
    @staticmethod
    def calculate_metrics(equity_curve: pd.DataFrame, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate performance metrics from equity curve and trade list.
        
        Args:
            equity_curve: DataFrame with timestamp and equity columns
            trades: List of trade dictionaries
            
        Returns:
            dict: Performance metrics
        """
        metrics = {}
        
        # Validate input data
        if equity_curve is None or len(equity_curve) < 2:
            return {'error': 'Insufficient equity curve data'}
            
        # Make sure DataFrame has required columns
        required_cols = ['timestamp', 'equity']
        if not all(col in equity_curve.columns for col in required_cols):
            # Try to adjust column names (sometimes 'timestamp' might be the index)
            if equity_curve.index.name == 'timestamp' or isinstance(equity_curve.index, pd.DatetimeIndex):
                equity_curve = equity_curve.reset_index()
            else:
                return {'error': 'Equity curve missing required columns'}
        
        # Sort by timestamp
        equity_curve = equity_curve.sort_values('timestamp')
        
        # Basic equity metrics
        metrics['initial_equity'] = float(equity_curve['equity'].iloc[0])
        metrics['final_equity'] = float(equity_curve['equity'].iloc[-1])
        metrics['peak_equity'] = float(equity_curve['equity'].max())
        metrics['min_equity'] = float(equity_curve['equity'].min())
        
        # Ensure final equity is correct after liquidation
        if metrics['final_equity'] <= 0:
            print("WARNING: Final equity is negative or zero, this indicates a calculation error")
            metrics['final_equity'] = max(1.0, metrics['final_equity'])  # Avoid division by zero
        
        # Return metrics
        metrics['absolute_return'] = metrics['final_equity'] - metrics['initial_equity']
        metrics['total_return'] = (metrics['final_equity'] / metrics['initial_equity'] - 1) * 100
        
        # Time-based metrics
        try:
            start_date = pd.to_datetime(equity_curve['timestamp'].iloc[0])
            end_date = pd.to_datetime(equity_curve['timestamp'].iloc[-1])
            days = (end_date - start_date).total_seconds() / (24 * 60 * 60)
            
            # Avoid division by zero
            if days > 0:
                metrics['annualized_return'] = (((1 + metrics['total_return']/100) ** (365/days)) - 1) * 100
            else:
                metrics['annualized_return'] = 0
        except:
            metrics['annualized_return'] = 0
        
        # Drawdown analysis
        rolling_max = equity_curve['equity'].cummax()
        drawdown = (equity_curve['equity'] / rolling_max - 1) * 100
        metrics['max_drawdown'] = float(drawdown.min())
        
        # Calculate drawdown periods
        drawdown_start = None
        drawdown_periods = []
        
        for i, row in enumerate(equity_curve.itertuples()):
            equity = row.equity
            max_equity = rolling_max[i]
            
            # If in drawdown and not already tracking
            if equity < max_equity and drawdown_start is None:
                drawdown_start = row.timestamp
            
            # If not in drawdown and was tracking
            elif equity == max_equity and drawdown_start is not None:
                drawdown_end = row.timestamp
                drawdown_periods.append((drawdown_start, drawdown_end))
                drawdown_start = None
        
        # Add any ongoing drawdown
        if drawdown_start is not None:
            drawdown_periods.append((drawdown_start, equity_curve['timestamp'].iloc[-1]))
            
        metrics['drawdown_periods'] = drawdown_periods
        metrics['max_drawdown_length'] = 0
        
        if drawdown_periods:
            # Calculate longest drawdown in days
            try:
                drawdown_lengths = [(pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / (24 * 60 * 60) 
                                  for start, end in drawdown_periods]
                metrics['max_drawdown_length'] = max(drawdown_lengths)
            except:
                pass
        
        # Trade metrics
        if trades:
            metrics['trade_count'] = len(trades)
            
            # Filter trades with PnL data
            pnl_trades = [t for t in trades if 'pnl' in t and t['pnl'] != 0]
            
            if pnl_trades:
                pnl_values = [t['pnl'] for t in pnl_trades]
                metrics['total_pnl'] = sum(pnl_values)
                metrics['avg_trade_pnl'] = sum(pnl_values) / len(pnl_values)
                
                winning_trades = [t for t in pnl_trades if t['pnl'] > 0]
                losing_trades = [t for t in pnl_trades if t['pnl'] < 0]
                
                metrics['win_count'] = len(winning_trades)
                metrics['loss_count'] = len(losing_trades)
                metrics['win_rate'] = len(winning_trades) / len(pnl_trades) * 100 if pnl_trades else 0
                
                if winning_trades:
                    metrics['avg_win'] = sum(t['pnl'] for t in winning_trades) / len(winning_trades)
                    metrics['max_win'] = max(t['pnl'] for t in winning_trades)
                else:
                    metrics['avg_win'] = 0
                    metrics['max_win'] = 0
                    
                if losing_trades:
                    metrics['avg_loss'] = sum(t['pnl'] for t in losing_trades) / len(losing_trades)
                    metrics['max_loss'] = min(t['pnl'] for t in losing_trades)
                else:
                    metrics['avg_loss'] = 0
                    metrics['max_loss'] = 0
                
                # Calculate profit factor
                gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
                gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
                metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0
                
                # Calculate expectancy
                if metrics['win_rate'] > 0 and metrics['loss_count'] > 0:
                    win_rate = metrics['win_rate'] / 100
                    loss_rate = 1 - win_rate
                    metrics['expectancy'] = (metrics['avg_win'] * win_rate) + (metrics['avg_loss'] * loss_rate)
                else:
                    metrics['expectancy'] = 0
            else:
                # No PnL data available
                metrics['total_pnl'] = 0
                metrics['avg_trade_pnl'] = 0
                metrics['win_count'] = 0
                metrics['loss_count'] = 0
                metrics['win_rate'] = 0
                metrics['avg_win'] = 0
                metrics['max_win'] = 0
                metrics['avg_loss'] = 0
                metrics['max_loss'] = 0
                metrics['profit_factor'] = 0
                metrics['expectancy'] = 0
        else:
            # No trades
            metrics['trade_count'] = 0
            metrics['total_pnl'] = 0
            metrics['avg_trade_pnl'] = 0
            metrics['win_count'] = 0
            metrics['loss_count'] = 0
            metrics['win_rate'] = 0
            metrics['avg_win'] = 0
            metrics['max_win'] = 0
            metrics['avg_loss'] = 0
            metrics['max_loss'] = 0
            metrics['profit_factor'] = 0
            metrics['expectancy'] = 0
        
        # Risk-adjusted metrics
        if len(equity_curve) > 1:
            # Calculate returns
            equity_curve['return'] = equity_curve['equity'].pct_change().fillna(0)
            
            # Calculate Sharpe ratio (assuming daily data)
            risk_free_rate = 0  # Simplification
            returns_mean = equity_curve['return'].mean()
            returns_std = equity_curve['return'].std()
            
            # Sharpe ratio
            if returns_std > 0:
                metrics['sharpe_ratio'] = (returns_mean - risk_free_rate) / returns_std
                # Annualized Sharpe (assuming daily data)
                metrics['sharpe_ratio'] = metrics['sharpe_ratio'] * np.sqrt(252)
            else:
                metrics['sharpe_ratio'] = 0
            
            # Sortino ratio (downside deviation only)
            downside_returns = equity_curve.loc[equity_curve['return'] < 0, 'return']
            downside_deviation = downside_returns.std()
            
            if downside_deviation > 0:
                metrics['sortino_ratio'] = (returns_mean - risk_free_rate) / downside_deviation
                # Annualized Sortino (assuming daily data)
                metrics['sortino_ratio'] = metrics['sortino_ratio'] * np.sqrt(252)
            else:
                metrics['sortino_ratio'] = 0
                
            # Calmar ratio (return / max drawdown)
            if metrics['max_drawdown'] != 0:
                metrics['calmar_ratio'] = abs(metrics['annualized_return'] / metrics['max_drawdown'])
            else:
                metrics['calmar_ratio'] = 0
        else:
            metrics['sharpe_ratio'] = 0
            metrics['sortino_ratio'] = 0
            metrics['calmar_ratio'] = 0
        
        return metrics
    
    @staticmethod
    def display_metrics(metrics: Dict[str, Any], title: str = "Strategy Performance") -> str:
        """
        Format metrics into a human-readable table.
        
        Args:
            metrics: Dictionary of performance metrics
            title: Title for the display
            
        Returns:
            str: Formatted table
        """
        def format_value(key, value):
            """Format values with appropriate precision and unit."""
            if key in ['initial_equity', 'final_equity', 'peak_equity', 'min_equity', 'total_pnl',
                      'avg_trade_pnl', 'avg_win', 'max_win', 'avg_loss', 'max_loss']:
                return f"${value:.2f}"
            elif key in ['total_return', 'annualized_return', 'max_drawdown', 'win_rate']:
                return f"{value:.2f}%"
            elif key in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'profit_factor',
                        'expectancy']:
                return f"{value:.2f}"
            elif key in ['trade_count', 'win_count', 'loss_count']:
                return f"{int(value)}"
            else:
                return value
        
        # Organize metrics into sections
        sections = [
            ("Return Metrics", [
                ("Initial Equity", format_value('initial_equity', metrics['initial_equity'])),
                ("Final Equity", format_value('final_equity', metrics['final_equity'])),
                ("Total Return", format_value('total_return', metrics['total_return'])),
                ("Annualized Return", format_value('annualized_return', metrics['annualized_return'])),
                ("Sharpe Ratio", format_value('sharpe_ratio', metrics['sharpe_ratio'])),
                ("Max Drawdown", format_value('max_drawdown', metrics['max_drawdown'])),
            ]),
            ("Trade Metrics", [
                ("Total Trades", format_value('trade_count', metrics['trade_count'])),
                ("Win Rate", format_value('win_rate', metrics['win_rate'])),
                ("Avg Win", format_value('avg_win', metrics['avg_win'])),
                ("Avg Loss", format_value('avg_loss', metrics['avg_loss'])),
                ("Profit Factor", format_value('profit_factor', metrics['profit_factor'])),
                ("Expectancy", format_value('expectancy', metrics['expectancy']))
            ])
        ]
        
        # Build table data
        table_data = []
        for section, items in sections:
            for item in items:
                table_data.append([item[0], item[1]])
        
        # Create table
        return tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid")
    
    @staticmethod
    def format_walk_forward_results(wf_result: Dict[str, Any]) -> str:
        """
        Format walk-forward optimization results.
        
        Args:
            wf_result: Walk-forward optimization result dictionary
            
        Returns:
            str: Formatted results
        """
        output = []
        output.append("\nWalk-Forward Optimization Results\n")
        
        # Extract best parameters
        best_params = wf_result.get('best_params', {})
        window_results = wf_result.get('window_results', [])
        
        # Format best parameters
        param_table = []
        for param, value in best_params.items():
            param_table.append([param, value])
        
        output.append("Best Overall Parameters:")
        output.append(tabulate(param_table, headers=["Parameter", "Value"], tablefmt="grid"))
        output.append("")
        
        # Format window results
        if window_results:
            window_table = []
            for window in window_results:
                window_idx = window.get('window', 0)
                train_period = window.get('train_period', ('unknown', 'unknown'))
                test_period = window.get('test_period', ('unknown', 'unknown'))
                
                # Format train period
                if isinstance(train_period[0], (datetime.datetime, pd.Timestamp)):
                    train_start = train_period[0].strftime('%Y-%m-%d')
                else:
                    train_start = str(train_period[0])
                    
                if isinstance(train_period[1], (datetime.datetime, pd.Timestamp)):
                    train_end = train_period[1].strftime('%Y-%m-%d')
                else:
                    train_end = str(train_period[1])
                
                # Format test period
                if isinstance(test_period[0], (datetime.datetime, pd.Timestamp)):
                    test_start = test_period[0].strftime('%Y-%m-%d')
                else:
                    test_start = str(test_period[0])
                    
                if isinstance(test_period[1], (datetime.datetime, pd.Timestamp)):
                    test_end = test_period[1].strftime('%Y-%m-%d')
                else:
                    test_end = str(test_period[1])
                
                # Format parameters into a string
                params = window.get('params', {})
                param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                
                # Add row to table
                window_table.append([
                    window_idx,
                    f"{train_start} to {train_end}",
                    f"{test_start} to {test_end}",
                    window.get('train_score', 0),
                    window.get('test_score', 0),
                    param_str
                ])
            
            output.append("Window Results:")
            output.append(tabulate(window_table, 
                headers=["Window", "Train Period", "Test Period", "Train Score", "Test Score", "Parameters"],
                tablefmt="grid"))
        
        return "\n".join(output)
    
    @staticmethod
    def plot_equity_curve(equity_curve: pd.DataFrame, trades: List[Dict[str, Any]] = None, 
                        title: str = "Equity Curve", figsize: Tuple[int, int] = (12, 6)):
        """
        Plot equity curve with trade markers.
        
        Args:
            equity_curve: DataFrame with timestamp and equity columns
            trades: Optional list of trades to mark on the curve
            title: Plot title
            figsize: Figure size (width, height)
            
        Returns:
            matplotlib.figure.Figure: Plot figure
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot equity curve
            ax.plot(equity_curve['timestamp'], equity_curve['equity'], label='Equity')
            
            # Mark trades if provided
            if trades:
                # Convert trade timestamps to datetime if needed
                trade_times = []
                trade_equities = []
                trade_colors = []
                
                for trade in trades:
                    if 'timestamp' in trade and 'price' in trade:
                        # Find closest equity point
                        timestamp = pd.to_datetime(trade['timestamp'])
                        trade_times.append(timestamp)
                        
                        # Find equity at this point
                        idx = equity_curve['timestamp'].searchsorted(timestamp)
                        idx = min(idx, len(equity_curve) - 1)
                        equity = equity_curve['equity'].iloc[idx]
                        trade_equities.append(equity)
                        
                        # Determine color by direction
                        if trade.get('direction') == 'BUY':
                            trade_colors.append('green')
                        else:
                            trade_colors.append('red')
                
                # Plot trade markers
                if trade_times:
                    ax.scatter(trade_times, trade_equities, c=trade_colors, marker='^', s=50, alpha=0.7)
            
            # Format plot
            ax.set_title(title)
            ax.set_xlabel('Date')
            ax.set_ylabel('Equity ($)')
            ax.grid(True, alpha=0.3)
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            # If matplotlib is not available, return None
            print("Matplotlib is required for plotting.")
            return None
