import pandas as pd
import numpy as np
from tabulate import tabulate  # Make sure to install this package (pip install tabulate)

class PerformanceAnalytics:
    """Analytics module for evaluating trading strategy performance."""
    
    @staticmethod
    def calculate_metrics(equity_curve, trades=None):
        """Calculate key performance metrics from equity curve."""
        if len(equity_curve) < 2:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'trades': 0,
                'win_rate': 0.0
            }
        
        # Calculate returns
        equity_curve['returns'] = equity_curve['equity'].pct_change().fillna(0)
        
        # Initial and final equity
        initial_equity = equity_curve['equity'].iloc[0]
        final_equity = equity_curve['equity'].iloc[-1]
        
        # Total return
        total_return = (final_equity / initial_equity) - 1
        
        # Annualized return (assuming 252 trading days)
        days = (equity_curve['timestamp'].iloc[-1] - equity_curve['timestamp'].iloc[0]).days
        if days > 0:
            if total_return <= -1:  # Complete loss of capital
                annualized_return = -1.0  # -100%
            else:
                # For negative returns that aren't complete losses
                if total_return < 0:
                    # Alternative calculation for negative returns
                    # Using log returns, which handles negative values properly
                    # import numpy as np
                    log_return = np.log(1 + total_return)
                    annualized_log_return = log_return * (252 / days)
                    annualized_return = np.exp(annualized_log_return) - 1
                else:
                    # Standard calculation for positive returns
                    annualized_return = (1 + total_return) ** (252 / days) - 1
        else:
            annualized_return = 0
        
        # Sharpe ratio (assuming risk-free rate of 0)
        daily_returns = equity_curve['returns']
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / max(daily_returns.std(), 1e-8)
        
        # Maximum drawdown
        equity_curve['peak'] = equity_curve['equity'].cummax()
        equity_curve['drawdown'] = (equity_curve['equity'] / equity_curve['peak']) - 1
        max_drawdown = equity_curve['drawdown'].min()
        
        # Trade statistics
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0
        
        if trades is not None and isinstance(trades, list) and len(trades) > 0:
            # Calculate trade statistics
            wins = [t for t in trades if t.get('pnl', 0) > 0]
            losses = [t for t in trades if t.get('pnl', 0) <= 0]
            
            win_rate = len(wins) / len(trades) if len(trades) > 0 else 0
            avg_win = sum(t.get('pnl', 0) for t in wins) / len(wins) if len(wins) > 0 else 0
            avg_loss = sum(t.get('pnl', 0) for t in losses) / len(losses) if len(losses) > 0 else 0
            
            gross_profit = sum(t.get('pnl', 0) for t in wins)
            gross_loss = abs(sum(t.get('pnl', 0) for t in losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Compile all metrics
        metrics = {
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': len(trades) if trades else 0,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
        
        return metrics
    
    @staticmethod
    def display_metrics(metrics, title="Strategy Performance Metrics"):
        """Generate a formatted string table of performance metrics."""
        # Format metrics for display
        display_data = [
            ["Initial Equity", f"${metrics['initial_equity']:.2f}"],
            ["Final Equity", f"${metrics['final_equity']:.2f}"],
            ["Total Return", f"{metrics['total_return']:.2%}"],
            ["Annualized Return", f"{metrics['annualized_return']:.2%}"],
            ["Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}"],
            ["Max Drawdown", f"{metrics['max_drawdown']:.2%}"],
            ["Total Trades", f"{metrics['trades']}"],
            ["Win Rate", f"{metrics['win_rate']:.2%}"],
            ["Avg Win", f"${metrics['avg_win']:.2f}"],
            ["Avg Loss", f"${metrics['avg_loss']:.2f}"],
            ["Profit Factor", f"{metrics['profit_factor']:.2f}"]
        ]
        
        # Create table with tabulate
        table = tabulate(display_data, headers=["Metric", "Value"], tablefmt="grid")
        
        return f"\n{title}\n{table}\n"
    
    @staticmethod
    def format_optimization_results(optimization_result, title="Optimization Results"):
        """Format optimization results for display."""
        best_params = optimization_result.get('best_params', {})
        best_score = optimization_result.get('best_score', 0)
        
        # Create table with parameter names and values
        param_data = [[param, value] for param, value in best_params.items()]
        param_table = tabulate(param_data, headers=["Parameter", "Value"], tablefmt="grid")
        
        # Top results table
        top_results = optimization_result.get('all_results', [])
        if top_results:
            # Sort by score (descending)
            top_results = sorted(top_results, key=lambda x: x.get('score', 0), reverse=True)[:5]
            
            top_data = []
            for i, result in enumerate(top_results):
                params_str = ", ".join(f"{k}={v}" for k, v in result.get('params', {}).items())
                top_data.append([i+1, f"{result.get('score', 0):.4f}", params_str])
            
            top_table = tabulate(top_data, headers=["Rank", "Score", "Parameters"], tablefmt="grid")
        else:
            top_table = "No detailed results available"
        
        return f"\n{title}\n\nBest Score: {best_score:.4f}\n\nBest Parameters:\n{param_table}\n\nTop 5 Results:\n{top_table}\n"
    
    @staticmethod
    def format_walk_forward_results(wf_result, title="Walk-Forward Optimization Results"):
        """Format walk-forward optimization results for display."""
        best_params = wf_result.get('best_params', {})
        
        # Create table with parameter names and values
        param_data = [[param, value] for param, value in best_params.items()]
        param_table = tabulate(param_data, headers=["Parameter", "Value"], tablefmt="grid")
        
        # Window results
        windows = wf_result.get('window_results', [])
        if windows:
            window_data = []
            for i, window in enumerate(windows):
                train_period = window.get('train_period', ('N/A', 'N/A'))
                test_period = window.get('test_period', ('N/A', 'N/A'))
                
                # Format dates for display
                train_start = train_period[0].strftime('%Y-%m-%d') if hasattr(train_period[0], 'strftime') else str(train_period[0])
                train_end = train_period[1].strftime('%Y-%m-%d') if hasattr(train_period[1], 'strftime') else str(train_period[1])
                test_start = test_period[0].strftime('%Y-%m-%d') if hasattr(test_period[0], 'strftime') else str(test_period[0])
                test_end = test_period[1].strftime('%Y-%m-%d') if hasattr(test_period[1], 'strftime') else str(test_period[1])
                
                params_str = ", ".join(f"{k}={v}" for k, v in window.get('params', {}).items())
                
                window_data.append([
                    i+1, 
                    f"{train_start} to {train_end}", 
                    f"{test_start} to {test_end}",
                    f"{window.get('train_score', 0):.4f}",
                    f"{window.get('test_score', 0):.4f}",
                    params_str
                ])
            
            window_table = tabulate(window_data, 
                                    headers=["Window", "Train Period", "Test Period", 
                                            "Train Score", "Test Score", "Parameters"], 
                                    tablefmt="grid")
        else:
            window_table = "No window results available"
        
        return f"\n{title}\n\nBest Overall Parameters:\n{param_table}\n\nWindow Results:\n{window_table}\n"
