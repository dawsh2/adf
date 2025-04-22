"""
Regime-Based Optimization Module

This module provides optimizers that tune strategy parameters specifically
for different market regimes, enabling adaptive strategy behavior.
"""
import logging
import itertools
import datetime
import traceback
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable, Union

from .optimizer_base import OptimizerBase
from .grid_search import GridSearchOptimizer
from ..filters.regime.regime_detector import RegimeDetectorBase, MarketRegime

logger = logging.getLogger(__name__)

class RegimeSpecificOptimizer:
    """
    Optimizer that tunes strategy parameters specifically for each market regime.
    
    This optimizer analyzes historical data to identify different market regimes,
    then optimizes strategy parameters separately for each regime to maximize
    performance in each specific market condition.
    """
    
    def __init__(self, regime_detector: RegimeDetectorBase, grid_optimizer: OptimizerBase = None):
        """
        Initialize the regime-specific optimizer.
        
        Args:
            regime_detector: Regime detector instance
            grid_optimizer: Optional grid search optimizer instance
        """
        self.regime_detector = regime_detector
        self.grid_optimizer = grid_optimizer or GridSearchOptimizer()
        self.results = {}
        self.all_param_results = []  # Track all parameter combinations evaluated
        self.optimization_scores = {}  # Track scores by regime


    def optimize(self, param_grid: Dict[str, List[Any]], 
                data_handler, evaluation_func: Callable, 
                start_date=None, end_date=None, min_regime_bars=30,
                optimize_metric='sharpe_ratio', min_improvement=0.1) -> Dict[MarketRegime, Dict[str, Any]]:
        """
        Perform regime-specific optimization.

        Args:
            param_grid: Dictionary of parameter names to possible values
            data_handler: Data handler with loaded data
            evaluation_func: Function to evaluate parameter combinations
            start_date: Start date for optimization
            end_date: End date for optimization
            min_regime_bars: Minimum bars required for a regime to be optimized
            optimize_metric: Metric to optimize ('sharpe_ratio', 'return', 'max_drawdown')
            min_improvement: Minimum improvement required to use regime-specific parameters

        Returns:
            dict: Best parameters for each regime
        """
        symbol = data_handler.get_symbols()[0]  # Use first symbol

        # 1. Run regime detection on historical data
        logger.info(f"Performing regime detection on historical data for {symbol}")
        self._detect_regimes(data_handler, start_date, end_date)

        # Print a summary of detected regimes
        self.regime_detector.print_regime_summary(symbol)

        # 2. Segment data by regime
        regime_periods = self.regime_detector.get_regime_periods(symbol, start_date, end_date)

        # Print periods for each regime
        logger.info("\nRegime Periods:")
        for regime, periods in regime_periods.items():
            total_days = sum((end - start).total_seconds() / (24 * 3600) for start, end in periods)
            logger.info(f"  {regime.value}: {len(periods)} periods, {total_days:.1f} days")

        # 3. First, optimize parameters on the entire dataset (baseline)
        logger.info("\n--- Optimizing Baseline Parameters (All Regimes) ---")
        baseline_params, baseline_score = self._grid_search(
            param_grid, 
            data_handler,
            evaluation_func,
            start_date, 
            end_date,
            optimize_metric
        )

        logger.info(f"Baseline parameters: {baseline_params}, Score: {baseline_score:.4f}")

        # 4. For each regime, optimize parameters
        regime_params = {
            MarketRegime.UNKNOWN: baseline_params  # Default to baseline for unknown
        }

        for regime in list(MarketRegime):
            if regime == MarketRegime.UNKNOWN:
                continue  # Already set to baseline

            if regime not in regime_periods or not regime_periods[regime]:
                logger.info(f"No data for {regime.value} regime, using baseline parameters")
                regime_params[regime] = baseline_params
                continue

            # Count total bars in this regime
            try:
                periods = regime_periods[regime]
                bar_count = self._count_bars_in_periods(data_handler, symbol, periods)

                if bar_count < min_regime_bars:
                    logger.info(f"Skipping optimization for {regime.value} - insufficient data ({bar_count} bars)")
                    regime_params[regime] = baseline_params
                    continue

                logger.info(f"\n--- Optimizing for {regime.value} regime ({bar_count} bars) ---")

                # First evaluate baseline parameters on this regime's data
                baseline_regime_score = self._evaluate_in_periods(
                    baseline_params, 
                    data_handler, 
                    evaluation_func, 
                    periods, 
                    optimize_metric
                )

                # Optimize for this regime
                regime_best_params, regime_score = self._optimize_for_regime(
                    param_grid, data_handler, evaluation_func, regime, periods, optimize_metric
                )

                # Store scores for reporting
                self.optimization_scores[regime] = regime_score

                # Check if regime-specific parameters are better than baseline ON THE SAME REGIME DATA
                improvement = (regime_score - baseline_regime_score) / abs(baseline_regime_score) if baseline_regime_score != 0 else float('inf')

                if improvement >= min_improvement:
                    logger.info(f"Best parameters for {regime.value}: {regime_best_params}, "
                              f"Score: {regime_score:.4f} (Improvement: {improvement:.2%} over baseline {baseline_regime_score:.4f})")
                    regime_params[regime] = regime_best_params
                else:
                    logger.info(f"Parameters for {regime.value} not better than baseline "
                              f"(Score: {regime_score:.4f} vs baseline {baseline_regime_score:.4f}, Improvement: {improvement:.2%})")
                    regime_params[regime] = baseline_params
            except Exception as e:
                logger.error(f"Error optimizing for {regime.value}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                regime_params[regime] = baseline_params

        # Store and return results
        self.results = {
            'regime_parameters': regime_params,
            'baseline_parameters': baseline_params,
            'baseline_score': baseline_score,
            'regime_periods': {r.value: periods for r, periods in regime_periods.items() if r in regime_periods},
            'all_param_results': self.all_param_results
        }

        # Add regime scores to results
        for regime, score in self.optimization_scores.items():
            self.results[f'{regime.value}_score'] = score

        return regime_params        



    def _detect_regimes(self, data_handler, start_date, end_date):
        """
        Run regime detection on historical data.
        
        Args:
            data_handler: Data handler with loaded data
            start_date: Optional start date for analysis
            end_date: Optional end date for analysis
        """
        symbol = data_handler.get_symbols()[0]

        # Reset data handler to start of data
        data_handler.reset()

        # Reset detector
        self.regime_detector.reset()

        # Convert string dates to pandas Timestamp objects if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # Process each bar to detect regimes
        bar_count = 0
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break

            # Get bar timestamp
            bar_timestamp = bar.get_timestamp()

            # Skip bars outside date range, handling timezone differences
            if start_date:
                # Make timestamp comparison timezone-consistent
                if (hasattr(bar_timestamp, 'tzinfo') and bar_timestamp.tzinfo is not None and 
                    hasattr(start_date, 'tzinfo') and start_date.tzinfo is None):
                    # Bar has timezone but start_date doesn't, so convert start_date
                    start_date = start_date.tz_localize(bar_timestamp.tzinfo)
                elif (hasattr(bar_timestamp, 'tzinfo') and bar_timestamp.tzinfo is None and 
                     hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None):
                    # Start_date has timezone but bar doesn't, so convert bar timestamp
                    bar_ts_aware = bar_timestamp.tz_localize(start_date.tzinfo)
                    if bar_ts_aware < start_date:
                        continue
                elif bar_timestamp < start_date:
                    continue

            if end_date:
                # Make timestamp comparison timezone-consistent
                if (hasattr(bar_timestamp, 'tzinfo') and bar_timestamp.tzinfo is not None and 
                    hasattr(end_date, 'tzinfo') and end_date.tzinfo is None):
                    # Bar has timezone but end_date doesn't, so convert end_date
                    end_date = end_date.tz_localize(bar_timestamp.tzinfo)
                elif (hasattr(bar_timestamp, 'tzinfo') and bar_timestamp.tzinfo is None and 
                     hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None):
                    # End_date has timezone but bar doesn't, so convert bar timestamp
                    bar_ts_aware = bar_timestamp.tz_localize(end_date.tzinfo)
                    if bar_ts_aware > end_date:
                        break
                elif bar_timestamp > end_date:
                    break

            # Detect regime
            self.regime_detector.update(bar)
            bar_count += 1

            if bar_count % 100 == 0:
                logger.debug(f"Processed {bar_count} bars for regime detection")

        logger.info(f"Processed {bar_count} bars for regime detection")

        # Reset data handler for optimization
        data_handler.reset()

    def _count_bars_in_periods(self, data_handler, symbol, periods):
        """
        Count total bars across multiple time periods.
        
        Args:
            data_handler: Data handler with loaded data
            symbol: Symbol to count bars for
            periods: List of (start, end) tuples defining time periods
            
        Returns:
            int: Total bar count
        """
        # Reset data handler
        data_handler.reset()

        total_bars = 0

        # Process each bar
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break

            # Get bar timestamp
            timestamp = bar.get_timestamp()

            for start, end in periods:
                # Make timestamps compatible for comparison
                start_ts = start.replace(tzinfo=None) if hasattr(start, 'tzinfo') and start.tzinfo else start
                end_ts = end.replace(tzinfo=None) if hasattr(end, 'tzinfo') and end.tzinfo else end
                bar_ts = timestamp.replace(tzinfo=None) if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo else timestamp

                # Compare with compatible timestamps
                if start_ts <= bar_ts <= end_ts:
                    total_bars += 1
                    break

        # Reset data handler
        data_handler.reset()

        return total_bars

    def _optimize_for_regime(self, param_grid, data_handler, evaluation_func, regime, periods, optimize_metric):
        """
        Optimize parameters for a specific regime.

        Args:
            param_grid: Dictionary of parameter names to possible values
            data_handler: Data handler with loaded data
            evaluation_func: Function to evaluate parameter combinations
            regime: MarketRegime to optimize for
            periods: List of (start, end) tuples defining time periods for this regime
            optimize_metric: Metric to optimize

        Returns:
            tuple: (best_params, best_score, baseline_regime_score)
        """
        # First, evaluate baseline parameters on regime data
        baseline_params = self.results.get('baseline_parameters', {})
        baseline_regime_score = self._evaluate_in_periods(
            baseline_params, data_handler, evaluation_func, periods, optimize_metric
        )

        # Define regime-specific evaluation wrapper
        def regime_evaluation(params):
            """
            Evaluate parameters only on bars in this regime.

            Args:
                params: Parameters to evaluate

            Returns:
                float: Performance score
            """
            # Evaluate only on bars in this regime
            result = self._evaluate_in_periods(
                params, data_handler, evaluation_func, periods, optimize_metric
            )
            return result

        # Run grid search with regime-specific evaluation
        try:
            param_combinations = []
            for name, values in param_grid.items():
                param_combinations.append(values)

            logger.info(f"Grid search: evaluating {len(list(itertools.product(*param_combinations)))} parameter combinations")

            # Run grid search
            result = self.grid_optimizer.optimize(param_grid, regime_evaluation)
            return result['best_params'], result['best_score'], baseline_regime_score
        except Exception as e:
            logger.error(f"Error in regime optimization: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise    



    def _evaluate_in_periods(self, params, data_handler, evaluation_func, periods, optimize_metric='sharpe_ratio'):
        """
        Evaluate parameters only on bars within specific time periods.

        This function extracts the bars within the specified periods and creates
        a mini-backtest environment to evaluate the parameters.
        
        Args:
            params: Parameters to evaluate
            data_handler: Data handler with loaded data
            evaluation_func: Function to evaluate parameter combinations
            periods: List of (start, end) tuples defining time periods
            optimize_metric: Metric to optimize
            
        Returns:
            float: Performance score
        """
        # Reset data handler
        data_handler.reset()

        symbol = data_handler.get_symbols()[0]

        # Get the time range from periods
        min_start = None
        max_end = None
        for start, end in periods:
            start_naive = start.replace(tzinfo=None) if hasattr(start, 'tzinfo') and start.tzinfo else start
            end_naive = end.replace(tzinfo=None) if hasattr(end, 'tzinfo') and end.tzinfo else end

            if min_start is None or start_naive < min_start:
                min_start = start_naive
            if max_end is None or end_naive > max_end:
                max_end = end_naive

        # If no periods, return low score
        if min_start is None or max_end is None:
            return float('-inf')

        # Now we have the full date range to test
        try:
            # Call evaluation_func with the date range parameters it expects
            result = evaluation_func(params, data_handler, min_start, max_end)
            
            # If result is a dict, extract the metric
            if isinstance(result, dict):
                score = result.get(optimize_metric, float('-inf'))
                
                # Handle metrics where lower is better (like drawdown)
                if optimize_metric == 'max_drawdown':
                    score = -score  # Negate drawdown so lower values are better
                    
                # Store result for reporting
                self.all_param_results.append({
                    'params': params,
                    'score': score,
                    'regime': str(regime) if 'regime' in locals() else 'unknown'
                })
            else:
                # Assume result is already the score
                score = result
                
                # Store result for reporting
                self.all_param_results.append({
                    'params': params,
                    'score': score,
                    'regime': str(regime) if 'regime' in locals() else 'unknown'
                })
                
            return score
        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return float('-inf')

    def _grid_search(self, param_grid, data_handler, evaluation_func, start_date, end_date, optimize_metric):
        """
        Perform grid search on a specific time period.
        
        Args:
            param_grid: Dictionary of parameters to test
            data_handler: Data handler with loaded data
            evaluation_func: Function to evaluate parameters
            start_date: Start date for training
            end_date: End date for training
            optimize_metric: Metric to optimize
            
        Returns:
            tuple: (best_params, best_score)
        """
        # Define evaluation wrapper for the metric
        def metric_evaluation(params):
            """
            Evaluate parameters and extract specific metric.
            
            Args:
                params: Parameters to evaluate
                
            Returns:
                float: Performance score
            """
            result = evaluation_func(params, data_handler, start_date, end_date)
            
            if isinstance(result, dict):
                # If result is a dict, extract the metric
                score = result.get(optimize_metric, float('-inf'))
                
                # Handle metrics where lower is better
                if optimize_metric == 'max_drawdown':
                    score = -score  # Negate drawdown so lower values are better
                    
                # Store result for reporting
                self.all_param_results.append({
                    'params': params,
                    'score': score,
                    'regime': 'baseline'
                })
            else:
                # Assume result is already the score
                score = result
                
                # Store result for reporting
                self.all_param_results.append({
                    'params': params,
                    'score': score,
                    'regime': 'baseline'
                })
                
            return score
        
        # Generate all parameter combinations for logging
        param_combinations = []
        for name, values in param_grid.items():
            param_combinations.append(values)
        
        logger.info(f"Grid search: evaluating {len(list(itertools.product(*param_combinations)))} parameter combinations")
        
        # Run grid search
        try:
            result = self.grid_optimizer.optimize(param_grid, metric_evaluation)
            return result['best_params'], result['best_score']
        except Exception as e:
            logger.error(f"Error in grid search: {e}")
            raise

    def generate_report(self, data_handler, symbol):
        """
        Generate a comprehensive report of optimization results.
        
        Args:
            data_handler: Data handler with loaded data
            symbol: Symbol being analyzed
            
        Returns:
            str: Text report
        """
        # [Report generation code omitted for brevity]

    def generate_report(self, data_handler, symbol):
        """
        Generate a comprehensive report of optimization results.
        
        Args:
            data_handler: Data handler with loaded data
            symbol: Symbol being analyzed
            
        Returns:
            str: Text report
        """
        if not self.results:
            return "No optimization results available."
            
        regime_params = self.results.get('regime_parameters', {})
        baseline_params = self.results.get('baseline_parameters', {})
        baseline_score = self.results.get('baseline_score', 0)
        
        report = [
            "\n" + "="*80,
            "             COMPREHENSIVE OPTIMIZATION REPORT",
            "="*80,
            
            "\n1. REGIME DISTRIBUTION ANALYSIS",
            "-"*50
        ]
        
        # Get regime periods and calculate stats
        regime_periods = self.regime_detector.get_regime_periods(symbol)
        
        # Count bars in each regime
        regime_bars = {}
        total_bars = 0
        
        for regime, periods in regime_periods.items():
            bar_count = self._count_bars_in_periods(data_handler, symbol, periods)
            regime_bars[regime] = bar_count
            total_bars += bar_count
        
        report.append(f"Total trading periods analyzed: {sum(len(periods) for periods in regime_periods.values())}")
        report.append(f"Estimated total bars: {total_bars}")
        report.append("\nRegime Distribution:")
        
        for regime, count in sorted(regime_bars.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_bars * 100) if total_bars > 0 else 0
            bar_length = int(pct / 2)  # Scale to make it fit in console
            bar = "█" * bar_length
            report.append(f"  {regime.value:<10}: {count:>6} bars ({pct:>6.2f}%) {bar}")
        
        report.append("\n2. OPTIMIZATION RESULTS BY REGIME")
        report.append("-"*50)
        
        report.append(f"Baseline Parameters: {baseline_params}")
        report.append(f"Baseline Performance Score: {baseline_score:.4f}")
        report.append("\nRegime-Specific Parameters:")
        
        # Table header
        report.append("\n{:<12} {:<20} {:<15} {:<15} {:<15}".format(
            "Regime", "Parameters", "Score", "vs Baseline", "Bars"
        ))
        report.append("-"*80)
        
        # Display each regime's results
        for regime in sorted(regime_bars.keys(), key=lambda r: regime_bars.get(r, 0), reverse=True):
            if regime in regime_params:
                params = regime_params[regime]
                is_baseline = params == baseline_params
                
                # Get performance metrics if available
                score_key = f'{regime.value}_score'
                score = self.results.get(score_key, baseline_score if is_baseline else "N/A")
                
                if isinstance(score, (int, float)):
                    vs_baseline = f"{((score / baseline_score) - 1) * 100:.2f}%" if baseline_score != 0 else "N/A"
                else:
                    vs_baseline = "N/A"
                
                params_str = f"fast={params.get('fast_window')}, slow={params.get('slow_window')}"
                report.append("{:<12} {:<20} {:<15} {:<15} {:<15}".format(
                    regime.value, 
                    params_str, 
                    f"{score:.4f}" if isinstance(score, (int, float)) else score,
                    vs_baseline,
                    f"{regime_bars.get(regime, 0):,}"
                ))
        
        report.append("\n3. PARAMETER PERFORMANCE HEATMAP")
        report.append("-"*50)
        
        # Extract parameter grid from results
        all_params_results = self.all_param_results
        if not all_params_results:
            report.append("Parameter performance data not available.")
        else:
            # Find unique parameter values
            param_sets = set()
            for result in all_params_results:
                param_tuple = tuple(sorted(result['params'].items()))
                param_sets.add(param_tuple)
            
            report.append(f"Total parameter combinations evaluated: {len(param_sets)}")
            
            # Find best parameters overall
            best_result = max(all_params_results, key=lambda x: x['score'])
            report.append(f"Best overall parameters: {best_result['params']} (Score: {best_result['score']:.4f})")
            
            # Find best parameters by regime
            regime_best = {}
            for result in all_params_results:
                regime = result.get('regime', 'unknown')
                if regime not in regime_best or result['score'] > regime_best[regime]['score']:
                    regime_best[regime] = result
            
            report.append("\nBest parameters by regime:")
            for regime, result in sorted(regime_best.items()):
                report.append(f"  {regime}: {result['params']} (Score: {result['score']:.4f})")
        
        report.append("\n4. KEY FINDINGS & RECOMMENDATIONS")
        report.append("-"*50)
        
        # Find best overall parameters
        best_regime = None
        best_score = baseline_score
        
        for regime in regime_params:
            score_key = f'{regime.value}_score'
            score = self.results.get(score_key, 0)
            if isinstance(score, (int, float)) and score > best_score:
                best_score = score
                best_regime = regime
        
        # Generate insights
        report.append("Key Insights:")
        
        if best_regime and best_score > baseline_score:
            report.append(f"• Best performance achieved in {best_regime.value} regime: {best_score:.4f} " +
                        f"({((best_score/baseline_score)-1)*100:.2f}% above baseline)")
        else:
            report.append("• Baseline parameters performed best overall - regime-specific optimization did not improve performance")
        
        # Most frequent regime
        most_frequent = max(regime_bars.items(), key=lambda x: x[1])[0] if regime_bars else None
        if most_frequent:
            report.append(f"• Most frequent regime: {most_frequent.value} ({regime_bars.get(most_frequent, 0)} bars, " +
                        f"{regime_bars.get(most_frequent, 0)/total_bars*100:.2f}% of data)")
        
        # Analyze parameter trends
        fast_trend = {}
        slow_trend = {}
        
        for regime, params in regime_params.items():
            if params != baseline_params:  # Only consider optimized params
                fast = params.get('fast_window')
                slow = params.get('slow_window')
                if fast:
                    fast_trend[regime] = fast
                if slow:
                    slow_trend[regime] = slow
        
        if fast_trend:
            report.append("\nParameter Trends by Regime:")
            for regime, fast in fast_trend.items():
                slow = slow_trend.get(regime)
                report.append(f"• {regime.value}: Prefers {'faster' if fast < baseline_params.get('fast_window', 0) else 'slower'} " +
                            f"fast MA ({fast} vs {baseline_params.get('fast_window')}) and " +
                            f"{'wider' if slow > baseline_params.get('slow_window', 0) else 'narrower'} " +
                            f"slow MA ({slow} vs {baseline_params.get('slow_window')})")
        
        # General recommendations
        report.append("\nRecommendations:")
        
        if most_frequent and most_frequent.value == "sideways" and regime_bars.get(most_frequent, 0)/total_bars > 0.8:
            report.append("• Data appears predominantly sideways - consider using mean-reversion strategies instead of trend-following")
        
        if best_regime and best_score > baseline_score * 1.2:  # 20% improvement
            report.append(f"• Consider using regime-specific parameters for {best_regime.value} periods")
        else:
            report.append("• Stick with baseline parameters as regime-specific optimization shows limited benefit")
        
        if sum(1 for _, bars in regime_bars.items() if bars > 100) < 3:  # Less than 3 regimes with sufficient data
            report.append("• Insufficient data in some regimes for reliable optimization - collect more data or adjust regime detection")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)


class RegimeOptimizer(OptimizerBase):
    """
    Optimizer that combines regime detection with parameter optimization.
    
    This optimizer extends the standard OptimizerBase interface to provide
    regime-specific optimization within the optimization framework.
    """
    
    def __init__(self, regime_detector: RegimeDetectorBase, base_optimizer: OptimizerBase,
                min_regime_bars=50, name="regime_optimizer"):
        """
        Initialize the regime optimizer.
        
        Args:
            regime_detector: Regime detector instance
            base_optimizer: Base optimizer to use for each regime
            min_regime_bars: Minimum bars required for a regime to be optimized
            name: Optimizer name
        """
        super().__init__(name)
        self.regime_detector = regime_detector
        self.base_optimizer = base_optimizer
        self.min_regime_bars = min_regime_bars
        self.results = {}
    
    def optimize(self, param_space: Dict[str, List[Any]], 
                fitness_function: Callable[[Dict[str, Any]], float],
                constraints: Optional[List[Callable[[Dict[str, Any]], bool]]] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Perform regime-specific optimization.

        Args:
            param_space: Dictionary mapping parameter names to possible values
            fitness_function: Function that evaluates parameter sets
            constraints: Optional list of constraint functions
            **kwargs: Additional parameters including:
                - data_handler: Required data handler with loaded data
                - start_date: Optional start date for optimization
                - end_date: Optional end date for optimization

        Returns:
            Dictionary with optimization results
        """
        # Extract required arguments
        data_handler = kwargs.get('data_handler')
        if data_handler is None:
            raise ValueError("data_handler is required for regime optimization")
        
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        optimize_metric = kwargs.get('optimize_metric', 'score')
        
        # Get symbols from data handler
        symbols = data_handler.get_symbols()
        if not symbols:
            raise ValueError("No symbols available in data handler")
        
        symbol = symbols[0]  # Use first symbol
        
        logger.info(f"Starting regime-based optimization for {symbol}")
        
        # 1. Perform regime detection on historical data
        logger.info(f"Performing regime detection on historical data")
        self._detect_regimes(data_handler, start_date, end_date)
        
        # 2. Segment data by regime
        regime_periods = self.regime_detector.get_regime_periods(symbol, start_date, end_date)
        
        # Log detected regimes
        for regime, periods in regime_periods.items():
            total_days = sum((end - start).total_seconds() / (24 * 3600) for start, end in periods) if periods else 0
            logger.info(f"  {regime.value}: {len(periods)} periods, {total_days:.1f} days")
        
        # 3. For each regime, optimize parameters
        regime_params = {}
        regime_scores = {}
        
        for regime, periods in regime_periods.items():
            if not periods:
                continue
                
            # Count total bars in this regime
            bar_count = self._count_bars_in_periods(data_handler, symbol, periods)
            
            if bar_count < self.min_regime_bars:
                logger.info(f"Skipping optimization for {regime.value} - insufficient data ({bar_count} bars)")
                continue
                
            logger.info(f"\n--- Optimizing for {regime.value} regime ({bar_count} bars) ---")
            
            # Optimize for this regime
            best_params, best_score = self._optimize_for_regime(
                param_space, 
                fitness_function, 
                constraints,
                regime, 
                periods, 
                data_handler,
                **kwargs
            )
            
            if best_params:
                logger.info(f"Best parameters for {regime.value}: {best_params}, Score: {best_score:.4f}")
                regime_params[regime] = best_params
                regime_scores[regime] = best_score
            else:
                logger.warning(f"Optimization failed for {regime.value}")
        
        # 4. Set default parameters if any regime is missing
        all_regimes = list(MarketRegime)
        for regime in all_regimes:
            if regime not in regime_params:
                # Use parameters from most similar regime or average of others
                regime_params[regime] = self._get_default_params(regime, regime_params)
                
        # Compile results
        optimization_result = {
            'regime_parameters': {r.value: p for r, p in regime_params.items()},
            'regime_scores': {r.value: s for r, s in regime_scores.items()},
            'regime_periods': {r.value: len(periods) for r, periods in regime_periods.items() if periods}
        }
        
        # Find best overall parameters (highest score)
        if regime_scores:
            best_regime = max(regime_scores.items(), key=lambda x: x[1])[0]
            best_score = regime_scores[best_regime]
            best_params = regime_params[best_regime]
        else:
            # No regimes optimized, use base optimizer on full dataset
            logger.info("\n--- Performing baseline optimization on entire dataset ---")
            base_result = self.base_optimizer.optimize(
                param_space, fitness_function, constraints, **kwargs
            )
            best_params = base_result.get('best_params', {})
            best_score = base_result.get('best_score', float('-inf'))
            
        # Store best result for get_best_result method
        self.best_result = {
            'params': best_params,
            'score': best_score
        }
        
        # Include best overall result
        optimization_result['best_params'] = best_params
        optimization_result['best_score'] = best_score
        
        # Store full results
        self.results = optimization_result
        
        return optimization_result
    
    def _detect_regimes(self, data_handler, start_date, end_date):
        """
        Run regime detection on historical data.
        
        Args:
            data_handler: Data handler with loaded data
            start_date: Optional start date for analysis
            end_date: Optional end date for analysis
        """
        symbol = data_handler.get_symbols()[0]
        
        # Reset data handler to start of data
        data_handler.reset()
        
        # Reset detector
        self.regime_detector.reset()
        
        # Process each bar to detect regimes
        bar_count = 0
        
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
                
            # Skip bars outside date range
            timestamp = bar.get_timestamp()
            
            if start_date and timestamp < start_date:
                continue
            if end_date and timestamp > end_date:
                break
                
            # Detect regime
            self.regime_detector.update(bar)
            bar_count += 1
            
            if bar_count % 100 == 0:
                logger.debug(f"Processed {bar_count} bars for regime detection")
        
        logger.info(f"Processed {bar_count} bars for regime detection")
        
        # Reset data handler for optimization
        data_handler.reset()
    
    def _count_bars_in_periods(self, data_handler, symbol, periods):
        """
        Count total bars across multiple time periods.
        
        Args:
            data_handler: Data handler with loaded data
            symbol: Symbol to count bars for
            periods: List of (start, end) tuples defining time periods
            
        Returns:
            int: Total bar count
        """
        # Reset data handler
        data_handler.reset()
        
        total_bars = 0
        
        # Process each bar
        while True:
            bar = data_handler.get_next_bar(symbol)
            if bar is None:
                break
                
            # Check if bar falls in any of the periods
            timestamp = bar.get_timestamp()
            
            for start, end in periods:
                if start <= timestamp <= end:
                    total_bars += 1
                    break
        
        # Reset data handler
        data_handler.reset()
        
        return total_bars
    
    def _optimize_for_regime(self, param_space, fitness_function, constraints, 
                           regime, periods, data_handler, **kwargs):
        """
        Optimize parameters for a specific regime.
        
        Args:
            param_space: Dictionary of parameter names to possible values
            fitness_function: Function to evaluate parameter sets
            constraints: Optional list of constraint functions
            regime: MarketRegime to optimize for
            periods: List of (start, end) tuples defining time periods for this regime
            data_handler: Data handler with loaded data
            **kwargs: Additional parameters
            
        Returns:
            tuple: (best_params, best_score)
        """
        # Get date range from periods
        min_start = min(start for start, _ in periods)
        max_end = max(end for _, end in periods)
        
        # Create regime-specific fitness function
        def regime_fitness(params):
            return self._evaluate_in_regime(
                params, fitness_function, data_handler, regime, periods, **kwargs
            )
        
        # Use base optimizer on regime data
        try:
            regime_kwargs = dict(kwargs)
            regime_kwargs.update({
                'start_date': min_start,
                'end_date': max_end,
                'data_handler': data_handler
            })
            
            result = self.base_optimizer.optimize(
                param_space, regime_fitness, constraints, **regime_kwargs
            )
            
            return result.get('best_params'), result.get('best_score', float('-inf'))
        except Exception as e:
            logger.error(f"Error in regime optimization for {regime.value}: {e}")
            return None, float('-inf')
    
    def _evaluate_in_regime(self, params, fitness_function, data_handler, 
                          regime, periods, **kwargs):
        """
        Evaluate parameters on data from a specific regime.
        
        Args:
            params: Parameters to evaluate
            fitness_function: Function to evaluate parameters
            data_handler: Data handler with loaded data
            regime: MarketRegime being evaluated
            periods: List of time periods for this regime
            **kwargs: Additional parameters
            
        Returns:
            float: Fitness score
        """
        # Calculate score on regime-specific data
        try:
            scores = []
            
            # Evaluate on each period
            for start, end in periods:
                period_kwargs = dict(kwargs)
                period_kwargs.update({
                    'start_date': start,
                    'end_date': end,
                    'data_handler': data_handler
                })
                
                score = fitness_function(params, **period_kwargs)
                
                # Handle different return types
                if isinstance(score, dict):
                    # Extract specific metric if result is a dict
                    metric = kwargs.get('optimize_metric', 'score')
                    if metric in score:
                        score = score[metric]
                    else:
                        # Use first value if metric not found
                        score = next(iter(score.values()))
                
                scores.append(score)
            
            # Aggregate scores (mean, min, etc.)
            if not scores:
                return float('-inf')
                
            # Use mean score by default, could use min for robustness
            agg_method = kwargs.get('aggregate_method', 'mean')
            
            if agg_method == 'mean':
                return np.mean(scores)
            elif agg_method == 'min':
                return min(scores)
            elif agg_method == 'median':
                return np.median(scores)
            else:
                return np.mean(scores)
            
        except Exception as e:
            logger.error(f"Error evaluating parameters in regime {regime.value}: {e}")
            return float('-inf')
    
    def _get_default_params(self, regime, existing_params):
        """
        Create default parameters for a regime with no optimization data.
        
        Args:
            regime: Market regime to get parameters for
            existing_params: Parameters for other regimes
            
        Returns:
            dict: Default parameters
        """
        # If we have parameters for other regimes, use similar or average
        if not existing_params:
            return {}
            
        # Map of regime similarities
        similarities = {
            MarketRegime.UPTREND: [MarketRegime.VOLATILE, MarketRegime.SIDEWAYS, MarketRegime.DOWNTREND],
            MarketRegime.DOWNTREND: [MarketRegime.VOLATILE, MarketRegime.SIDEWAYS, MarketRegime.UPTREND],
            MarketRegime.SIDEWAYS: [MarketRegime.UPTREND, MarketRegime.DOWNTREND, MarketRegime.VOLATILE],
            MarketRegime.VOLATILE: [MarketRegime.UPTREND, MarketRegime.DOWNTREND, MarketRegime.SIDEWAYS],
            MarketRegime.UNKNOWN: [MarketRegime.SIDEWAYS, MarketRegime.UPTREND, MarketRegime.DOWNTREND, MarketRegime.VOLATILE]
        }
        
        # Try to find parameters from most similar regime
        for similar_regime in similarities.get(regime, []):
            if similar_regime in existing_params:
                logger.info(f"Using {similar_regime.value} parameters for {regime.value}")
                return existing_params[similar_regime]
        
        # If no similar regime has parameters, use average of all available parameters
        logger.info(f"Using average parameters for {regime.value}")
        avg_params = {}
        
        # Get all parameter keys
        all_keys = set()
        for params in existing_params.values():
            all_keys.update(params.keys())
            
        # Calculate average for each parameter
        for key in all_keys:
            values = [params[key] for params in existing_params.values() if key in params]
            if values:
                if all(isinstance(v, (int, float)) for v in values):
                    # Use mean for numeric parameters
                    avg_params[key] = sum(values) / len(values)
                else:
                    # Use most common value for non-numeric
                    from collections import Counter
                    counter = Counter(values)
                    avg_params[key] = counter.most_common(1)[0][0]
                
        return avg_params
