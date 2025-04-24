#!/usr/bin/env python
"""
Run Complete Regime Filter Analysis

This script runs both the regime filter backtest comparison and optimization.
"""
import os
import logging
import argparse
import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_complete_analysis(data_path, output_dir=None, initial_cash=10000.0, skip_optimization=False):
    """
    Run both backtest comparison and optimization.
    
    Args:
        data_path: Path to price data CSV file
        output_dir: Directory for saving results
        initial_cash: Initial portfolio cash
        skip_optimization: Whether to skip the optimization step
    """
    # Create output directory if needed
    if output_dir is None:
        # Create timestamp-based directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(data_path) if os.path.dirname(data_path) else "."
        output_dir = os.path.join(base_dir, f"regime_analysis_{timestamp}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Extract symbol from filename for better organization
    filename = os.path.basename(data_path)
    symbol = filename.split('_')[0]  # Assumes format SYMBOL_timeframe.csv
    
    # Create subdirectory for this symbol
    symbol_dir = os.path.join(output_dir, symbol)
    if not os.path.exists(symbol_dir):
        os.makedirs(symbol_dir)
    
    # Step 1: Run regime filter backtest comparison
    logger.info(f"===== STEP 1: RUNNING REGIME FILTER BACKTEST COMPARISON FOR {symbol} =====")
    
    try:
        from regime_filter_test import run_regime_filter_test
        backtest_results = run_regime_filter_test(
            data_path=data_path,
            output_dir=symbol_dir,
            initial_cash=initial_cash
        )
        logger.info("Regime filter backtest comparison completed successfully.")
    except Exception as e:
        logger.error(f"Error running regime filter backtest: {e}", exc_info=True)
        backtest_results = {"error": str(e)}
    
    # Step 2: Run optimization if not skipped
    if not skip_optimization:
        logger.info(f"===== STEP 2: RUNNING REGIME FILTER OPTIMIZATION FOR {symbol} =====")
        
        try:
            from regime_filter_optimizer import optimize_regime_filtered_strategy
            optimization_results = optimize_regime_filtered_strategy(
                data_path=data_path,
                output_dir=symbol_dir,
                initial_cash=initial_cash
            )
            logger.info("Regime filter optimization completed successfully.")
        except Exception as e:
            logger.error(f"Error running regime filter optimization: {e}", exc_info=True)
            optimization_results = {"error": str(e)}
    else:
        logger.info("Optimization step skipped as requested.")
        optimization_results = {"skipped": True}
    
    # Generate overall summary
    logger.info(f"===== ANALYSIS COMPLETE FOR {symbol} =====")
    logger.info(f"Results saved to: {symbol_dir}")
    
    # Create a simple summary file
    summary_path = os.path.join(symbol_dir, "analysis_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Regime Filter Analysis Summary for {symbol}\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Comparison Results:\n")
        if "error" in backtest_results:
            f.write(f"  Error: {backtest_results['error']}\n")
        else:
            # Extract key metrics from comparison
            ma_metrics = backtest_results.get('ma_metrics', {})
            regime_metrics = backtest_results.get('regime_metrics', {})
            
            f.write(f"  Standard MA Return: {ma_metrics.get('total_return', 'N/A')}%\n")
            f.write(f"  Regime-Filtered Return: {regime_metrics.get('total_return', 'N/A')}%\n")
            f.write(f"  Improvement: {regime_metrics.get('total_return', 0) - ma_metrics.get('total_return', 0):.2f}%\n\n")
        
        if not skip_optimization:
            f.write("Optimization Results:\n")
            if "error" in optimization_results:
                f.write(f"  Error: {optimization_results['error']}\n")
            else:
                # Extract key optimization results
                best_params = optimization_results.get('best_params', {})
                final_metrics = optimization_results.get('final_metrics', {})
                
                f.write(f"  Best Fast Window: {best_params.get('fast_window', 'N/A')}\n")
                f.write(f"  Best Slow Window: {best_params.get('slow_window', 'N/A')}\n")
                f.write(f"  Optimized Return: {final_metrics.get('total_return', 'N/A')}%\n")
                f.write(f"  Optimized Sharpe: {final_metrics.get('sharpe_ratio', 'N/A')}\n")
        else:
            f.write("Optimization: Skipped\n")
    
    logger.info(f"Summary saved to: {summary_path}")
    
    return {
        "symbol": symbol,
        "output_dir": symbol_dir,
        "backtest_results": backtest_results,
        "optimization_results": optimization_results if not skip_optimization else {"skipped": True}
    }

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Run Complete Regime Filter Analysis')
    parser.add_argument('--data', type=str, required=True, help='Path to price data CSV file')
    parser.add_argument('--output', type=str, default=None, help='Directory to save results')
    parser.add_argument('--cash', type=float, default=10000.0, help='Initial portfolio cash')
    parser.add_argument('--skip-optimization', action='store_true', help='Skip the optimization step')
    
    args = parser.parse_args()
    
    run_complete_analysis(
        data_path=args.data,
        output_dir=args.output,
        initial_cash=args.cash,
        skip_optimization=args.skip_optimization
    )

if __name__ == "__main__":
    main()
