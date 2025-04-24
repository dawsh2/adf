#!/usr/bin/env python
"""
Test script to verify the fix implementation for the algorithmic trading system.
This test focuses on the position tracking, fill processing, and performance calculation.
"""
import logging
import sys
import os
import pandas as pd
import datetime

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_ma_strategy_backtest():
    """Run a simple backtest with the moving average strategy to verify the fixes."""
    logger.info("\n===== TESTING MA STRATEGY BACKTEST =====\n")
    
    # Import components
    from src.core.events.event_bus import EventBus
    from src.core.events.event_manager import EventManager
    from src.core.events.event_types import EventType
    from src.data.sources.csv_handler import CSVDataSource
    from src.data.historical_data_handler import HistoricalDataHandler
    from src.strategy.strategies.ma_crossover import MovingAverageCrossoverStrategy
    from src.execution.backtest.backtest import run_backtest
    from src.analytics.performance import PerformanceAnalytics
    
    # Create data source
    data_dir = os.path.join('data')
    data_source = CSVDataSource(data_dir=data_dir)
    
    # Create data handler
    data_handler = HistoricalDataHandler(
        data_source=data_source,
        bar_emitter=None  # Will be set by backtest
    )
    
    # Set date range for test
    start_date = pd.to_datetime("2024-03-26")
    end_date = pd.to_datetime("2024-03-28")
    
    # Load data
    data_handler.load_data(
        symbols=["SAMPLE"],
        start_date=start_date,
        end_date=end_date,
        timeframe="1m"
    )
    
    # Create strategy with small windows for frequent signals
    strategy = MovingAverageCrossoverStrategy(
        name="ma_crossover_test",
        symbols=["SAMPLE"],
        fast_window=5,  # Small window for testing
        slow_window=20
    )
    
    # Run backtest
    logger.info(f"Running backtest from {start_date} to {end_date}")
    equity_curve, trades = run_backtest(
        component=strategy,
        data_handler=data_handler,
        start_date=start_date,
        end_date=end_date
    )
    
    # Calculate and display performance metrics
    logger.info("\nCalculating performance metrics...")
    metrics = PerformanceAnalytics.calculate_metrics(equity_curve, trades)
    
    # Display key metrics
    logger.info("\nPerformance Results:")
    logger.info(f"Initial Equity: ${metrics['initial_equity']:.2f}")
    logger.info(f"Final Equity: ${metrics['final_equity']:.2f}")
    logger.info(f"Total Return: {metrics['total_return']:.2f}%")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    logger.info(f"Total Trades: {metrics['trade_count']}")
    logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
    
    # Check for success criteria
    test_passed = False
    
    # 1. Check if equity is positive and sensible
    equity_ok = metrics['final_equity'] > 0 and metrics['final_equity'] < 20000
    
    # 2. Check if trades were executed
    trades_ok = metrics['trade_count'] > 0
    
    # 3. Check if metrics are not NaN or inf
    metrics_ok = all(
        not pd.isna(v) and not np.isinf(v) 
        for k, v in metrics.items() 
        if isinstance(v, (int, float))
    )
    
    # Final evaluation
    test_passed = equity_ok and trades_ok and metrics_ok
    
    if test_passed:
        logger.info("\nTEST PASSED: Strategy executed correctly with valid results")
    else:
        logger.error("\nTEST FAILED: Issues detected in backtest execution")
        if not equity_ok:
            logger.error("- Final equity is not in a reasonable range")
        if not trades_ok:
            logger.error("- No trades were executed")
        if not metrics_ok:
            logger.error("- Performance metrics contain invalid values")
    
    return test_passed, equity_curve, trades, metrics

def test_portfolio_position_tracking():
    """Test the position tracking functionality in isolation."""
    logger.info("\n===== TESTING PORTFOLIO POSITION TRACKING =====\n")
    
    # Import the fixed portfolio manager
    from src.execution.portfolio import PortfolioManager, Position
    
    # Create a test portfolio
    portfolio = PortfolioManager(initial_cash=10000.0)
    
    # Create test position
    position = Position("TEST")
    
    # Test 1: Adding to position
    logger.info("Test 1: Adding to position")
    position.add_quantity(100, 100.0)  # Buy 100 shares at $100
    
    # Check position state
    logger.info(f"After buy: Quantity={position.quantity}, Cost Basis=${position.cost_basis:.2f}")
    test1_passed = position.quantity == 100 and position.cost_basis == 100.0
    
    # Test 2: Adding more at a different price
    logger.info("Test 2: Adding more at a different price")
    position.add_quantity(50, 110.0)  # Buy 50 more shares at $110
    
    # Check position state - cost basis should be weighted average
    expected_cost_basis = ((100 * 100.0) + (50 * 110.0)) / 150
    logger.info(f"After second buy: Quantity={position.quantity}, Cost Basis=${position.cost_basis:.2f}")
    test2_passed = position.quantity == 150 and abs(position.cost_basis - expected_cost_basis) < 0.01
    
    # Test 3: Partial reduction
    logger.info("Test 3: Partial reduction")
    position.reduce_quantity(50, 120.0)  # Sell 50 shares at $120
    
    # Check position state
    logger.info(f"After partial sell: Quantity={position.quantity}, Cost Basis=${position.cost_basis:.2f}")
    test3_passed = position.quantity == 100
    
    # Test 4: Complete liquidation
    logger.info("Test 4: Complete liquidation")
    position.reduce_quantity(100, 130.0)  # Sell all remaining shares at $130
    
    # Check position state
    logger.info(f"After full sell: Quantity={position.quantity}, Cost Basis=${position.cost_basis:.2f}")
    test4_passed = position.quantity == 0 and position.cost_basis == 0.0
    
    # Test 5: Short position
    logger.info("Test 5: Short position")
    position.add_quantity(-100, 140.0)  # Short 100 shares at $140
    
    # Check position state
    logger.info(f"After short: Quantity={position.quantity}, Cost Basis=${position.cost_basis:.2f}")
    test5_passed = position.quantity == -100 and position.cost_basis == 140.0
    
    # Test 6: Covering short
    logger.info("Test 6: Covering short")
    position.reduce_quantity(100, 130.0)  # Cover 100 shares at $130
    
    # Check position state
    logger.info(f"After cover: Quantity={position.quantity}, Cost Basis=${position.cost_basis:.2f}")
    test6_passed = position.quantity == 0 and position.cost_basis == 0.0
    
    # Overall test result
    all_tests_passed = test1_passed and test2_passed and test3_passed and test4_passed and test5_passed and test6_passed
    
    if all_tests_passed:
        logger.info("\nPOSITION TRACKING TESTS PASSED")
    else:
        logger.error("\nPOSITION TRACKING TESTS FAILED")
        if not test1_passed:
            logger.error("- Test 1 (Adding to position) failed")
        if not test2_passed:
            logger.error("- Test 2 (Adding more at different price) failed")
        if not test3_passed:
            logger.error("- Test 3 (Partial reduction) failed")
        if not test4_passed:
            logger.error("- Test 4 (Complete liquidation) failed")
        if not test5_passed:
            logger.error("- Test 5 (Short position) failed")
        if not test6_passed:
            logger.error("- Test 6 (Covering short) failed")
    
    return all_tests_passed

def run_all_tests():
    """Run all tests and report results."""
    # Import numpy for metrics validation
    import numpy as np
    
    print("\n" + "="*80)
    print("RUNNING ALGORITHMIC TRADING SYSTEM TESTS")
    print("="*80 + "\n")
    
    # Run position tracking tests
    position_tests_passed = test_portfolio_position_tracking()
    
    # Run the backtest test
    backtest_passed, equity_curve, trades, metrics = test_ma_strategy_backtest()
    
    # Get overall results
    all_tests_passed = position_tests_passed and backtest_passed
    
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    print(f"Position Tracking Tests: {'PASSED' if position_tests_passed else 'FAILED'}")
    print(f"Strategy Backtest Tests: {'PASSED' if backtest_passed else 'FAILED'}")
    print(f"Overall Test Result: {'ALL TESTS PASSED' if all_tests_passed else 'TESTS FAILED'}")
    print("="*80 + "\n")
    
    # Write detailed test report
    try:
        from src.analytics.performance import PerformanceAnalytics
        
        # Display formatted results
        metrics_table = PerformanceAnalytics.display_metrics(metrics, "Fixed Strategy Performance")
        print("\n" + metrics_table + "\n")
    except ImportError:
        # If analytics module not available, show simple report
        print("\nDetailed Performance Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                print(f"  {key}: {value}")
    
    return all_tests_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
