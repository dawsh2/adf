# Moving Average Crossover Strategy Optimization Report

## Summary

- Symbol: SAMPLE
- Total parameter combinations tested: 48
- Best Sharpe ratio: -1.5942
- Best strategy uses **traditional signals** (Buy when fast MA crosses above slow MA)

## Top Results

| Rank | Fast Window | Slow Window | Price | Invert Signals | Sharpe | Return (%) | P&L ($) | Max DD (%) | Trades | Win Rate (%) |
|------|------------|------------|-------|---------------|--------|------------|---------|------------|--------|--------------|
| 1 | 21 | 100 | close | No | -1.5942 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 2 | 21 | 100 | close | Yes | -1.5942 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 3 | 21 | 89 | close | No | -1.8210 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 4 | 21 | 89 | close | Yes | -1.8210 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 5 | 13 | 89 | close | No | -1.9959 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 6 | 13 | 89 | close | Yes | -1.9959 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 7 | 13 | 100 | close | No | -2.0562 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 8 | 13 | 100 | close | Yes | -2.0562 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 9 | 8 | 100 | close | No | -2.4678 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 10 | 8 | 100 | close | Yes | -2.4678 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 11 | 8 | 89 | close | No | -2.5932 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 12 | 8 | 89 | close | Yes | -2.5932 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 13 | 21 | 55 | close | No | -2.6315 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 14 | 21 | 55 | close | Yes | -2.6315 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 15 | 5 | 89 | close | No | -2.6538 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 16 | 5 | 89 | close | Yes | -2.6538 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 17 | 5 | 100 | close | No | -2.6785 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 18 | 5 | 100 | close | Yes | -2.6785 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 19 | 13 | 55 | close | No | -2.7049 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 20 | 13 | 55 | close | Yes | -2.7049 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |

## Best Parameter Set Analysis

- Fast window: 21
- Slow window: 100
- Price data: close
- Invert signals: No

## Normal vs. Inverted Signal Performance

| Signal Type | Combinations Tested | Avg Sharpe Ratio | Best Sharpe Ratio |
|------------|---------------------|------------------|-------------------|
| Normal | 24 | -2.9725 | -1.5942 |
| Inverted | 24 | -2.9725 | -1.5942 |

### Interpretation

Both traditional and inverted signals showed similar performance.

**Signal Interpretation:**
- **Traditional Signals**: Buy when the fast MA crosses above the slow MA (trend-following)
- **Inverted Signals**: Buy when the fast MA crosses below the slow MA (mean-reversion)


## Parameter Sensitivity Analysis

### Fast Window Sensitivity

| Fast Window | Avg Sharpe | Best Sharpe | Worst Sharpe | Avg Return (%) |
|------------|------------|-------------|--------------|----------------|
| 2 | -3.8890 | -3.3084 | -4.4998 | 0.0000 |
| 3 | -3.3744 | -2.8469 | -4.2165 | 0.0000 |
| 5 | -3.0175 | -2.6538 | -3.5803 | 0.0000 |
| 8 | -2.8022 | -2.4678 | -3.2428 | 0.0000 |
| 13 | -2.4598 | -1.9959 | -3.0820 | 0.0000 |
| 21 | -2.2921 | -1.5942 | -3.1218 | 0.0000 |

### Slow Window Sensitivity

| Slow Window | Avg Sharpe | Best Sharpe | Worst Sharpe | Avg Return (%) |
|------------|------------|-------------|--------------|----------------|
| 34 | -3.6239 | -3.0820 | -4.4998 | 0.0000 |
| 55 | -3.1894 | -2.6315 | -4.2672 | 0.0000 |
| 89 | -2.5365 | -1.8210 | -3.3084 | 0.0000 |
| 100 | -2.5402 | -1.5942 | -3.4804 | 0.0000 |

### Signal Inversion Sensitivity

| Invert Signals | Avg Sharpe | Best Sharpe | Worst Sharpe | Avg Return (%) |
|---------------|------------|-------------|--------------|----------------|
| No | -2.9725 | -1.5942 | -4.4998 | 0.0000 |
| Yes | -2.9725 | -1.5942 | -4.4998 | 0.0000 |

## Conclusion

**Warning: No profitable parameter combinations were found for this strategy on this dataset.**

Recommendations:
1. Try a different strategy that may be better suited to this dataset
2. Use a larger dataset with more price history
3. Consider testing on a different timeframe or market condition

The least unprofitable parameters are:
- Fast window: 21
- Slow window: 100
- Price data: close
- Invert signals: No

*Report generated on 2025-04-24 01:29:39*