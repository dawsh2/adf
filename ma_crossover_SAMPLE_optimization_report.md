# Moving Average Crossover Strategy Optimization Report

## Summary

- Symbol: SAMPLE
- Total parameter combinations tested: 24
- Best Sharpe ratio: -1.5942

## Top Results

| Rank | Fast Window | Slow Window | Price | Sharpe | Return (%) | P&L ($) | Max DD (%) | Trades | Win Rate (%) |
|------|------------|------------|-------|--------|------------|---------|------------|--------|--------------|
| 1 | 21 | 100 | close | -1.5942 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 2 | 21 | 89 | close | -1.8210 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 3 | 13 | 89 | close | -1.9959 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 4 | 13 | 100 | close | -2.0562 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 5 | 8 | 100 | close | -2.4678 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 6 | 8 | 89 | close | -2.5932 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 7 | 21 | 55 | close | -2.6315 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 8 | 5 | 89 | close | -2.6538 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 9 | 5 | 100 | close | -2.6785 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 10 | 13 | 55 | close | -2.7049 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 11 | 3 | 89 | close | -2.8469 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 12 | 8 | 55 | close | -2.9050 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 13 | 3 | 100 | close | -2.9641 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 14 | 13 | 34 | close | -3.0820 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 15 | 21 | 34 | close | -3.1218 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 16 | 5 | 55 | close | -3.1572 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 17 | 8 | 34 | close | -3.2428 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 18 | 2 | 89 | close | -3.3084 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 19 | 3 | 55 | close | -3.4703 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |
| 20 | 2 | 100 | close | -3.4804 | 0.0000 | 0.00 | 0.0000 | 0 | 0.0000 |

## Best Parameter Set Analysis

- Fast window: 21
- Slow window: 100
- Price data: close


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

*Report generated on 2025-04-24 11:09:39*