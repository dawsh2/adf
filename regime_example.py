from src.core.config.config import Config
from src.execution.backtest.backtest import BacktestRunner
from src.data.sources.csv_handler import CSVDataSource
from src.data.historical_data_handler import HistoricalDataHandler
from src.strategy.strategies.mean_reversion import MeanReversionStrategy
from src.models.filters.regime.simple import SimpleRegimeFilteredStrategy

# Set the symbol to test - change this single variable to test different symbols
SYMBOL = "SAMPLE"
TIMEFRAME = "1m"  # You can also adjust the timeframe here

# Load configuration
config = Config()
config.load_file('src/core/config/backtest.yaml')

# Get strategy parameters
strategy_params = config.get_section('strategies').get_section('mean_reversion').as_dict()
regime_params = config.get_section('filters').get_section('regime').as_dict()

# Set up data - uses the symbol variable
data_source = CSVDataSource(
    data_dir='./data', 
    filename_pattern=f'{SYMBOL}_{TIMEFRAME}.csv'
)
data_handler = HistoricalDataHandler(data_source=data_source, bar_emitter=None)
data_handler.load_data([SYMBOL])

# Create strategies - uses the symbol variable
base_strategy = MeanReversionStrategy(symbols=[SYMBOL], **strategy_params)
regime_strategy = SimpleRegimeFilteredStrategy(
    base_strategy=MeanReversionStrategy(symbols=[SYMBOL], **strategy_params),
    **regime_params
)

# Create and set up runner
runner = BacktestRunner(config=config.get_section('backtest').as_dict())
runner.setup()

# Run comparison
results = runner.compare_strategies(
    [base_strategy, regime_strategy],
    data_handler,
    ['Base Strategy', 'Regime Strategy']
)

# Print results
print(f"\n=== BACKTEST RESULTS FOR {SYMBOL} ===")
print(f"Base return: {results[0]['return']:.2f}%")
print(f"Regime return: {results[1]['return']:.2f}%")
print(f"Improvement: {results[1]['return'] - results[0]['return']:.2f}%")

# Print additional statistics if available
if 'trade_stats' in results[0]:
    print(f"\nBase Strategy Trades: {len(results[0]['trades'])}")
    print(f"Base Win Rate: {results[0]['trade_stats'].get('win_rate', 0):.2f}%")

if 'trade_stats' in results[1]:
    print(f"Regime Strategy Trades: {len(results[1]['trades'])}")
    print(f"Regime Win Rate: {results[1]['trade_stats'].get('win_rate', 0):.2f}%")

if 'regime_stats' in results[1] and results[1]['regime_stats']:
    rs = results[1]['regime_stats']
    passed = rs.get('passed_signals', 0)
    filtered = rs.get('filtered_signals', 0)
    total = passed + filtered
    filter_rate = filtered / total * 100 if total > 0 else 0
    print(f"\nRegime Filter Stats:")
    print(f"- Signals Passed: {passed}")
    print(f"- Signals Filtered: {filtered}")
    print(f"- Filter Rate: {filter_rate:.2f}%")
