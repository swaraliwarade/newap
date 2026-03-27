# config.py — Central configuration for ML Stock Predictor

TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
DEFAULT_TICKER = "AAPL"
START_DATE = "2020-01-01"
END_DATE = None  # None = today

# Feature engineering
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
SMA_PERIODS = [10, 20, 50, 200]
EMA_PERIODS = [12, 26]
ATR_PERIOD = 14
VOLUME_SMA_PERIOD = 20

# ML settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Backtest settings
INITIAL_CAPITAL = 100_000
POSITION_SIZE = 0.95       # 95% of capital per trade
TRANSACTION_COST = 0.001   # 0.1% per trade

# Paths
MODEL_SAVE_PATH = "output/best_model.joblib"
RESULTS_SAVE_PATH = "output/backtest_results.csv"
METRICS_SAVE_PATH = "output/metrics.json"
