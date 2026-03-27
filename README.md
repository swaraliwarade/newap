# 📈 ML Stock Predictor

A fully automated machine learning pipeline for stock prediction, backtesting, and analysis with a Streamlit dashboard.

## 🚀 Quick Start (One Command)

```bash
./run.sh AAPL
```

That single command will: create a virtual env, install dependencies, run the full ML pipeline, and launch the dashboard at http://localhost:8501.

---

## 📁 Project Structure

```
ml_stock_predictor/
├── run.sh                    ← One-command launcher
├── pipeline.py               ← Master pipeline orchestrator
├── config.py                 ← Central configuration
├── requirements.txt
├── src/
│   ├── data/collector.py     ← Yahoo Finance OHLCV fetcher + cleaner
│   ├── features/engineer.py  ← 40+ technical indicators + labels
│   ├── models/trainer.py     ← RF / GradientBoosting / XGBoost
│   ├── backtest/engine.py    ← Trading simulator + metrics
│   └── dashboard/app.py      ← Streamlit dashboard
└── output/                   ← Generated artefacts (auto-created)
```

---

## 🛠️ Manual Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run pipeline only
python pipeline.py --ticker AAPL

# Launch dashboard
streamlit run src/dashboard/app.py
```

---

## ⚙️ Configuration (config.py)

Edit config.py to change tickers, date range, capital, position sizing, or model hyperparameters.

---

## 📊 Features (40+)

Trend: SMA/EMA distances, Golden cross
Momentum: RSI, MACD, Stochastic, Williams %R, ROC
Volatility: Bollinger Bands, ATR, Historical Volatility
Volume: OBV, VPT, Volume ratio
Price Action: Multi-day returns, candle body/shadow ratios

Label: Next-candle close > current close (binary classification)

---

## ⚠️ Disclaimer

For educational purposes only. Not financial advice.
