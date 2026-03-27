# pipeline.py — Master pipeline: run everything end-to-end

import logging
import json
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.data.collector import get_stock_data
from src.features.engineer import engineer_features, get_feature_columns
from src.models.trainer import (train_all_models, select_best_model,
                                 get_feature_importance, save_model, save_metrics)
from src.backtest.engine import run_backtest, buy_and_hold, calculate_metrics
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pipeline.log")
    ]
)
logger = logging.getLogger(__name__)


def run_pipeline(ticker: str = None) -> dict:
    ticker = ticker or config.DEFAULT_TICKER
    os.makedirs("output", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    logger.info("=" * 60)
    logger.info(f"ML STOCK PREDICTION PIPELINE — {ticker}")
    logger.info("=" * 60)

    # 1. Collect data
    logger.info("STEP 1: Collecting OHLCV data...")
    df = get_stock_data(ticker)

    # 2. Feature engineering
    logger.info("STEP 2: Engineering features...")
    df = engineer_features(df)
    feat_cols = get_feature_columns(df)
    logger.info(f"  -> {len(feat_cols)} features generated")

    # 3. Train models
    logger.info("STEP 3: Training ML models...")
    results, X_train, X_test, y_train, y_test, dates_test = train_all_models(df, feat_cols)

    # 4. Select best model
    logger.info("STEP 4: Selecting best model...")
    best_name, best_result = select_best_model(results)
    save_model(best_result['model'])
    save_metrics(results)

    # Feature importance
    fi_df = get_feature_importance(best_result['model'], feat_cols)
    if not fi_df.empty:
        fi_df.to_csv("output/feature_importance.csv", index=False)

    # 5. Backtest
    logger.info("STEP 5: Running backtest...")
    portfolio_df, trades_df = run_backtest(df, best_result['y_pred'], dates_test)
    benchmark = buy_and_hold(df, dates_test)

    # 6. Performance metrics
    logger.info("STEP 6: Calculating performance metrics...")
    metrics = calculate_metrics(portfolio_df, trades_df, benchmark)
    metrics['ticker'] = ticker
    metrics['best_model'] = best_name
    metrics['model_accuracy'] = best_result['metrics']['accuracy']
    metrics['model_roc_auc'] = best_result['metrics']['roc_auc']

    # Save results
    portfolio_df.to_csv(config.RESULTS_SAVE_PATH)
    if not trades_df.empty:
        trades_df.to_csv("output/trades.csv", index=False)
    benchmark.to_csv("output/benchmark.csv")

    with open("output/final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE - SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Ticker:         {ticker}")
    logger.info(f"  Best Model:     {best_name}")
    logger.info(f"  Model Accuracy: {metrics['model_accuracy']:.1%}")
    logger.info(f"  Model AUC:      {metrics['model_roc_auc']:.3f}")
    logger.info(f"  Total Return:   {metrics['total_return_pct']:.1f}%")
    logger.info(f"  B&H Return:     {metrics['benchmark_return_pct']:.1f}%")
    logger.info(f"  Alpha:          {metrics['alpha_pct']:.1f}%")
    logger.info(f"  Sharpe Ratio:   {metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown:   {metrics['max_drawdown_pct']:.1f}%")
    logger.info(f"  Win Rate:       {metrics['win_rate_pct']:.1f}%")
    logger.info(f"  Total Trades:   {metrics['n_trades']}")
    logger.info(f"  Final Capital:  ${metrics['final_capital_usd']:,.2f}")
    logger.info("=" * 60)

    return {
        "df":           df,
        "feat_cols":    feat_cols,
        "results":      results,
        "best_name":    best_name,
        "best_result":  best_result,
        "portfolio_df": portfolio_df,
        "trades_df":    trades_df,
        "benchmark":    benchmark,
        "metrics":      metrics,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ML Stock Prediction Pipeline")
    parser.add_argument("--ticker", type=str, default=config.DEFAULT_TICKER,
                        help="Stock ticker symbol (default: AAPL)")
    args = parser.parse_args()
    run_pipeline(args.ticker)
