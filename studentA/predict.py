"""
predict.py

Inference script for XGBoost stock baseline model.
Generates direction predictions and probability estimates for a given processed stock dataset.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("predict")


def predict_stock_direction(symbol: str) -> pd.DataFrame:
    """
    Loads the single generalized model (trained across all pooled stocks by
    train_xgboost.py) and predicts on a given symbol's processed data.
    """
    model_path = config.MODELS_DIR / config.GENERALIZED_MODEL_NAME
    data_path = config.PROCESSED_DATA_DIR / f"{symbol}.csv"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained generalized model not found at {model_path}. Run train_xgboost.py first."
        )
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found for symbol: {symbol} at {data_path}")

    # Load data and model
    df = pd.read_csv(data_path)
    model = XGBClassifier()
    model.load_model(str(model_path))

    # Derive features using the same single source of truth as training
    exclude_set = set(config.EXCLUDE_COLUMNS)
    feature_cols = [c for c in df.columns if c not in exclude_set]
    X = df[feature_cols].values

    # Run predictions
    df["Predicted_Signal"] = model.predict(X)
    df["Confidence_Prob"] = model.predict_proba(X)[:, 1]

    logger.info("Inference complete for %s. Samples predicted: %d", symbol, len(df))
    return df


def predict_all_stocks() -> pd.DataFrame:
    """
    Batch inference: runs predict_stock_direction for every processed stock
    CSV, keeps the latest (most recent date) row per stock, and writes the
    consolidated result to reports/latest_batch_predictions.csv.
    """
    processed_files = sorted(config.PROCESSED_DATA_DIR.glob("*.csv"))
    if not processed_files:
        raise FileNotFoundError(
            f"No processed CSV files found in {config.PROCESSED_DATA_DIR}. Run preprocess.py first."
        )

    latest_rows = []
    for file_path in processed_files:
        symbol = file_path.stem
        try:
            df = predict_stock_direction(symbol)
            latest = df.sort_values(by="Date").tail(1).copy()
            latest.insert(0, "StockFile", symbol)
            latest_rows.append(latest[["StockFile", "Date", "Close", "Predicted_Signal", "Confidence_Prob"]])
        except Exception as e:
            logger.error("Skipping %s due to error: %s", symbol, str(e))

    if not latest_rows:
        logger.warning("No predictions generated.")
        return pd.DataFrame()

    batch_df = pd.concat(latest_rows, ignore_index=True)
    out_path = config.REPORTS_DIR / "latest_batch_predictions.csv"
    batch_df.to_csv(out_path, index=False)
    logger.info("Batch predictions for %d stock(s) saved to %s", len(batch_df), out_path)
    return batch_df


if __name__ == "__main__":
    # Usage:
    #   python predict.py                          -> batch predict all stocks, save report
    #   python predict.py data/raw/ASIANPAINT.csv   -> predict a single stock
    if len(sys.argv) > 1:
        target_symbol = Path(sys.argv[1]).stem
        try:
            results = predict_stock_direction(target_symbol)
            print(results[["Date", "Close", "Target", "Predicted_Signal", "Confidence_Prob"]].tail(10))
        except Exception as e:
            logger.error("Inference failed: %s", str(e))
    else:
        try:
            batch_results = predict_all_stocks()
            print(batch_results)
        except Exception as e:
            logger.error("Batch inference failed: %s", str(e))