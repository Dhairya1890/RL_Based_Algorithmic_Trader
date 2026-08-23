"""
predict.py
Inference script for the generalized LightGBM stock baseline model.
Generates directional predictions and probability estimates for single or batch processed datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import joblib
import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("predict_lightgbm")


def predict_stock_direction(symbol: str) -> pd.DataFrame:
    """Loads the trained LightGBM model and predicts on a given symbol's processed data."""
    model_path = Path(config.MODELS_DIR) / "lightgbm_generalized_50stocks.joblib"
    data_path = Path(config.PROCESSED_DATA_DIR) / f"{symbol}.csv"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained LightGBM model not found at {model_path}. Run train_lightgbm.py first."
        )
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found for symbol: {symbol} at {data_path}")

    # Load data and model
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    # Derive feature matrix
    exclude_set = set(config.EXCLUDE_COLUMNS)
    feature_cols = [c for c in df.columns if c not in exclude_set]
    X = df[feature_cols].values

    # Run predictions
    df["Predicted_Signal"] = model.predict(X)
    df["Confidence_Prob"] = model.predict_proba(X)[:, 1]

    logger.info("Inference complete for %s. Samples predicted: %d", symbol, len(df))
    return df


def predict_all_stocks() -> pd.DataFrame:
    """Batch inference across all processed stock CSVs."""
    processed_files = sorted(Path(config.PROCESSED_DATA_DIR).glob("*.csv"))
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
    out_path = Path(config.REPORTS_DIR) / "latest_batch_predictions_lightgbm.csv"
    batch_df.to_csv(out_path, index=False)
    logger.info("Batch predictions for %d stock(s) saved to %s", len(batch_df), out_path)
    return batch_df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_symbol = Path(sys.argv[1]).stem
        try:
            results = predict_stock_direction(target_symbol)
            cols = [c for c in ["Date", "Close", "Target", "Predicted_Signal", "Confidence_Prob"] if c in results.columns]
            print(results[cols].tail(10))
        except Exception as e:
            logger.error("Inference failed: %s", str(e))
    else:
        try:
            batch_results = predict_all_stocks()
            print(batch_results)
        except Exception as e:
            logger.error("Batch inference failed: %s", str(e))