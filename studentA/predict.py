"""
predict.py

Inference script to predict tomorrow's stock price direction (HIGHER / LOWER)
for ALL raw CSV files inside `data/raw/` (or a single file passed via CLI).
Dynamically handles updated lag features from feature_engineering.py.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import xgboost as xgb

from feature_engineering import compute_technical_features
from preprocess import clean_dataframe
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("predict")

# Columns to strictly exclude from prediction features
EXCLUDE_COLUMNS = {
    "Date", "Symbol", "Series", "Prev Close", "Open", "High", "Low", "Last", 
    "Close", "VWAP", "Volume", "Turnover", "Trades", "Deliverable Volume", 
    "%Deliverble", "Tomorrow_Close", "Tomorrow_Return", "Target"
}


def load_model() -> xgb.XGBClassifier:
    """Load trained generalized XGBoost model."""
    model_path = config.MODEL_DIR / "xgboost_generalized_50stocks.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found at {model_path}. Please run train_xgboost.py first."
        )
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Dynamically identify technical and lagged feature columns."""
    return [col for col in df.columns if col not in EXCLUDE_COLUMNS]


def predict_single_file(model: xgb.XGBClassifier, file_path: Path) -> dict | None:
    """Run preprocessing, feature engineering, and inference on a single raw stock CSV."""
    try:
        df_raw = pd.read_csv(file_path)
        if df_raw.empty:
            return None

        # Clean & engineer technical + lagged features
        df_clean, _ = clean_dataframe(df_raw)
        df_feat = compute_technical_features(df_clean)
        
        feature_cols = get_feature_columns(df_feat)
        df_feat.dropna(subset=feature_cols, inplace=True)

        if df_feat.empty:
            logger.warning("Insufficient rows after computing indicators for %s", file_path.name)
            return None

        # Get latest available trading day
        latest_row = df_feat.iloc[[-1]]
        latest_date = pd.to_datetime(latest_row["Date"].values[0]).strftime("%Y-%m-%d")
        latest_close = latest_row["Close"].values[0]

        X_latest = latest_row[feature_cols]

        # Predict probability
        prob_up = model.predict_proba(X_latest)[0, 1]
        predicted_label = 1 if prob_up > 0.50 else 0
        direction_str = "UP" if predicted_label == 1 else "DOWN"

        return {
            "Symbol": file_path.stem,
            "Latest Date": latest_date,
            "Close Price": round(latest_close, 2),
            "Prob (UP %)": round(prob_up * 100, 2),
            "Prediction": direction_str,
        }

    except Exception as e:
        logger.error("Failed to generate prediction for %s: %s", file_path.name, str(e))
        return None


def predict_all_raw_stocks(raw_dir: Path):
    """Loop through all CSV files in raw_dir and summarize predictions."""
    model = load_model()

    ignored_files = {"stock_metadata.csv", "metadata.csv"}
    raw_files = sorted([
        f for f in raw_dir.glob("*.csv")
        if f.name not in ignored_files and f.stat().st_size > 0
    ])

    if not raw_files:
        logger.error("No valid CSV files found in directory: %s", raw_dir)
        return

    logger.info("Found %d stock CSV(s) in %s. Running inference...", len(raw_files), raw_dir)

    results = []
    for raw_file in raw_files:
        res = predict_single_file(model, raw_file)
        if res:
            results.append(res)

    if not results:
        logger.error("No valid predictions were generated.")
        return

    # Display Batch Predictions Table
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("         MULTI-STOCK TOMORROW DIRECTIONAL PREDICTIONS         ")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70)

    # Save summary report
    output_report_path = config.REPORT_DIR / "latest_batch_predictions.csv"
    results_df.to_csv(output_report_path, index=False)
    logger.info("Saved batch predictions report to: %s", output_report_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_path = Path(sys.argv[1])
        if target_path.is_dir():
            predict_all_raw_stocks(target_path)
        else:
            model = load_model()
            res = predict_single_file(model, target_path)
            if res:
                print("\nSingle Stock Prediction:")
                print(pd.DataFrame([res]).to_string(index=False))
    else:
        # Default behavior: Predict for all files in config.RAW_DATA_DIR
        predict_all_raw_stocks(config.RAW_DATA_DIR)