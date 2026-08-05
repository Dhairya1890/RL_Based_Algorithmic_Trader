"""
train_xgboost.py

Generalized XGBoost Binary Classification Model for Multi-Stock Trading Signals.
Automatically resolves all dynamic features generated in feature_engineering.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, log_loss

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_xgboost")

# Columns to strictly exclude from prediction features
EXCLUDE_COLUMNS = {
    "Date", "Symbol", "Series", "Prev Close", "Open", "High", "Low", "Last", 
    "Close", "VWAP", "Volume", "Turnover", "Trades", "Deliverable Volume", 
    "%Deliverble", "Tomorrow_Close", "Tomorrow_Return", "Target"
}

TARGET_COLUMN = "Target"


def load_and_combine_processed_dataset(data_dir: Path) -> pd.DataFrame:
    """Pool all processed stock CSVs into a single combined DataFrame."""
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No processed CSV files found in {data_dir}")
    
    logger.info("Loading and pooling %d stock CSV files...", len(csv_files))
    df_list = []
    
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df_list.append(df)
        
    full_df = pd.concat(df_list, axis=0, ignore_index=True)
    full_df.sort_values(by="Date", ascending=True, inplace=True)
    full_df.reset_index(drop=True, inplace=True)
    
    logger.info("Combined Dataset: %d total rows across %d stocks.", len(full_df), len(csv_files))
    return full_df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Dynamically identify technical and lagged feature columns."""
    return [col for col in df.columns if col not in EXCLUDE_COLUMNS]


def chronological_train_val_test_split(
    df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15
):
    """
    Strict Chronological Date Split across ALL stocks simultaneously.
    Prevents time-travel data leakage.
    """
    unique_dates = np.sort(df["Date"].unique())
    n_dates = len(unique_dates)
    
    train_end_idx = int(n_dates * train_ratio)
    val_end_idx = int(n_dates * (train_ratio + val_ratio))
    
    train_dates = unique_dates[:train_end_idx]
    val_dates = unique_dates[train_end_idx:val_end_idx]
    test_dates = unique_dates[val_end_idx:]
    
    train_df = df[df["Date"].isin(train_dates)].copy()
    val_df = df[df["Date"].isin(val_dates)].copy()
    test_df = df[df["Date"].isin(test_dates)].copy()
    
    def fmt_date(dt) -> str:
        return pd.to_datetime(dt).strftime("%Y-%m-%d")

    logger.info("Time Split Boundaries:")
    logger.info("  Train: %s to %s (%d rows)", fmt_date(train_dates[0]), fmt_date(train_dates[-1]), len(train_df))
    logger.info("  Val:   %s to %s (%d rows)", fmt_date(val_dates[0]), fmt_date(val_dates[-1]), len(val_df))
    logger.info("  Test:  %s to %s (%d rows)", fmt_date(test_dates[0]), fmt_date(test_dates[-1]), len(test_df))
    
    return train_df, val_df, test_df


def build_and_train_xgboost(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]):
    """Train XGBoost with early stopping on validation log-loss."""
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COLUMN]
    X_val, y_val = val_df[feature_cols], val_df[TARGET_COLUMN]
    
    params = {
        "objective": config.OBJECTIVE,
        "eval_metric": config.EVAL_METRIC,
        "max_depth": 3,
        "learning_rate": 0.015,
        "n_estimators": 1000,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "random_state": config.RANDOM_STATE,
        "n_jobs": -1
    }
    
    model = xgb.XGBClassifier(**params)
    
    logger.info("Starting XGBoost training with %d features...", len(feature_cols))
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )
    
    return model


def evaluate_model(model: xgb.XGBClassifier, test_df: pd.DataFrame, feature_cols: list[str]):
    """Evaluate performance on out-of-time unseen test dataset."""
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COLUMN]
    
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.50).astype(int)
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    loss = log_loss(y_test, probs)
    
    print("\n" + "=" * 50)
    print("       OUT-OF-TIME TEST EVALUATION REPORT       ")
    print("=" * 50)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test ROC-AUC:  {auc:.4f}")
    print(f"Test LogLoss:  {loss:.4f}\n")
    print(classification_report(y_test, preds, digits=4))
    print("=" * 50)


if __name__ == "__main__":
    # 1. Load data
    df_full = load_and_combine_processed_dataset(config.PROCESSED_DATA_DIR)
    
    # 2. Extract Feature Columns Dynamically
    feature_columns = get_feature_columns(df_full)
    logger.info("Detected %d features for training.", len(feature_columns))

    # 3. Chronological Split
    train_df, val_df, test_df = chronological_train_val_test_split(df_full)
    
    # 4. Fit Generalized Model
    model = build_and_train_xgboost(train_df, val_df, feature_columns)
    
    # 5. Evaluate on Test Set
    evaluate_model(model, test_df, feature_columns)
    
    # 6. Save Model
    model_save_path = config.MODEL_DIR / "xgboost_generalized_50stocks.json"
    model.save_model(model_save_path)
    logger.info("Model saved successfully at %s", model_save_path)