"""
test_eval.py
Independent out-of-time test evaluation harness for the LightGBM model.
Loads the trained LightGBM model, evaluates on the final 15% of chronological dates,
and computes ML diagnostics, feature importance (Gain), and high-conviction metrics.
"""

from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

import config


def load_and_split_test_data(test_ratio: float = 0.15) -> Tuple[pd.DataFrame, List[str]]:
    """Loads all processed stock CSVs and extracts the final 15% out-of-time date split."""
    processed_dir = Path(config.PROCESSED_DATA_DIR)
    csv_files = sorted(list(processed_dir.glob("*.csv")))
    
    if not csv_files:
        raise FileNotFoundError(f"No processed files found in {processed_dir}")

    dfs = []
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        df["Date"] = pd.to_datetime(df["Date"])
        dfs.append(df)

    full_df = pd.concat(dfs, axis=0, ignore_index=True)
    full_df.sort_values(by=["Date"], ascending=True, inplace=True)
    full_df.reset_index(drop=True, inplace=True)

    # Chronological date thresholding
    unique_dates = np.sort(full_df["Date"].unique())
    n_dates = len(unique_dates)
    test_start_idx = int(n_dates * (1 - test_ratio))
    test_dates = unique_dates[test_start_idx:]

    test_df = full_df[full_df["Date"].isin(test_dates)].copy()
    test_df.sort_values(by=["Date"], ascending=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    feature_cols = [
        col for col in test_df.columns if col not in config.EXCLUDE_COLUMNS
    ]

    return test_df, feature_cols


def evaluate_lightgbm_test():
    model_path = Path(config.MODELS_DIR) / "lightgbm_generalized_50stocks.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train_lightgbm.py first."
        )

    model = joblib.load(model_path)
    print(f"Loaded model from: {model_path}\n")

    test_df, feature_cols = load_and_split_test_data(test_ratio=0.15)
    X_test = test_df[feature_cols].values
    y_test = test_df[config.TARGET_COLUMN].values

    # Generate probabilities and predictions
    test_probas = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probas > 0.50).astype(int)

    # Core Metrics
    acc = accuracy_score(y_test, test_preds)
    auc = roc_auc_score(y_test, test_probas)
    loss = log_loss(y_test, test_probas)
    cm = confusion_matrix(y_test, test_preds)

    print("=======================================================")
    print("         LIGHTGBM TEST EVALUATION RESULTS             ")
    print("=======================================================")
    print(f"Total Test Samples: {len(y_test)}")
    print(f"Accuracy:           {acc:.4f} ({acc * 100:.2f}%)")
    print(f"ROC-AUC Score:      {auc:.4f}")
    print(f"Log Loss:           {loss:.4f}\n")

    print("Confusion Matrix:")
    print(f"  [TN: {cm[0, 0]} | FP: {cm[0, 1]}]")
    print(f"  [FN: {cm[1, 0]} | TP: {cm[1, 1]}]\n")

    print("Detailed Classification Report:")
    print(classification_report(y_test, test_preds, digits=4))

    # Feature Importance (Gain)
    print("=======================================================")
    print("Top 20 Feature Importances (Gain):")
    importances = model.booster_.feature_importance(importance_type="gain")
    gain_series = pd.Series(importances / importances.sum(), index=feature_cols).sort_values(ascending=False)
    
    for feat, gain in gain_series.head(20).items():
        print(f"  {feat:<24} {gain:.4f}")

    # High Conviction Report
    print("\n=======================================================")
    print("      HIGH CONVICTION ONLY ACCURACY REPORT             ")
    print("=======================================================")
    conviction_mask = (test_probas >= 0.54) | (test_probas <= 0.46)
    filtered_count = np.sum(conviction_mask)
    
    if filtered_count > 0:
        filtered_acc = accuracy_score(y_test[conviction_mask], test_preds[conviction_mask])
        coverage = (filtered_count / len(y_test)) * 100
        print(f"Total Filtered Trades: {filtered_count} / {len(y_test)} ({coverage:.1f}%)")
        print(f"Filtered Accuracy:     {filtered_acc:.4f} ({filtered_acc * 100:.2f}%)")
    else:
        print("No predictions exceeded the confidence bounds (0.46 - 0.54).")
    print("=======================================================")


if __name__ == "__main__":
    evaluate_lightgbm_test()