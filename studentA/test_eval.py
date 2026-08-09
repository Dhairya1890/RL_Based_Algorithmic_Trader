"""
test_eval.py

Evaluates the saved XGBoost model on out-of-time test data
and generates feature importance diagnostics.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    log_loss,
)

import config

# Use config.EXCLUDE_COLUMNS as the single source of truth so this script
# always stays in sync with what train_xgboost.py actually trained on.
EXCLUDE_COLUMNS = set(config.EXCLUDE_COLUMNS)

TARGET_COLUMN = config.TARGET_COLUMN


def load_and_split_test_data(data_dir: Path, test_ratio: float = 0.15):
    """Load pooled dataset and extract only the chronological test set."""
    csv_files = list(data_dir.glob("*.csv"))
    df_list = [pd.read_csv(f).assign(Date=lambda x: pd.to_datetime(x["Date"])) for f in csv_files]
    full_df = pd.concat(df_list, axis=0, ignore_index=True).sort_values("Date").reset_index(drop=True)

    unique_dates = np.sort(full_df["Date"].unique())
    n_dates = len(unique_dates)
    
    test_start_idx = int(n_dates * (1 - test_ratio))
    test_dates = unique_dates[test_start_idx:]
    
    test_df = full_df[full_df["Date"].isin(test_dates)].copy()
    return full_df, test_df


def main():
    # 1. Load Model
    model_path = config.MODELS_DIR / config.GENERALIZED_MODEL_NAME
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found at {model_path}. Run train_xgboost.py first.")

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    print(f"Loaded model from: {model_path}")

    # 2. Get Test Data & Feature Columns
    full_df, test_df = load_and_split_test_data(config.PROCESSED_DATA_DIR)
    feature_cols = [c for c in full_df.columns if c not in EXCLUDE_COLUMNS]

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COLUMN]

    # 3. Predict Probabilities & Labels
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.50).astype(int)

    # 4. Metrics Report
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    loss = log_loss(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    print("\n" + "=" * 55)
    print("           MODEL TEST EVALUATION RESULTS           ")
    print("=" * 55)
    print(f"Total Test Samples: {len(y_test)}")
    print(f"Accuracy:           {acc:.4f} ({acc*100:.2f}%)")
    print(f"ROC-AUC Score:      {auc:.4f}")
    print(f"Log Loss:           {loss:.4f}\n")

    print("Confusion Matrix:")
    print(f"  [TN: {cm[0][0]} | FP: {cm[0][1]}]")
    print(f"  [FN: {cm[1][0]} | TP: {cm[1][1]}]\n")

    print("Detailed Classification Report:")
    print(classification_report(y_test, preds, digits=4))
    print("=" * 55)

    # 5. Top 20 Feature Importances
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 20 Feature Importances (Gain):")
    for feat, imp in importances.head(20).items():
        print(f"  {feat:<25} {imp:.4f}")
    # Add this snippet at the bottom of main() in test_eval.py:

    high_conviction_mask = (probs >= 0.54) | (probs <= 0.46)
    filtered_y = y_test[high_conviction_mask]
    filtered_preds = preds[high_conviction_mask]
    
    if len(filtered_y) > 0:
        filtered_acc = accuracy_score(filtered_y, filtered_preds)
        print("\n" + "=" * 55)
        print("      HIGH CONVICTION ONLY ACCURACY REPORT         ")
        print("=" * 55)
        print(f"Total Filtered Trades: {len(filtered_y)} / {len(y_test)} ({len(filtered_y)/len(y_test)*100:.1f}%)")
        print(f"Filtered Accuracy:     {filtered_acc:.4f} ({filtered_acc*100:.2f}%)")
        print("=" * 55)


if __name__ == "__main__":
    main()