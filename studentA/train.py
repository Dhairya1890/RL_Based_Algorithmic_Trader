"""
train.py
Generalized LightGBM training script for pooled NIFTY 50 multi-stock daily data.
Implements:
  - Strict chronological sorting & TimeSeriesSplit cross-validation
  - Early stopping on out-of-fold validation log-loss
  - Directional classification metrics (Accuracy, ROC-AUC, Log Loss)
  - High-conviction threshold filtering
  - Realistic financial backtesting (+1 day execution offset & 10 bps friction)
  - Model serialization and feature importance reporting
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

import config


def load_and_pool_processed_data() -> Tuple[pd.DataFrame, List[str]]:
    """Loads and pools all processed stock CSVs, sorted strictly by Date."""
    processed_dir = Path(config.PROCESSED_DATA_DIR)
    csv_files = sorted(list(processed_dir.glob("*.csv")))
    
    if not csv_files:
        raise FileNotFoundError(f"No processed CSV files found in {processed_dir}")

    dfs = []
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        df["Date"] = pd.to_datetime(df["Date"])
        dfs.append(df)

    pooled_df = pd.concat(dfs, axis=0, ignore_index=True)
    pooled_df.sort_values(by=["Date"], ascending=True, inplace=True)
    pooled_df.reset_index(drop=True, inplace=True)

    # Extract clean feature matrix columns
    feature_cols = [
        col for col in pooled_df.columns if col not in config.EXCLUDE_COLUMNS
    ]
    return pooled_df, feature_cols


def run_strategy_backtest(
    df: pd.DataFrame,
    probas: np.ndarray,
    cost_bps: float = 0.0010,
) -> Dict[str, float]:
    """
    Simulates a long/cash trading strategy:
      - 1-day execution offset (Signal at close t executes on close t+1)
      - Fixed 10 bps transaction fee per position turnover
    """
    df_bt = df.copy()
    # Binary signal: Long (1) if Prob > 0.50 else Cash (0)
    df_bt["Signal"] = (probas > 0.50).astype(int)
    
    # 1-day execution offset
    df_bt["Position"] = df_bt["Signal"].shift(1).fillna(0)
    
    # Strategy Return
    trade_costs = (df_bt["Position"].diff().abs().fillna(0)) * cost_bps
    df_bt["Strat_Return"] = (df_bt["Position"] * df_bt["Tomorrow_Return"]) - trade_costs

    active_days = df_bt[df_bt["Position"] == 1]
    win_rate = (
        (active_days["Strat_Return"] > 0).sum() / len(active_days)
        if len(active_days) > 0
        else 0.0
    )
    
    # Annualized CAGR & Sharpe
    total_ret = (1 + df_bt["Strat_Return"]).prod() - 1
    n_years = len(df_bt["Date"].unique()) / 252.0
    cagr = ((1 + total_ret) ** (1 / max(n_years, 0.01))) - 1
    
    daily_mean = df_bt["Strat_Return"].mean()
    daily_std = df_bt["Strat_Return"].std() + 1e-8
    sharpe = (daily_mean / daily_std) * np.sqrt(252)

    # Max Drawdown
    cum_returns = (1 + df_bt["Strat_Return"]).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_dd = drawdown.min()

    return {
        "CAGR": cagr,
        "Sharpe_Ratio": sharpe,
        "Max_Drawdown": max_dd,
        "Win_Rate": win_rate,
    }


def train_lightgbm_generalized(n_splits: int = 5):
    """Executes walk-forward cross validation and fits a generalized LightGBM model."""
    print("=======================================================")
    print("        INITIALIZING LIGHTGBM POOLED TRAINING          ")
    print("=======================================================")

    pooled_df, feature_cols = load_and_pool_processed_data()
    print(f"Loaded {len(pooled_df)} rows across 50 stocks.")
    print(f"Feature count ({len(feature_cols)}): {feature_cols}\n")

    X = pooled_df[feature_cols].values
    y = pooled_df[config.TARGET_COLUMN].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_preds = np.zeros(len(pooled_df))
    oof_probas = np.zeros(len(pooled_df))
    oof_mask = np.zeros(len(pooled_df), dtype=bool)

    # Hyperparameters tuned for low signal-to-noise financial data
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 4,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 50,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_estimators": 1000,
        "verbose": -1,
    }

    fold_losses = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        val_probas = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probas > 0.50).astype(int)

        loss = log_loss(y_val, val_probas)
        acc = accuracy_score(y_val, val_preds)
        fold_losses.append(loss)

        oof_probas[val_idx] = val_probas
        oof_preds[val_idx] = val_preds
        oof_mask[val_idx] = True

        print(f"Fold {fold + 1}/{n_splits} | Log Loss: {loss:.4f} | Accuracy: {acc * 100:.2f}% | Best Iter: {model.best_iteration_}")

    # --- Out-of-Fold Evaluation ---
    y_eval = y[oof_mask]
    eval_preds = oof_preds[oof_mask]
    eval_probas = oof_probas[oof_mask]
    eval_df = pooled_df[oof_mask].copy()

    total_acc = accuracy_score(y_eval, eval_preds)
    total_auc = roc_auc_score(y_eval, eval_probas)
    total_loss = log_loss(y_eval, eval_probas)

    print("\n=======================================================")
    print("         LIGHTGBM CROSS-VALIDATION RESULTS             ")
    print("=======================================================")
    print(f"Total Evaluated Samples: {len(y_eval)}")
    print(f"Mean Fold Log Loss:     {np.mean(fold_losses):.4f}")
    print(f"Overall Accuracy:       {total_acc:.4f} ({total_acc * 100:.2f}%)")
    print(f"ROC-AUC Score:          {total_auc:.4f}")
    print(f"Overall Log Loss:       {total_loss:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_eval, eval_preds)
    print(f"  [TN: {cm[0,0]} | FP: {cm[0,1]}]")
    print(f"  [FN: {cm[1,0]} | TP: {cm[1,1]}]")

    print("\nClassification Report:")
    print(classification_report(y_eval, eval_preds, digits=4))

    # --- High Conviction Filter ---
    conviction_mask = (eval_probas >= 0.54) | (eval_probas <= 0.46)
    filtered_samples = np.sum(conviction_mask)
    if filtered_samples > 0:
        filtered_acc = accuracy_score(y_eval[conviction_mask], eval_preds[conviction_mask])
        coverage = (filtered_samples / len(y_eval)) * 100
        print("=======================================================")
        print("        HIGH CONVICTION ONLY ACCURACY REPORT           ")
        print("=======================================================")
        print(f"Filtered Trades:  {filtered_samples} / {len(y_eval)} ({coverage:.1f}%)")
        print(f"Filtered Accuracy: {filtered_acc:.4f} ({filtered_acc * 100:.2f}%)")

    # --- Backtest Financial Performance ---
    bt_metrics = run_strategy_backtest(eval_df, eval_probas)
    print("=======================================================")
    print("         OUT-OF-FOLD FINANCIAL BACKTEST                ")
    print("=======================================================")
    print(f"CAGR:             {bt_metrics['CAGR'] * 100:.2f}%")
    print(f"Sharpe Ratio:     {bt_metrics['Sharpe_Ratio']:.2f}")
    print(f"Max Drawdown:     {bt_metrics['Max_Drawdown'] * 100:.2f}%")
    print(f"Win Rate:         {bt_metrics['Win_Rate'] * 100:.2f}%")

    # --- Retrain Final Model on 100% Chronological Data ---
    print("\nRetraining final generalized model on full historical dataset...")
    final_model = lgb.LGBMClassifier(**lgb_params)
    final_model.fit(X, y)

    # Feature Importances (Gain)
    importances = final_model.booster_.feature_importance(importance_type="gain")
    gain_series = pd.Series(importances / importances.sum(), index=feature_cols).sort_values(ascending=False)

    print("\nTop 15 Feature Importances (Gain):")
    for feat, gain in gain_series.items():
        print(f"  {feat:<22} {gain:.4f}")

    # Save Artifacts
    models_dir = Path(config.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "lightgbm_generalized_50stocks.txt"
    final_model.booster_.save_model(str(model_path))

    joblib_path = models_dir / "lightgbm_generalized_50stocks.joblib"
    joblib.dump(final_model, joblib_path)
    
    print(f"\nSaved LightGBM model to:")
    print(f"  - Booster text: {model_path}")
    print(f"  - Joblib file:  {joblib_path}")


if __name__ == "__main__":
    train_lightgbm_generalized()