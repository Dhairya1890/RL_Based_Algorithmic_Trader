"""
load_studentA_features.py
--------------------------
Loads Nitesh's CLEAN, feature-engineered, per-stock data from
studentA/data/processed/. This REPLACES the yfinance data pulled
earlier. Nitesh has already computed RSI, MACD, volatility, etc.
correctly per stock, so this file builds on top of his output
instead of duplicating that work.

Explicitly refuses to load NIFTY50_all.csv, which is a known-corrupted
file mixing all 50 companies together -- see config_env.EXCLUDED_FILES.
"""

import os
import pandas as pd

import config_env as cfg


def list_available_tickers() -> list:
    """
    Returns every valid single-stock ticker available in Nitesh's
    processed folder, with the corrupted/non-stock files filtered out.
    """
    all_files = os.listdir(cfg.STUDENT_A_PROCESSED_DIR)
    tickers = [
        f.replace(".csv", "")
        for f in all_files
        if f.endswith(".csv") and f not in cfg.EXCLUDED_FILES
    ]
    return sorted(tickers)


def load_stock_features(ticker: str = cfg.DEFAULT_TICKER) -> pd.DataFrame:
    """
    Loads one stock's clean, feature-engineered data.

    Returns a DataFrame with:
      - 'Date' (datetime, sorted ascending)
      - 'Close' (kept separately for reward calculation)
      - every column in config_env.RAW_NUMERIC_COLUMNS + ENGINEERED_COLUMNS

    Raises an error rather than silently loading bad data if:
      - the requested ticker is in the excluded-files list
      - the file does not exist
      - any leakage column (Tomorrow_Close, Tomorrow_Return, Target)
        would end up in the returned feature set
    """
    filename = f"{ticker}.csv"

    if filename in cfg.EXCLUDED_FILES:
        raise ValueError(
            f"'{filename}' is a known-corrupted / non-single-stock file "
            f"(see config_env.EXCLUDED_FILES) and must never be loaded "
            f"as if it were one ticker."
        )

    filepath = os.path.join(cfg.STUDENT_A_PROCESSED_DIR, filename)
    if not os.path.exists(filepath):
        available = list_available_tickers()
        raise FileNotFoundError(
            f"No processed data found for ticker '{ticker}' at {filepath}.\n"
            f"Available tickers: {available}"
        )

    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    required_cols = set(cfg.RAW_NUMERIC_COLUMNS + cfg.ENGINEERED_COLUMNS + ["Date"])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"'{filename}' is missing expected columns: {missing}")

    # Defensive check: make sure none of Nitesh's leakage columns are
    # accidentally about to be treated as features anywhere downstream.
    leaked = set(cfg.LEAKAGE_COLUMNS_NEVER_USE_AS_FEATURES) & set(
        cfg.RAW_NUMERIC_COLUMNS + cfg.ENGINEERED_COLUMNS
    )
    if leaked:
        raise ValueError(
            f"Leakage column(s) {leaked} found inside the feature column "
            f"config -- this must never happen. Check config_env.py."
        )

    keep_cols = ["Date", "Close"] + cfg.RAW_NUMERIC_COLUMNS + cfg.ENGINEERED_COLUMNS
    # dict.fromkeys instead of set() to preserve column order, dedupe 'Close'
    keep_cols = list(dict.fromkeys(keep_cols))

    return df[keep_cols].reset_index(drop=True)


if __name__ == "__main__":
    # Quick manual check: python load_studentA_features.py
    print("Available tickers:", list_available_tickers())
    df = load_stock_features(cfg.DEFAULT_TICKER)
    print(f"\nLoaded {cfg.DEFAULT_TICKER}: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
    print("\nNull check:", df.isnull().sum().sum(), "total nulls")
