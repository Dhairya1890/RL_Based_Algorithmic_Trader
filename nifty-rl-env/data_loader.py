"""
data_loader.py
--------------
Handles downloading, cleaning, and splitting NIFTY 50 index data.
Kept separate from the environment class so the environment stays
data-source-agnostic — you can swap in the Kaggle CSV later without
touching nifty_trading_env.py at all.
"""

import os
import pandas as pd
import yfinance as yf


def download_nifty_data(
    start: str = "2015-01-01",
    end: str = "2023-12-31",
    save_path: str = "data/nifty50_raw.csv",
) -> pd.DataFrame:
    """
    Downloads NIFTY 50 index (^NSEI) OHLCV data via yfinance and saves it
    to disk. Raises an error instead of returning fake data if the
    download fails.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = yf.download("^NSEI", start=start, end=end)

    if df.empty:
        raise ValueError(
            "Download failed — got an empty dataframe. Check your internet "
            "connection or try again in a few minutes (Yahoo Finance rate limits)."
        )

    # yfinance sometimes returns MultiIndex columns (e.g. when multiple
    # tickers are requested) — flatten just in case.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.to_csv(save_path)
    return df


def load_nifty_data(path: str = "data/nifty50_raw.csv") -> pd.DataFrame:
    """Loads previously saved NIFTY data from disk."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def train_test_split_by_date(
    df: pd.DataFrame,
    test_start: str = "2023-01-01",
    test_end: str = "2023-12-31",
):
    """
    Splits data chronologically — NEVER randomly — because financial time
    series must preserve order. Random splitting here would leak future
    information into training (lookahead bias).

    Returns (train_df, test_df).
    """
    train_df = df[df.index < test_start].copy()
    test_df = df[(df.index >= test_start) & (df.index <= test_end)].copy()
    return train_df, test_df


if __name__ == "__main__":
    # Quick manual run: python data_loader.py
    raw = download_nifty_data()
    train_df, test_df = train_test_split_by_date(raw)

    print(f"Raw shape:   {raw.shape}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    train_df.to_csv("data/train.csv")
    test_df.to_csv("data/test.csv")
    print("Saved train.csv and test.csv to data/")