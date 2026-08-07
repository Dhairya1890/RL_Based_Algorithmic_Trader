"""
feature_engineering.py

Quantitative Feature Engineering Engine for Stock Market Data.

Responsibilities:
- Compute stationary technical indicators (Returns, Volatility, SMA distances, RSI, MACD, Bollinger Bands, ATR).
- Compute market microstructure signals (Volume ratios, VWAP distance, Deliverable %).
- Compute multi-day sequence lags (1, 2, 3, 5 day lags) to provide historical temporal memory.
- Pure data transformation module (No file I/O or target creation).
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("feature_engineering")


def compute_technical_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Computes stationary technical, statistical, momentum, and lagged features on an OHLCV DataFrame.
    All calculations are backward-looking to eliminate look-ahead bias.
    """
    df_feat = df.copy()

    # Ensure chronological ordering
    if date_col in df_feat.columns:
        df_feat[date_col] = pd.to_datetime(df_feat[date_col])
        df_feat.sort_values(by=date_col, ascending=True, inplace=True)

    close = df_feat["Close"]
    high = df_feat["High"]
    low = df_feat["Low"]
    volume = df_feat["Volume"]

    # --- 1. Stationary Price Returns & Volatility ---
    df_feat["Log_Return"] = np.log(close / close.shift(1))
    df_feat["Vol_10D"] = df_feat["Log_Return"].rolling(window=10).std()
    df_feat["Vol_20D"] = df_feat["Log_Return"].rolling(window=20).std()

    # Multi-day momentum velocity (Cumulative returns over 3, 5, and 10 days)
    df_feat["Return_3D"] = (close - close.shift(3)) / (close.shift(3) + 1e-8)
    df_feat["Return_5D"] = (close - close.shift(5)) / (close.shift(5) + 1e-8)
    df_feat["Return_10D"] = (close - close.shift(10)) / (close.shift(10) + 1e-8)

    # --- 2. Normalized Trend Indicators (Distance to Moving Averages) ---
    sma_10 = close.rolling(window=10).mean()
    sma_20 = close.rolling(window=20).mean()
    sma_50 = close.rolling(window=50).mean()

    df_feat["Dist_SMA_10"] = (close - sma_10) / (sma_10 + 1e-8)
    df_feat["Dist_SMA_20"] = (close - sma_20) / (sma_20 + 1e-8)
    df_feat["Dist_SMA_50"] = (close - sma_50) / (sma_50 + 1e-8)

    # --- 3. Relative Strength Index (RSI - 14 Days) ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df_feat["RSI_14"] = 100 - (100 / (1 + rs))

    # --- 4. Moving Average Convergence Divergence (MACD) ---
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df_feat["MACD"] = ema_12 - ema_26
    df_feat["MACD_Signal"] = df_feat["MACD"].ewm(span=9, adjust=False).mean()
    df_feat["MACD_Diff"] = df_feat["MACD"] - df_feat["MACD_Signal"]

    # --- 5. Volatility & Range Indicators (Bollinger Bands & ATR) ---
    rolling_std_20 = close.rolling(window=20).std()
    bb_upper = sma_20 + (rolling_std_20 * 2)
    bb_lower = sma_20 - (rolling_std_20 * 2)
    df_feat["BB_Pband"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)

    # Average True Range (ATR) normalized by Close
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df_feat["ATR_14"] = tr.rolling(window=14).mean() / (close + 1e-8)

    # --- 6. Volume Microstructure Features ---
    vol_sma_20 = volume.rolling(window=20).mean()
    df_feat["Volume_Ratio_20"] = volume / (vol_sma_20 + 1e-8)

    if "VWAP" in df_feat.columns:
        df_feat["VWAP_Dist"] = (close - df_feat["VWAP"]) / (df_feat["VWAP"] + 1e-8)
    if "%Deliverble" in df_feat.columns:
        df_feat["Deliverable_Pct"] = pd.to_numeric(df_feat["%Deliverble"], errors="coerce")

    # --- 7. Temporal Lagged Features (1, 2, 3, 5 Trading Days Memory) ---
    lagged_target_cols = [
        "Log_Return",
        "RSI_14",
        "MACD_Diff",
        "Volume_Ratio_20",
        "Dist_SMA_20",
        "BB_Pband",
        "ATR_14",
    ]

    for col in lagged_target_cols:
        if col in df_feat.columns:
            for lag in [1, 2, 3, 5]:
                df_feat[f"{col}_lag_{lag}"] = df_feat[col].shift(lag)

    return df_feat