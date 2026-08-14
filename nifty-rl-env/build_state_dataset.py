"""
build_state_dataset.py
------------------------
This is the "Merge on (date, ticker)" step from the team DFD — owned
by me (sub-problem B). Combines Nitesh's clean price/technical features
with Dhairya's sentiment features (placeholder for now) into one final
dataset the environment consumes.
"""

import pandas as pd

import config_env as cfg
from load_studentA_features import load_stock_features
from sentiment_loader import generate_placeholder_sentiment


def build_state_dataset(ticker: str = cfg.DEFAULT_TICKER, use_real_sentiment_path: str = None) -> pd.DataFrame:
    """
    Builds the final merged dataset for one ticker: Nitesh's features
    left-joined with sentiment features on Date, missing sentiment
    filled with 0 (matches the team's documented merge rule).

    Parameters
    ----------
    ticker : str
        Which stock to build the dataset for.
    use_real_sentiment_path : str or None
        If None (default), uses the placeholder sentiment generator.
        Once Dhairya delivers his file, pass its path here to switch to
        real sentiment data with no other code changes.

    Returns
    -------
    pd.DataFrame with columns:
        Date, Close, <RAW_NUMERIC_COLUMNS>, <ENGINEERED_COLUMNS>, <SENTIMENT_COLUMNS>
    """
    price_df = load_stock_features(ticker)

    if use_real_sentiment_path is None:
        sentiment_df = generate_placeholder_sentiment(price_df["Date"])
    else:
        from sentiment_loader import load_real_sentiment_features
        sentiment_df = load_real_sentiment_features(use_real_sentiment_path)

    merged = price_df.merge(sentiment_df, on="Date", how="left")
    merged[cfg.SENTIMENT_COLUMNS] = merged[cfg.SENTIMENT_COLUMNS].fillna(0.0)

    # Final safety check: confirm every feature column the environment
    # expects is actually present before handing this off.
    missing = set(cfg.FEATURE_COLUMNS) - set(merged.columns)
    if missing:
        raise ValueError(f"Merged dataset is missing expected feature columns: {missing}")

    # Final safety check: confirm no leakage column snuck in.
    leaked = set(cfg.LEAKAGE_COLUMNS_NEVER_USE_AS_FEATURES) & set(merged.columns)
    if leaked:
        raise ValueError(f"Leakage column(s) {leaked} found in merged dataset — must not happen.")

    return merged.reset_index(drop=True)


if __name__ == "__main__":
    # Quick manual check: python build_state_dataset.py
    df = build_state_dataset()
    print(f"Merged dataset shape: {df.shape}")
    print(f"Feature columns ({len(cfg.FEATURE_COLUMNS)}): {cfg.FEATURE_COLUMNS}")
    print(df.head())
