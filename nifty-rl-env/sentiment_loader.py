"""
sentiment_loader.py
--------------------
Dhairya's real sentiment pipeline (sub-problem D) isn't delivered yet.
This file provides a PLACEHOLDER so the rest of the pipeline — the
merge step and the environment — can be built and tested now, against
the exact schema his real output will use.

WHEN DHAIRYA DELIVERS sentiment_features.csv:
Just call load_real_sentiment_features(path) instead of
generate_placeholder_sentiment(dates) in build_state_dataset.py —
one line change, nothing else needs to move.
"""

import pandas as pd

import config_env as cfg


def generate_placeholder_sentiment(dates: pd.Series) -> pd.DataFrame:
    """
    TEMPORARY STAND-IN for Dhairya's sentiment pipeline.

    Returns a DataFrame with 'Date' plus all columns in
    config_env.SENTIMENT_COLUMNS, filled with neutral values (0.0).
    This means: until real sentiment data is plugged in, the agent's
    sentiment inputs carry zero information — which is intentional and
    safe (equivalent to training the price-only agent), NOT a bug.

    Do not treat any results from a run using this placeholder as
    evidence about whether sentiment helps — that comparison is only
    valid once load_real_sentiment_features() replaces this function.
    """
    placeholder = pd.DataFrame({"Date": pd.to_datetime(dates)})
    for col in cfg.SENTIMENT_COLUMNS:
        placeholder[col] = 0.0
    return placeholder


def load_real_sentiment_features(path: str) -> pd.DataFrame:
    """
    Loads Dhairya's REAL sentiment_features.csv once it's delivered.

    Expected schema: 'Date' column plus every column listed in
    config_env.SENTIMENT_COLUMNS. Raises an error if the schema doesn't
    match, rather than silently proceeding with wrong/missing columns.
    """
    df = pd.read_csv(path, parse_dates=["Date"])

    required_cols = set(cfg.SENTIMENT_COLUMNS) | {"Date"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Dhairya's sentiment file is missing expected columns: {missing}.\n"
            f"Expected schema: Date + {cfg.SENTIMENT_COLUMNS}"
        )

    return df[["Date"] + cfg.SENTIMENT_COLUMNS].sort_values("Date").reset_index(drop=True)


if __name__ == "__main__":
    # Quick manual check: python sentiment_loader.py
    dummy_dates = pd.date_range("2020-01-01", periods=5)
    df = generate_placeholder_sentiment(dummy_dates)
    print(df)
