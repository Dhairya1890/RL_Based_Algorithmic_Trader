"""
split_dataset.py
------------------
Chronological train/test split — never random, since financial time
series must preserve order (random splitting leaks future information
into training).

IMPORTANT: Nitesh's current Kaggle data ends April 2021, not 2023. The
team's original evaluation plan assumed a 2023 test window that this
data cannot provide. Until the team agrees on updated source data, this
function defaults to using the LAST FULL YEAR actually present in the
data as the test window, so training/testing code keeps working rather
than silently producing an empty test set. This is flagged loudly below
and must be raised with the team/mentor — it is a stand-in, not a fix.
"""

import pandas as pd


def split_train_test(df: pd.DataFrame, test_start: str = None, test_end: str = None):
    """
    Splits a merged feature dataset chronologically by Date.

    If test_start/test_end are not provided, auto-derives a test window
    using the last full year of dates actually present in df, and prints
    exactly what window was chosen so it's never a silent assumption.

    Returns (train_df, test_df), both with the index reset.
    """
    max_date = df["Date"].max()
    min_date = df["Date"].min()

    if test_start is None or test_end is None:
        auto_test_start = pd.Timestamp(year=max_date.year - 1, month=max_date.month, day=max_date.day)
        test_start = auto_test_start.strftime("%Y-%m-%d")
        test_end = max_date.strftime("%Y-%m-%d")
        print(
            f"NOTE: No test window specified, and the team's original 2023 "
            f"target isn't reachable (data ends {max_date.date()}). "
            f"Auto-using the last available year as the test window: "
            f"{test_start} to {test_end}. This is a temporary stand-in — "
            f"raise the real 2023 data gap with your team/mentor."
        )

    train_df = df[df["Date"] < test_start].reset_index(drop=True)
    test_df = df[(df["Date"] >= test_start) & (df["Date"] <= test_end)].reset_index(drop=True)

    if test_df.empty:
        raise ValueError(
            f"Test set is still empty after auto-detection. Requested window "
            f"{test_start} to {test_end}, but data only covers "
            f"{min_date.date()} to {max_date.date()}. Check the source data."
        )
    if train_df.empty:
        raise ValueError("Train set is empty — test window covers the entire dataset.")

    return train_df, test_df


if __name__ == "__main__":
    from build_state_dataset import build_state_dataset

    df = build_state_dataset()
    print(f"Full dataset date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    train_df, test_df = split_train_test(df)
    print(f"Train: {train_df.shape[0]} rows ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
    print(f"Test:  {test_df.shape[0]} rows ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")
