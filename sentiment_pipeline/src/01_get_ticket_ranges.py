"""
Step 1 — Scan the NIFTY 50 dataset folder and extract the exact date range
for every ticker. Saves ticker_ranges.csv to data/processed/.

Run this first before anything else.
Usage: python src/01_get_ticker_ranges.py
"""

import pandas as pd
from pathlib import Path
from config import NIFTY50_DIR, TICKER_RANGES_PATH, PROCESSED_DIR

def get_ticker_ranges(nifty50_dir: Path) -> pd.DataFrame:
    """
    Scan every CSV in the nifty50 folder and extract min/max date per ticker.
    Returns a DataFrame with columns: ticker, start_date, end_date, trading_days
    """
    records = []
    csv_files = sorted(nifty50_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {nifty50_dir}\n"
            f"Make sure your Kaggle NIFTY 50 CSVs are in: {nifty50_dir}"
        )

    print(f"Found {len(csv_files)} ticker files in {nifty50_dir}\n")

    for fpath in csv_files:
        ticker = fpath.stem.upper() # Gets the filename without its extention

        try:
            df = pd.read_csv(fpath, parse_dates=["Date"])

            # Handle case where date column might be named differently
            if "Date" not in df.columns:
                date_col = [c for c in df.columns if "date" in c.lower()]
                if not date_col:
                    print(f"  [SKIP] {ticker} — no Date column found")
                    continue
                df = df.rename(columns={date_col[0]: "Date"})
                df["Date"] = pd.to_datetime(df["Date"])

            df = df.dropna(subset=["Date"])
            start = df["Date"].min()
            end   = df["Date"].max()
            days  = len(df)

            records.append({
                "ticker":        ticker,
                "start_date":    start.strftime("%Y-%m-%d"),
                "end_date":      end.strftime("%Y-%m-%d"),
                "trading_days":  days,
            })

            #print(f"  {ticker:20s}  {start.strftime('%Y-%m-%d')}  →  {end.strftime('%Y-%m-%d')}  ({days} trading days)")

        except Exception as e:
            print(f"  [ERROR] {ticker}: {e}")

    return pd.DataFrame(records).sort_values("ticker").reset_index(drop=True)


def summarise(df: pd.DataFrame) -> None:
    """Print a summary of the date ranges across all tickers."""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total tickers:        {len(df)}")
    print(f"Earliest start date:  {df['start_date'].min()}")
    print(f"Latest end date:      {df['end_date'].max()}")
    print(f"Avg trading days:     {df['trading_days'].mean():.0f}")

    # # Show tickers that start before GDELT coverage (2013-01-01)
    # pre_gdelt = df[df["start_date"] < "2013-01-01"]
    # if not pre_gdelt.empty:
    #     print(f"\nTickers with data before 2013 (need BSE filings for full coverage):")
    #     for _, row in pre_gdelt.iterrows():
    #         print(f"  {row['ticker']:20s} starts {row['start_date']}")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Scanning NIFTY 50 dataset...\n")
    df = get_ticker_ranges(NIFTY50_DIR)

    if df.empty:
        print("No tickers found. Check your data directory.")
        return

    summarise(df)

    df.to_csv(TICKER_RANGES_PATH, index=False)
    print(f"\nSaved ticker ranges to: {TICKER_RANGES_PATH}")
    print("Run 02_fetch_bse.py next.")


if __name__ == "__main__":
    main()