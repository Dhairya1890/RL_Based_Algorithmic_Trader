"""
Step 3 — Clean and prepare BSE raw announcements for Gemini scoring.

Input:  data/raw/bse_raw.csv
Output: data/raw/combined_raw.csv  (cleaned, deduplicated, ready for scorer)

Cleaning steps:
  1. Drop rows with missing date, ticker, or headline
  2. Drop headlines under 10 characters (too short to score meaningfully)
  3. Drop exact duplicates (same ticker + date + headline)
  4. Drop routine low-signal headlines (board meeting notices, AGM dates)
  5. Sort by ticker and date

Usage: python src/04_merge_clean.py
Prereq: 02_fetch_bse.py must have been run first.
"""

import pandas as pd
import re
from config import BSE_RAW_PATH, RAW_DIR

COMBINED_RAW_PATH = RAW_DIR / "combined_raw.csv"

# ── Low-signal headline patterns to drop ─────────────────────────────────────
# These are routine procedural announcements with no price-moving information.
# Scoring them wastes Gemini quota and adds noise.
LOW_SIGNAL_PATTERNS = [
    r"^board meeting$",
    r"^outcome of board meeting$",
    r"^board meeting notice$",
    r"notice of board meeting",
    r"^agm notice$",
    r"^agm date$",
    r"annual general meeting notice",
    r"^intimation of board meeting$",
    r"closure of trading window",
    r"^trading window$",
    r"^loss of share certificate",
    r"^duplicate share certificate",
    r"change in address",
    r"^newspaper publication",
    r"^submission of",
    r"^compliance certificate",
    r"^reg\. \d+",          # regulatory compliance filings
    r"^regulation \d+",
]

LOW_SIGNAL_RE = re.compile(
    "|".join(LOW_SIGNAL_PATTERNS),
    flags=re.IGNORECASE,
)


def is_low_signal(headline: str) -> bool:
    return bool(LOW_SIGNAL_RE.search(headline.strip()))

def sample_company_updates(df: pd.DataFrame) -> pd.DataFrame:
    """
    For Company Update category, keep only 1 headline per (ticker, date).
    All other categories are kept in full.
    """
    updates  = df[df["category"] == "Company Update"]
    rest     = df[df["category"] != "Company Update"]

    # Keep the longest headline per (ticker, date) — more text = more signal
    updates_sampled = (
        updates
        .assign(length=updates["headline"].str.len())
        .sort_values("length", ascending=False)
        .drop_duplicates(subset=["ticker", "date"])
        .drop(columns=["length"])
    )

    return pd.concat([rest, updates_sampled], ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    original_count = len(df)
    print(f"Input rows: {original_count:,}")

    # 1. Drop missing values
    df = df.dropna(subset=["date", "ticker", "headline"])
    print(f"After dropping missing values : {len(df):,} rows")

    # 2. Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    print(f"After dropping invalid dates  : {len(df):,} rows")

    # 3. Drop headlines under 10 characters
    df = df[df["headline"].str.strip().str.len() >= 10]
    print(f"After dropping short headlines: {len(df):,} rows")

    # 4. Drop low-signal routine announcements
    mask = df["headline"].apply(is_low_signal)
    df = df[~mask]
    print(f"After dropping low-signal     : {len(df):,} rows")

    # 5. Drop exact duplicates
    df = df.drop_duplicates(subset=["ticker", "date", "headline"])
    print(f"After deduplication           : {len(df):,} rows")

    dropped = original_count - len(df)
    print(f"\nTotal rows dropped: {dropped:,} ({dropped/original_count*100:.1f}%)")

    # Sample company updates to 1 per (ticker, date)
    df = sample_company_updates(df)
    print(f"After sampling company updates: {len(df):,} rows")
    
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)



def main():
    if not BSE_RAW_PATH.exists():
        raise FileNotFoundError(
            f"bse_raw.csv not found at {BSE_RAW_PATH}\n"
            "Run 02_fetch_bse.py first."
        )

    print("Loading BSE raw data...")
    df = pd.read_csv(BSE_RAW_PATH)
    print(f"Loaded {len(df):,} rows from {BSE_RAW_PATH}\n")

    print("Cleaning...\n")
    df = clean(df)

    df.to_csv(COMBINED_RAW_PATH, index=False)

    print(f"\n{'='*60}")
    print(f"Cleaning complete")
    print(f"{'='*60}")
    print(f"Final rows          : {len(df):,}")
    print(f"Tickers covered     : {df['ticker'].nunique()}")
    print(f"Date range          : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\nRows per category:")
    print(df["category"].value_counts().to_string())
    print(f"\nSaved to: {COMBINED_RAW_PATH}")
    print(f"\nEstimated Gemini API calls at batch size 10: {len(df) // 10:,}")
    print(f"Estimated time at 15 req/min               : {len(df) // 10 // 15:.0f} mins")
    print("\nRun 05_score_sentiment.py next.")


if __name__ == "__main__":
    main()