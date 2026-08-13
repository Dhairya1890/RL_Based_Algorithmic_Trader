"""
Step 2a — Fetch BSE corporate announcements for all 50 NIFTY tickers.
Uses the unofficial `bse` Python library which wraps BSE's internal API
with built-in rate limiting. No scraping, no browser automation needed.

Correct signature (v3.x):
    BSE.announcements(
        page_no=1,
        from_date=datetime,
        to_date=datetime,
        scripcode=str,
        category=str,       # from CATEGORY constants
        subcategory=str,
    ) -> Dict[str, List[dict]]
    Response keys: Table (list of announcements), Table1 (pagination info)

Usage: python src/02_fetch_bse.py
Prereq: 01_get_ticker_ranges.py must have been run first.
"""

import time
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from bse import BSE
from bse.constants import CATEGORY
from config import (
    TICKER_TO_BSE_CODE,
    TICKER_RANGES_PATH,
    BSE_RAW_PATH,
    RAW_DIR,
)

# ── Categories to fetch ───────────────────────────────────────────────────────
FETCH_CATEGORIES = [
    CATEGORY.RESULT,
    CATEGORY.ACTION,
    CATEGORY.BOARD_MEETING,
    CATEGORY.AGM,
    CATEGORY.UPDATE,
    CATEGORY.OTHERS,
]

CATEGORY_LABELS = {
    CATEGORY.RESULT:        "Financial Results",
    CATEGORY.ACTION:        "Corporate Action",
    CATEGORY.BOARD_MEETING: "Board Meeting",
    CATEGORY.AGM:           "AGM/EGM",
    CATEGORY.UPDATE:        "Company Update",
    CATEGORY.OTHERS:        "Others",
}


def fetch_all_pages(bse: BSE, scripcode: str, category: str,
                    from_date: datetime, to_date: datetime) -> list[dict]:
    """
    Paginate through all pages of announcements for one ticker + category.
    Returns a flat list of raw announcement dicts.
    """
    all_items = []
    page = 1

    while True:
        try:
            resp = bse.announcements(
                page_no=page,
                from_date=from_date,
                to_date=to_date,
                scripcode=scripcode,
                category=category,
            )
        except Exception as e:
            tqdm.write(f"      [WARN] page {page} error: {e}")
            break

        # Response is a dict with Table and Table1 keys
        if not resp or "Table" not in resp:
            break

        items = resp["Table"]
        if not items:
            break

        all_items.extend(items)

        # Check if more pages exist
        try:
            total = int(resp["Table1"][0]["ROWCNT"])
        except (KeyError, IndexError, TypeError):
            break

        if len(all_items) >= total:
            break

        page += 1
        time.sleep(0.5)  # be respectful between pages

    return all_items


def parse_record(item: dict, ticker: str, category_label: str) -> dict | None:
    """
    Extract and clean the fields we need from a raw BSE announcement dict.
    Returns None if the record is unusable.
    """
    # Date field (BSE uses different key names across versions)
    raw_date = (
        item.get("news_submission_dt")
        or item.get("DT_TM")
        or item.get("dt")
        or ""
    )
    try:
        ann_date = pd.Timestamp(raw_date).strftime("%Y-%m-%d")
    except Exception:
        return None

    # Headline field
    headline = (
        item.get("HEADLINE")
        or item.get("headline")
        or item.get("NEWSSUB")
        or item.get("subject")
        or ""
    ).strip()

    if not headline:
        return None

    return {
        "date":     ann_date,
        "ticker":   ticker,
        "headline": headline,
        "category": category_label,
        "source":   "BSE",
    }


def fetch_ticker(bse: BSE, ticker: str, scrip_code: str,
                 start_date: str, end_date: str) -> list[dict]:
    """Fetch all categories for one ticker within its date range."""
    records = []
    from_dt = pd.Timestamp(start_date).to_pydatetime()
    to_dt   = pd.Timestamp(end_date).to_pydatetime()

    for category in FETCH_CATEGORIES:
        label = CATEGORY_LABELS[category]
        try:
            items = fetch_all_pages(bse, scrip_code, category, from_dt, to_dt)
            for item in items:
                rec = parse_record(item, ticker, label)
                if rec:
                    records.append(rec)
        except Exception as e:
            tqdm.write(f"    [WARN] {ticker}/{label}: {e}")
            time.sleep(2)

    return records

def build_scrip_code_map(bse: BSE, tickers: list[str]) -> dict[str, str]:
    """Auto-resolve BSE scrip codes from ticker symbols."""
    mapping = {}
    print("Resolving BSE scrip codes...\n")
    for ticker in tickers:
        try:
            code = bse.getScripCode(ticker)
            if code:
                mapping[ticker] = str(code)
                print(f"  {ticker:20s} → {code}")
            else:
                print(f"  {ticker:20s} → NOT FOUND")
        except Exception as e:
            print(f"  {ticker:20s} → ERROR: {e}")
        time.sleep(0.5)
    return mapping

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not TICKER_RANGES_PATH.exists():
        raise FileNotFoundError(
            f"ticker_ranges.csv not found at {TICKER_RANGES_PATH}\n"
            "Run 01_get_ticker_ranges.py first."
        )

    ranges_df = pd.read_csv(TICKER_RANGES_PATH).set_index("ticker")
    all_records = []
    missing = []

    print(f"Fetching BSE announcements for {len(TICKER_TO_BSE_CODE)} tickers...\n")

    with BSE(download_folder=str(RAW_DIR)) as bse:
        
        tickers = list(ranges_df.index)
        scrip_map = build_scrip_code_map(bse, tickers)
        for ticker, scrip_code in tqdm(scrip_map.items(), ...):

            if ticker not in ranges_df.index:
                missing.append(ticker)
                continue

            start = ranges_df.loc[ticker, "start_date"]
            end   = min(str(ranges_df.loc[ticker, "end_date"]), "2021-12-31")

            tqdm.write(f"  {ticker} ({scrip_code}): {start} → {end}")
            records = fetch_ticker(bse, ticker, scrip_code, start, end)
            tqdm.write(f"    → {len(records)} announcements")
            all_records.extend(records)
            time.sleep(1)

    if not all_records:
        print("\n[ERROR] No announcements fetched. Check BSE codes and date ranges.")
        return

    df = (
        pd.DataFrame(all_records)
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    df.to_csv(BSE_RAW_PATH, index=False)

    print(f"\n{'='*60}")
    print(f"BSE fetch complete")
    print(f"{'='*60}")
    print(f"Total announcements : {len(df)}")
    print(f"Tickers covered     : {df['ticker'].nunique()}")
    print(f"Date range          : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\nBreakdown by category:")
    print(df["category"].value_counts().to_string())
    print(f"\nSaved to: {BSE_RAW_PATH}")

    if missing:
        print(f"\n[WARN] No BSE code mapping for: {missing}")

    print("\nRun 03_fetch_gdelt.py next.")


if __name__ == "__main__":
    main()