"""
Step 2b — Fetch news headlines from GDELT for all 50 NIFTY tickers.
Covers 2013-01-01 to 2021-12-31 (GDELT reliable coverage window).

IMPORTANT: GDELT API officially supports only 3-month windows per query.
This script chunks the full date range into 3-month intervals and queries
each chunk separately to ensure complete coverage.

Returns: data/raw/gdelt_raw.csv with columns:
    date, ticker, headline, category, source

Usage: python src/03_fetch_gdelt.py
Prereq: 01_get_ticker_ranges.py must have been run first.
"""

import time
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
from gdeltdoc import GdeltDoc, Filters
from tqdm import tqdm
from config import (
    TICKER_RANGES_PATH,
    GDELT_RAW_PATH,
    RAW_DIR,
    GDELT_START_DATE,
    MAX_GDELT_ARTICLES,
)

# ── Trusted Indian financial news domains ─────────────────────────────────────
# Restricting to known reliable sources improves signal quality significantly
INDIAN_FINANCE_DOMAINS = [
    "economictimes.indiatimes.com",
    "livemint.com",
    "business-standard.com",
    "moneycontrol.com",
    "financialexpress.com",
    "ndtv.com",
    "thehindu.com",
    "reuters.com",
    "bloomberg.com",
    "ft.com",
]

# ── Company name mappings ─────────────────────────────────────────────────────
# GDELT searches by company name, not ticker symbol.
# These are the most common names used in news articles for each ticker.
TICKER_TO_SEARCH_NAME = {
    "ADANIPORTS":   "Adani Ports",
    "ASIANPAINT":   "Asian Paints",
    "AXISBANK":     "Axis Bank",
    "BAJAJ-AUTO":   "Bajaj Auto",
    "BAJFINANCE":   "Bajaj Finance",
    "BAJAJFINSV":   "Bajaj Finserv",
    "BPCL":         "Bharat Petroleum",
    "BHARTIARTL":   "Bharti Airtel",
    "BRITANNIA":    "Britannia Industries",
    "CIPLA":        "Cipla",
    "COALINDIA":    "Coal India",
    "DIVISLAB":     "Divi's Laboratories",
    "DRREDDY":      "Dr Reddy's Laboratories",
    "EICHERMOT":    "Eicher Motors",
    "GRASIM":       "Grasim Industries",
    "HCLTECH":      "HCL Technologies",
    "HDFCBANK":     "HDFC Bank",
    "HDFCLIFE":     "HDFC Life",
    "HEROMOTOCO":   "Hero MotoCorp",
    "HINDALCO":     "Hindalco Industries",
    "HINDUNILVR":   "Hindustan Unilever",
    "HDFC":         "HDFC Limited",
    "ICICIBANK":    "ICICI Bank",
    "ITC":          "ITC Limited",
    "INDUSINDBK":   "IndusInd Bank",
    "INFY":         "Infosys",
    "JSWSTEEL":     "JSW Steel",
    "KOTAKBANK":    "Kotak Mahindra Bank",
    "LT":           "Larsen Toubro",
    "M&M":          "Mahindra Mahindra",
    "MARUTI":       "Maruti Suzuki",
    "NTPC":         "NTPC Limited",
    "NESTLEIND":    "Nestle India",
    "ONGC":         "Oil Natural Gas Corporation",
    "POWERGRID":    "Power Grid Corporation",
    "RELIANCE":     "Reliance Industries",
    "SBILIFE":      "SBI Life Insurance",
    "SHREECEM":     "Shree Cement",
    "SBIN":         "State Bank India",
    "SUNPHARMA":    "Sun Pharmaceutical",
    "TCS":          "Tata Consultancy Services",
    "TATACONSUM":   "Tata Consumer Products",
    "TATAMOTORS":   "Tata Motors",
    "TATASTEEL":    "Tata Steel",
    "TECHM":        "Tech Mahindra",
    "TITAN":        "Titan Company",
    "UPL":          "UPL Limited",
    "ULTRACEMCO":   "UltraTech Cement",
    "WIPRO":        "Wipro",
    "ZEEL":         "Zee Entertainment",
}


def generate_3month_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """
    Split a date range into 3-month chunks.
    Returns list of (chunk_start, chunk_end) string tuples.
    """
    chunks = []
    current = date.fromisoformat(start)
    end_dt  = date.fromisoformat(end)

    while current < end_dt:
        chunk_end = min(current + relativedelta(months=3), end_dt)
        chunks.append((current.isoformat(), chunk_end.isoformat()))
        current = chunk_end

    return chunks


def fetch_gdelt_for_ticker(
    gd: GdeltDoc,
    ticker: str,
    search_name: str,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """
    Fetch all GDELT articles for one ticker across the full date range,
    querying in 3-month chunks to respect API limitations.
    """
    chunks  = generate_3month_chunks(start_date, end_date)
    records = []
    seen_titles = set()  # deduplicate within ticker

    for chunk_start, chunk_end in chunks:
        try:
            f = Filters(
                keyword    = search_name,
                start_date = chunk_start,
                end_date   = chunk_end,
                num_records= MAX_GDELT_ARTICLES,
                domain     = INDIAN_FINANCE_DOMAINS,
                language   = "English",
            )

            articles = gd.article_search(f)

            if articles is None or articles.empty:
                continue

            for _, row in articles.iterrows():
                title = str(row.get("title", "")).strip()

                # Skip empty or duplicate headlines
                if not title or title in seen_titles:
                    continue

                # Parse date from seendate field (format: YYYYMMDDTHHMMSSZ)
                raw_date = str(row.get("seendate", ""))
                try:
                    art_date = pd.Timestamp(raw_date).strftime("%Y-%m-%d")
                except Exception:
                    continue

                seen_titles.add(title)
                records.append({
                    "date":     art_date,
                    "ticker":   ticker,
                    "headline": title,
                    "category": "News",
                    "source":   row.get("domain", "GDELT"),
                })

        except Exception as e:
            tqdm.write(f"    [WARN] {ticker} chunk {chunk_start}→{chunk_end}: {e}")

        # Pause between chunks to avoid rate limiting
        time.sleep(1)

    return records


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not TICKER_RANGES_PATH.exists():
        raise FileNotFoundError(
            f"ticker_ranges.csv not found at {TICKER_RANGES_PATH}\n"
            "Run 01_get_ticker_ranges.py first."
        )

    ranges_df = pd.read_csv(TICKER_RANGES_PATH).set_index("ticker")
    gd        = GdeltDoc()
    all_records = []

    # Only fetch from GDELT_START_DATE onward regardless of ticker start
    gdelt_start = GDELT_START_DATE  # "2013-01-01" from config

    print(f"Fetching GDELT headlines for {len(TICKER_TO_SEARCH_NAME)} tickers")
    print(f"Window: {gdelt_start} → 2021-12-31")
    print(f"Querying in 3-month chunks...\n")

    for ticker, search_name in tqdm(TICKER_TO_SEARCH_NAME.items(), desc="Tickers"):

        if ticker not in ranges_df.index:
            tqdm.write(f"  [SKIP] {ticker} not in ticker_ranges.csv")
            continue

        # Ticker end date from dataset, capped at 2021-12-31
        ticker_end = min(str(ranges_df.loc[ticker, "end_date"]), "2021-12-31")

        tqdm.write(f"  {ticker} ({search_name}): {gdelt_start} → {ticker_end}")

        records = fetch_gdelt_for_ticker(
            gd, ticker, search_name, gdelt_start, ticker_end
        )

        tqdm.write(f"    → {len(records)} articles fetched")
        all_records.extend(records)

        # Pause between tickers
        time.sleep(2)

    if not all_records:
        print("\n[ERROR] No articles fetched. Check network and GDELT availability.")
        return

    df = (
        pd.DataFrame(all_records)
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    df.to_csv(GDELT_RAW_PATH, index=False)

    print(f"\n{'='*60}")
    print(f"GDELT fetch complete")
    print(f"{'='*60}")
    print(f"Total articles      : {len(df)}")
    print(f"Tickers covered     : {df['ticker'].nunique()}")
    print(f"Date range          : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\nTop sources:")
    print(df["source"].value_counts().head(10).to_string())
    print(f"\nSaved to: {GDELT_RAW_PATH}")
    print("\nRun 04_merge_clean.py next.")


if __name__ == "__main__":
    main()