import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load environment variables from .env ──────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent.parent
DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
NIFTY50_DIR     = RAW_DIR / "nifty50"

BSE_RAW_PATH        = RAW_DIR / "bse_raw.csv"
GDELT_RAW_PATH      = RAW_DIR / "gdelt_raw.csv"
TICKER_RANGES_PATH  = PROCESSED_DIR / "ticker_ranges.csv"
SENTIMENT_OUT_PATH  = PROCESSED_DIR / "sentiment_features.csv"

# ── API keys ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ── Gemini settings ───────────────────────────────────────────────────────────
GEMINI_MODEL        = "gemini-1.5-flash"   # fast and cheap for batch scoring
GEMINI_RPM_LIMIT    = 15                   # free tier: 15 requests per minute
GEMINI_BATCH_SIZE   = 10                   # headlines per API call (batching saves quota)

# ── GDELT settings ────────────────────────────────────────────────────────────
GDELT_START_DATE    = "2013-01-01"         # GDELT reliable coverage starts here
MAX_GDELT_ARTICLES  = 250                  # per ticker per query

# ── BSE settings ──────────────────────────────────────────────────────────────
BSE_BASE_URL = "https://www.bseindia.com/corporates/ann.html"

# ── Sentiment feature columns (final output schema) ───────────────────────────
SENTIMENT_COLUMNS = [
    "date",
    "ticker",
    "sentiment_score",       # [-1.0, +1.0] weighted average across all sources
    "magnitude",             # [0.0, 1.0]  average confidence of scores
    "article_count",         # int — number of headlines that day
    "rolling_3d_avg",        # float — 3-day EMA of sentiment_score
    "sentiment_available",   # 0 or 1 — whether real data exists for this row
]