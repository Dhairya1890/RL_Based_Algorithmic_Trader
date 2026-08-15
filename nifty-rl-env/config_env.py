"""
config_env.py
--------------
Single source of truth for which columns are used, and which are banned.
Every other file imports from here instead of hardcoding column lists —
this is the exact DRY fix I flagged as missing in Nitesh's code, applied
to my own environment so it doesn't repeat the same mistake.
"""

# ----------------------------------------------------------------------
# Where Nitesh's CLEAN, per-stock, feature-engineered data lives.
# This is his data/processed/ folder — NOT his data/raw/ folder, and
# NOT the yfinance data I pulled myself in Week 1 (that's now retired).
# ----------------------------------------------------------------------
STUDENT_A_PROCESSED_DIR = "studentA/data/processed"

# Files in that folder that are NOT valid single-stock data and must
# NEVER be loaded as if they were one ticker. NIFTY50_all.csv mixes all
# 50 companies' rows together — loading it as a single stock silently
# computes indicators across unrelated companies.
EXCLUDED_FILES = {"NIFTY50_all.csv", "stock_metadata.csv"}

# ----------------------------------------------------------------------
# Feature groups — matches the team's NEEX.txt state vector spec, split
# by source so it's obvious which teammate owns which columns.
# ----------------------------------------------------------------------

# Raw OHLCV-derived numeric columns (from Nitesh, sub-problem A).
# NOTE: Date, Symbol, Series from NEEX.txt's "raw" group are identifiers,
# not numeric features — they are handled separately, never fed to the
# model as numbers.
RAW_NUMERIC_COLUMNS = [
    "Prev Close", "Open", "High", "Low", "Last", "Close",
    "VWAP", "Volume", "Turnover", "Trades",
    "Deliverable Volume", "%Deliverble",
]

# Engineered technical indicators (from Nitesh, sub-problem A).
ENGINEERED_COLUMNS = [
    "Log_Return", "Vol_10D", "Vol_20D",
    "Dist_SMA_10", "Dist_SMA_20", "Dist_SMA_50",
    "RSI_14", "MACD", "MACD_Signal", "MACD_Diff",
    "BB_Pband", "ATR_14", "Volume_Ratio_20",
    "VWAP_Dist", "Deliverable_Pct",
]

# Sentiment columns (from Dhairya, sub-problem D). Real data isn't
# delivered yet — see sentiment_loader.py for the placeholder.
SENTIMENT_COLUMNS = [
    "Sentiment_Score", "Sentiment_Magnitude",
    "Article_Count", "Sentiment_Rolling_3D",
]

# ----------------------------------------------------------------------
# CRITICAL — columns that must NEVER appear in the RL state vector.
# These are Nitesh's XGBoost labels, computed with a forward shift
# (tomorrow's price). Including any of these in the observation would
# be a severe lookahead-bias bug — the agent would be seeing the future.
# ----------------------------------------------------------------------
LEAKAGE_COLUMNS_NEVER_USE_AS_FEATURES = [
    "Tomorrow_Close", "Tomorrow_Return", "Target",
]

# Identifier columns — kept for merging/bookkeeping, never fed to the model.
IDENTIFIER_COLUMNS = ["Date", "Symbol", "Series"]

# The complete, final list of numeric feature columns used in the
# observation vector (excludes identifiers and leakage columns by
# construction — nothing else in the codebase should redefine this list).
FEATURE_COLUMNS = RAW_NUMERIC_COLUMNS + ENGINEERED_COLUMNS + SENTIMENT_COLUMNS

# Which single stock to build the environment on for now. Change this
# one line to build the env for a different ticker — nothing else in
# the codebase needs to change.
DEFAULT_TICKER = "RELIANCE"
