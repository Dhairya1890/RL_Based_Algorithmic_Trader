"""
config.py

Centralized Configuration for Quantitative Machine Learning Pipeline.
Serves as the Single Source of Truth (SSOT) across all scripts.
"""

from pathlib import Path

# ==========================================================
# Directory Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_DIR = PROJECT_ROOT / "metadata"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure all required project directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, METADATA_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Pipeline Rules & Single Source of Truth
# ==========================================================

# Files to ignore in files/raw/ during ingestion
IGNORED_FILES = {
    "NIFTY50_all.csv",     # Stacked 50-stock file (Data poisoning risk if processed as 1 stock)
    "stock_metadata.csv",  # Non-price metadata
    "metadata.csv"
}

# Non-feature columns excluded from XGBoost model training
EXCLUDE_COLUMNS = [
    "Date", "Symbol", "Series", "Prev Close", "Open", "High", "Low", "Last", "Close",
    "VWAP", "Volume", "Turnover", "Trades", "Deliverable Volume", "%Deliverble",
    "Tomorrow_Close", "Tomorrow_Return", "Target", "Signal", "Position", "Trades_Count"
]

# Name of the pooled, generalized model artifact trained across all stocks
GENERALIZED_MODEL_NAME = "xgboost_generalized_50stocks.json"

# Core Constants
DATE_COLUMN = "Date"
TARGET_COLUMN = "Target"
TARGET_SHIFT = 1
TRANSACTION_COST_BPS = 0.0010  # 10 bps (0.10%) per trade (slippage + brokerage)