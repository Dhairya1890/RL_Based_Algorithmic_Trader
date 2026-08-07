"""
config.py

Central configuration file for the stock prediction project.
"""

from pathlib import Path

# ==========================================================
# Directories
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_DIR = PROJECT_ROOT / "metadata"

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = PROJECT_ROOT / "reports"
FEATURE_IMPORTANCE_DIR = REPORT_DIR / "feature_importance"
SHAP_DIR = REPORT_DIR / "shap"
METRICS_DIR = REPORT_DIR / "metrics"

# Create directories automatically
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    METADATA_DIR,
    MODEL_DIR,
    REPORT_DIR,
    FEATURE_IMPORTANCE_DIR,
    SHAP_DIR,
    METRICS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Dataset Constants
# ==========================================================

DATE_COLUMN = "Date"
TARGET_COLUMN = "Target"
TARGET_SHIFT = 1

# Exclude non-feature columns across all scripts
EXCLUDE_COLUMNS = {
    "Date", "Symbol", "Series", "Prev Close", "Open", "High", "Low", "Last", 
    "Close", "VWAP", "Volume", "Turnover", "Trades", "Deliverable Volume", 
    "%Deliverble", "Tomorrow_Close", "Tomorrow_Return", "Target"
}

# ==========================================================
# Model & Training Constants
# ==========================================================

RANDOM_STATE = 42
OBJECTIVE = "binary:logistic"
EVAL_METRIC = "logloss"