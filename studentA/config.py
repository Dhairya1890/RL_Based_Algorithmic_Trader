"""
config.py

Central configuration file for the stock prediction project.
Edit values here instead of changing them throughout the code.
"""

from pathlib import Path

# ==========================================================
# Directories
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = PROJECT_ROOT / "reports"

FEATURE_IMPORTANCE_DIR = REPORT_DIR / "feature_importance"
SHAP_DIR = REPORT_DIR / "shap"
METRICS_DIR = REPORT_DIR / "metrics"

# Create directories automatically

for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    REPORT_DIR,
    FEATURE_IMPORTANCE_DIR,
    SHAP_DIR,
    METRICS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Dataset
# ==========================================================

DATE_COLUMN = "Date"

TARGET_COLUMN = "Target"

# Tomorrow Close > Today's Close

TARGET_SHIFT = 1

# ==========================================================
# Feature Engineering
# ==========================================================

SHORT_WINDOWS = [5, 10, 20]

LONG_WINDOWS = [50, 100, 200]

EMA_WINDOWS = [5, 10, 20, 50, 100, 200]

LAG_DAYS = list(range(1, 21))

VOLATILITY_WINDOWS = [5, 10, 20]

# ==========================================================
# Model
# ==========================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

N_SPLITS = 5

OPTUNA_TRIALS = 300

EARLY_STOPPING_ROUNDS = 50

# ==========================================================
# XGBoost
# ==========================================================

OBJECTIVE = "binary:logistic"

EVAL_METRIC = "logloss"

# ==========================================================
# Threshold Optimization
# ==========================================================

THRESHOLD_MIN = 0.30

THRESHOLD_MAX = 0.70

THRESHOLD_STEP = 0.01