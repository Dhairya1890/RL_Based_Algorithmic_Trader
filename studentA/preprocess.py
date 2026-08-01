"""
preprocess.py

Data Preprocessing Pipeline Orchestrator.

Responsibilities:
- Ingestion and validation of raw daily stock CSVs.
- Calls `compute_technical_features` from feature_engineering.py.
- Construct leakage-free target predictions.
- Save processed outputs and stock metadata.
- Multi-core CPU parallel execution via ProcessPoolExecutor.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Import modular feature engineering module
from feature_engineering import compute_technical_features

# ==========================================================
# Configuration & Directory Setup
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "files" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "files" / "processed"
METADATA_DIR = PROJECT_ROOT / "metadata"

DATE_COLUMN = "Date"
TARGET_COLUMN = "Target"
TARGET_SHIFT = 1

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("preprocess")


# ==========================================================
# Core Data Pipeline Functions
# ==========================================================

def load_and_validate_raw_csv(file_path: Path) -> pd.DataFrame:
    """Load stock CSV and assert minimum essential structure."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"Empty CSV detected: {file_path.name}")

    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing essential columns: {missing}")

    return df


def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean dates, drop duplicates, and filter unphysical price/volume rows."""
    stats = {"rows_raw": len(df), "duplicates_removed": 0, "invalid_ohlc_removed": 0}
    df_clean = df.copy()

    df_clean[DATE_COLUMN] = pd.to_datetime(df_clean[DATE_COLUMN])
    df_clean.sort_values(by=DATE_COLUMN, ascending=True, inplace=True)

    init_count = len(df_clean)
    df_clean.drop_duplicates(subset=[DATE_COLUMN], keep="last", inplace=True)
    stats["duplicates_removed"] = init_count - len(df_clean)

    num_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in num_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    valid_mask = (
        (df_clean["Open"] > 0)
        & (df_clean["High"] > 0)
        & (df_clean["Low"] > 0)
        & (df_clean["Close"] > 0)
        & (df_clean["Volume"] >= 0)
        & (df_clean["High"] >= df_clean["Low"])
        & (df_clean["High"] >= df_clean["Open"])
        & (df_clean["High"] >= df_clean["Close"])
        & (df_clean["Low"] <= df_clean["Open"])
        & (df_clean["Low"] <= df_clean["Close"])
    )

    rows_before_valid = len(df_clean)
    df_clean = df_clean[valid_mask].copy()
    stats["invalid_ohlc_removed"] = rows_before_valid - len(df_clean)

    return df_clean, stats


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct 1-day forward target predictions.
    Target = 1 if Tomorrow's Close > Today's Close else 0.
    """
    df_target = df.copy()

    df_target["Tomorrow_Close"] = df_target["Close"].shift(-TARGET_SHIFT)
    df_target["Tomorrow_Return"] = (
        df_target["Tomorrow_Close"] - df_target["Close"]
    ) / df_target["Close"]

    df_target[TARGET_COLUMN] = (df_target["Tomorrow_Return"] > 0).astype(int)
    df_target.dropna(subset=["Tomorrow_Close", "Tomorrow_Return"], inplace=True)

    return df_target


def process_single_stock(file_path: Path) -> Path:
    """Full processing workflow for a single stock CSV."""
    start_time = time.time()
    symbol = file_path.stem

    # 1. Ingest
    df_raw = load_and_validate_raw_csv(file_path)

    # 2. Clean
    df_clean, clean_stats = clean_dataframe(df_raw)

    # 3. Compute Features (imported from feature_engineering.py)
    df_features = compute_technical_features(df_clean, date_col=DATE_COLUMN)

    # 4. Generate Target
    df_processed = create_target(df_features)

    # 5. Drop rolling NaNs
    df_processed.dropna(inplace=True)

    # 6. Save Processed CSV
    output_path = PROCESSED_DATA_DIR / file_path.name
    df_processed.to_csv(output_path, index=False)

    # 7. Metadata Serialization
    rows_after = len(df_processed)
    target_dist = df_processed[TARGET_COLUMN].value_counts(normalize=True)
    pos_pct = float(target_dist.get(1, 0.0) * 100)
    neg_pct = float(target_dist.get(0, 0.0) * 100)
    exec_time = round(time.time() - start_time, 4)

    metadata = {
        "symbol": symbol,
        "rows_raw": clean_stats["rows_raw"],
        "rows_processed": rows_after,
        "features_generated": len(df_processed.columns),
        "duplicates_removed": clean_stats["duplicates_removed"],
        "invalid_ohlc_removed": clean_stats["invalid_ohlc_removed"],
        "start_date": str(df_processed[DATE_COLUMN].min()),
        "end_date": str(df_processed[DATE_COLUMN].max()),
        "positive_class_pct": round(pos_pct, 2),
        "negative_class_pct": round(neg_pct, 2),
        "execution_time_sec": exec_time,
    }

    meta_path = METADATA_DIR / f"{symbol}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    return output_path


def process_all_stocks() -> List[Path]:
    """Processes raw stock CSV files in parallel across CPU cores."""
    ignored_files = {"stock_metadata.csv", "metadata.csv"}
    raw_files = [
        f for f in RAW_DATA_DIR.glob("*.csv") 
        if f.name not in ignored_files and f.stat().st_size > 0
    ]

    if not raw_files:
        logger.warning("No valid stock CSV files found in directory: %s", RAW_DATA_DIR)
        return []

    logger.info("Found %d valid stock CSV file(s). Launching multi-core processing...", len(raw_files))
    processed_paths: List[Path] = []

    with ProcessPoolExecutor() as executor:
        future_map = {executor.submit(process_single_stock, file): file for file in raw_files}
        for future in as_completed(future_map):
            raw_file = future_map[future]
            try:
                out_path = future.result()
                processed_paths.append(out_path)
            except Exception as e:
                logger.error("Failed processing file %s: %s", raw_file.name, str(e))

    logger.info("Pipeline completed. Processed %d file(s) successfully.", len(processed_paths))
    return processed_paths


if __name__ == "__main__":
    logger.info("Initializing Preprocessing Pipeline...")
    process_all_stocks()
    logger.info("Execution complete.")