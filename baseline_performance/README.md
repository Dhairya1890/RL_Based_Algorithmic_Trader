# NIFTY 50 Directional Forecasting — Baseline Model

> **Status:** Baseline model quantitative trading project.
> This module owns data ingestion, feature engineering, labeling, model training,
> evaluation, and inference for a **generalized, pooled-stock** next-day direction
> classifier. Later phases of the project (portfolio construction, risk management,
> live execution, alternative model families, etc.) are expected to build on top of
> the artifacts this pipeline produces.

---

## 1. What this baseline does

Given raw daily OHLCV data for NIFTY 50 constituents, this pipeline:

1. Cleans and validates each stock's price history.
2. Engineers ~27 **stationary**, scale-free technical/microstructure features.
3. Labels each row with a 1-day-forward **Up / Down** binary target.
4. Trains **one generalized LightGBM classifier** pooled across all 50 stocks,
   validated with walk-forward (`TimeSeriesSplit`) cross-validation.
5. Backtests the resulting signal with realistic execution lag and transaction costs.
6. Evaluates the final model on a strictly out-of-time holdout.
7. Serves next-day predictions for a single symbol or the full universe.

It is intentionally simple — a single pooled gradient-boosted-tree model on
hand-engineered features — so that it can act as a **performance and correctness
baseline** against which future, more sophisticated models (deep learning,
per-stock models, ensemble stacking, alternative targets, etc.) can be compared.

---

## 2. Project structure

```
project_root/
├── config.py                  # Single source of truth: paths, constants, column rules
├── feature_engineering.py     # Stateless feature computation on one OHLCV DataFrame
├── preprocess.py              # Ingestion, cleaning, feature call, labeling, batch driver
├── train.py                   # Pooled LightGBM training + walk-forward CV + backtest
├── test_eval.py                # Independent out-of-time evaluation harness
├── predict.py                  # Inference (single-symbol CLI or full-batch)
│
├── data/
│   ├── raw/                   # Input: one CSV per stock (Date, OHLCV, VWAP, ...)
│   └── processed/             # Output: cleaned + featurized + labeled CSVs
├── metadata/                  # Per-symbol JSON audit logs from preprocessing
├── models/                    # Serialized LightGBM artifacts (.txt, .joblib)
└── reports/                   # Batch prediction CSVs
```

All of the above directories are created automatically (`config.py`, on import)
if they don't already exist.

---

## 3. Pipeline architecture

```
Raw CSVs  →  Ingest & Clean  →  Engineer Features  →  Label  →  Train  →  Evaluate  →  Predict
(data/raw)   (preprocess.py)   (feature_engineering.py) (preprocess.py) (train.py) (test_eval.py) (predict.py)
```

Each arrow is a strict forward-in-time operation: rolling windows, target
construction, cross-validation splits, and the backtest itself all consume the
past only. This is the pipeline's central design discipline — see [@8](#8-design-principles).

Everything is orchestrated through `config.py`, which fixes directory paths,
excluded/ignored files, the target definition, and cost assumptions once, so
every other script reads from the same contract instead of hardcoding values.

---

## 4. Module-by-module documentation

### 4.1 `config.py` — Single Source of Truth

Centralizes everything the rest of the pipeline depends on so no script hardcodes
a path or a magic number.

| Constant | Purpose |
|---|---|
| `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `METADATA_DIR`, `MODELS_DIR`, `REPORTS_DIR` | Canonical directory layout; auto-created on import |
| `IGNORED_FILES` | `{"NIFTY50_all.csv", "stock_metadata.csv", "metadata.csv"}` — non-per-stock files excluded from ingestion so a 50-stock stacked file is never processed as if it were one stock |
| `EXCLUDE_COLUMNS` | Identifier, raw-price, and leakage columns kept out of the model's feature matrix (see [§4.4](#leakage-control)) |
| `GENERALIZED_MODEL_NAME` | Filename for the pooled model artifact |
| `DATE_COLUMN`, `TARGET_COLUMN` | `"Date"`, `"Target"` |
| `TARGET_SHIFT` | `1` — forecast horizon is fixed at one trading day |
| `TRANSACTION_COST_BPS` | `0.0010` (10 bps) — slippage + brokerage assumption used in the backtest |

**Why centralize this:** every downstream script (`preprocess.py`, `train.py`,
`test_eval.py`, `predict.py`) imports `config` rather than repeating these values.
Changing the forecast horizon, the cost assumption, or the excluded-column list
happens in exactly one place.

### 4.2 `feature_engineering.py` — Stationary Feature Engine

Exposes a single stateless function:

```python
compute_technical_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame
```

Takes a cleaned OHLCV DataFrame for **one stock**, sorts it chronologically, and
appends engineered columns. All rolling/EWM windows are strictly backward-looking
(`.rolling()`, `.shift()`, `.ewm()` — never a centered or forward window), so no
feature at row *t* ever uses information from row *t+1* or later.

#### Design rationale — why every feature is a return, ratio, or bounded oscillator

Raw prices (`Open`, `High`, `Low`, `Close`) are **non-stationary**: their scale
drifts over years and differs wildly across a ₹50 stock vs. a ₹3,000 stock. A
model trained on raw price levels would learn the price *era*, not the pattern,
and would fail to generalize — both across time and across the 50 pooled stocks.
Every feature below is therefore expressed as a **log return, a percentage
distance from an average, or a bounded 0–1 / 0–100 oscillator**, which keeps
all 50 stocks' histories on a comparable scale and lets one generalized model
be trained on the pooled dataset instead of 50 separate per-stock models.

#### Feature families

**1. Returns & Volatility** — the base layer everything else builds on

| Feature | Formula | Rationale |
|---|---|---|
| `Log_Return` | `ln(Close_t / Close_t-1)` | Additive, closer to normal than raw price change — the standard stationary return measure |
| `Vol_10D` | Rolling 10-day std of `Log_Return` | Short-horizon volatility regime |
| `Vol_20D` | Rolling 20-day std of `Log_Return` | Longer volatility baseline; `Vol_10D` vs `Vol_20D` spread signals a regime shift |

**2. Normalized Trend Indicators** — distance from moving averages, not the averages themselves

| Feature | Formula | Rationale |
|---|---|---|
| `Dist_SMA_10` | `(Close − SMA10) / SMA10` | Short-term trend deviation, ~2 trading weeks |
| `Dist_SMA_20` | `(Close − SMA20) / SMA20` | Medium-term deviation, ~1 trading month |
| `Dist_SMA_50` | `(Close − SMA50) / SMA50` | Longer trend context, ~1 quarter |

**3. Momentum Oscillators**

| Feature | Formula | Rationale |
|---|---|---|
| `RSI_14` | `100 − 100/(1 + avg_gain₁₄/avg_loss₁₄)` | Bounded 0–100 overbought/oversold gauge, naturally stationary |
| `MACD` | `(EMA12 − EMA26) / Close` | Trend-momentum crossover, **price-normalized** — the textbook MACD is priced in rupees and is not comparable across stocks; dividing by `Close` fixes this |
| `MACD_Signal` | 9-day EMA of `MACD` | Smoothed trigger line for crossover detection |
| `MACD_Diff` | `MACD − MACD_Signal` | The crossover histogram — accelerating vs. decelerating momentum in one number |

**4. Volatility Bands & Range**

| Feature | Formula | Rationale |
|---|---|---|
| `BB_Pband` | `(Close − BB_lower) / (BB_upper − BB_lower)` | Position inside the 20-day Bollinger Band, 0–1 scaled |
| `ATR_14` | 14-day avg. True Range `/ Close` | Normalized average daily trading range, comparable across stocks |

Both `RSI_14`'s ratio and `BB_Pband`'s denominator add a `1e-8` epsilon to avoid
division by zero on flat-price (e.g. halted) sessions.

**5. Volume & Market-Microstructure**

| Feature | Formula | Rationale |
|---|---|---|
| `Volume_Ratio_20` | `Volume / 20-day avg. Volume` | Flags abnormal participation, often precedes or confirms breakouts |
| `VWAP_Dist` *(optional)* | `(Close − VWAP) / VWAP` | Only computed if a `VWAP` column is present; shows close vs. day's volume-weighted average |
| `Deliverable_Pct` *(optional)* | `%Deliverble`, auto-scaled to a 0–1 fraction | NSE-specific proxy for genuine investor conviction vs. intraday speculative volume |

`VWAP_Dist` and `Deliverable_Pct` are each wrapped in an `if column in
df_feat.columns` guard, so the same engine runs unmodified on datasets that
lack these India-specific fields — new/optional features degrade gracefully
instead of raising.

**Total: 3 + 3 + 4 + 2 + 3 = 15 core numeric features**, plus the 2 optional
microstructure features when available (the presentation's "27+" figure
includes intermediate columns such as `MACD_Signal`'s inputs, raw indicator
components, and metadata columns present in the processed CSVs — the count
that reaches the model is governed by `config.EXCLUDE_COLUMNS`, see below).

### 4.3 `preprocess.py` — Ingestion, Cleaning, Labeling, Orchestration

| Function | Responsibility |
|---|---|
| `load_and_validate_raw_csv(file_path)` | Reads one raw CSV; raises if missing, empty, or lacking the required columns (`Date, Open, High, Low, Close, Volume`) |
| `clean_dataframe(df)` | Parses dates, sorts chronologically, drops duplicate trading dates, coerces OHLCV to numeric (corrupt values → `NaN`), and applies the **physical OHLC sanity mask** below |
| `create_target(df)` | Builds `Tomorrow_Close`, `Tomorrow_Return`, and the binary `Target` column; drops the final row (its future outcome doesn't exist yet) |
| `process_single_stock(file_path)` | Runs the full per-stock workflow: load → clean → feature-engineer → label → drop rolling-window NaNs → write processed CSV → write a metadata JSON |
| `process_all_stocks()` | Discovers all raw CSVs not in `IGNORED_FILES`, fans work out across CPU cores with `ProcessPoolExecutor`, and logs failures per file without halting the batch |

#### Physical OHLC sanity mask

A row survives `clean_dataframe()` only if **every** rule holds:

- `Open, High, Low, Close, Volume` are all `> 0` (`Volume ≥ 0`)
- `High` is the true maximum: `High ≥ Low`, `High ≥ Open`, `High ≥ Close`
- `Low` is the true minimum: `Low ≤ Open`, `Low ≤ Close`

**Why here, before feature engineering:** exchange feeds occasionally contain
zero-price ticks, split-adjustment glitches, or transposed High/Low fields.
Any one such row would silently corrupt every rolling-window feature computed
over it. Filtering at ingestion is cheaper and safer than filtering after
features have already propagated the error forward through 10–50 day windows.

#### Target construction

```
Tomorrow_Close  = Close.shift(-1)
Tomorrow_Return = (Tomorrow_Close - Close) / Close
Target          = 1 if Tomorrow_Return > 0 else 0
```

This is a **binary classification** problem, not a regression: the model
predicts direction only, never magnitude. This sidesteps the harder, noisier
problem of forecasting exact returns and keeps the label near a natural 50/50
base rate.

#### Metadata audit trail

Every processed symbol writes a JSON file to `METADATA_DIR` with row counts
before/after cleaning, duplicates removed, invalid-OHLC rows removed, date
range, class balance (`positive_class_pct` / `negative_class_pct`), and
execution time — a lightweight audit log for debugging data-quality issues
per stock.

### 4.4 `train.py` — Pooled Walk-Forward Training

| Function | Responsibility |
|---|---|
| `load_and_pool_processed_data()` | Concatenates all processed stock CSVs, sorts strictly by `Date`, and derives the feature column list as everything **not** in `config.EXCLUDE_COLUMNS` |
| `run_strategy_backtest(df, probas, cost_bps)` | Simulates a long/cash strategy on out-of-fold predictions with a 1-day execution offset and transaction friction |
| `train_lightgbm_generalized(n_splits=5)` | Runs the full walk-forward CV, reports diagnostics, retrains a final model on 100% of history, and serializes it |

<a name="leakage-control"></a>**Leakage control via `EXCLUDE_COLUMNS`:** the
feature matrix passed to LightGBM excludes `Date`, `Symbol`, `Series`, every
raw price/volume column (`Prev Close, Open, High, Low, Last, Close, VWAP,
Volume, Turnover, Trades, Deliverable Volume, %Deliverble`), and every
label/derived column that encodes the future or the model's own output
(`Tomorrow_Close, Tomorrow_Return, Target, Signal, Position, Trades_Count`).
Only the engineered, stationary features from `feature_engineering.py` reach
the model.

**Training procedure:**

1. Pool all 50 processed stock CSVs, sorted strictly by `Date`.
2. **5-fold `TimeSeriesSplit`** — each fold trains only on the past and
   validates only on a later, unseen future slice (no shuffling, no random
   splits — this is the walk-forward discipline applied to model selection).
3. **LightGBM** binary classifier with hyperparameters tuned for low
   signal-to-noise financial data: shallow trees (`max_depth=4, num_leaves=15`),
   `learning_rate=0.03`, `feature_fraction=0.8`, `bagging_fraction=0.8`,
   L1/L2 regularization (`reg_alpha=0.1, reg_lambda=1.0`), `min_child_samples=50`.
4. **Early stopping** (50 rounds) on validation log-loss per fold.
5. Aggregate out-of-fold predictions across all 5 folds for evaluation —
   metrics are never computed on training data.
6. **High-conviction filter**: predictions with probability `≥ 0.54` or
   `≤ 0.46` are evaluated separately, simulating a strategy that only trades
   its most confident calls (reported with coverage %).
7. **Financial backtest** (`run_strategy_backtest`):
   - **1-day execution offset** — a signal generated at `close(t)` executes at
     `close(t+1)`, never the same bar, to avoid an unrealistic same-bar fill.
   - **10 bps friction** (`config.TRANSACTION_COST_BPS`) charged on every
     position change (long → cash or cash → long).
   - Reports **CAGR, Sharpe Ratio, Max Drawdown, Win Rate**.
8. **Final retrain**: after CV, a last model is fit on 100% of chronological
   history (no holdout) for deployment, and both a LightGBM booster text file
   and a joblib-serialized sklearn wrapper are saved to `MODELS_DIR`.

### 4.5 `test_eval.py` — Independent Out-of-Time Evaluation

Deliberately separate from `train.py` so that final performance numbers come
from a script that never touched the training loop.

- Loads the serialized model and every processed stock CSV.
- Sorts unique trading dates and carves off the **final 15% chronologically**
  — dates the model never saw during CV, not even as a validation fold.
- Reports accuracy, ROC-AUC, log loss, confusion matrix, full classification
  report, and the **top-20 feature importances by Gain**.
- Re-applies the same 0.46–0.54 high-conviction filter for a final
  real-world-style accuracy check.

### 4.6 `predict.py` — Inference

| Function | Responsibility |
|---|---|
| `predict_stock_direction(symbol)` | Loads the trained model and one processed CSV, appends `Predicted_Signal` and `Confidence_Prob` columns |
| `predict_all_stocks()` | Batch-scores every processed file, keeps each stock's latest row, and writes `reports/latest_batch_predictions_lightgbm.csv` |

CLI usage:

```bash
python predict.py ADANIPORTS      # single-symbol mode — prints the last 10 rows
python predict.py                 # no argument — full batch sweep across all processed stocks
```

Per-symbol failures during batch inference are logged and skipped rather than
raising, so one malformed file never halts the full sweep.

---

## 5. End-to-end usage

```bash
# 1. Drop raw per-stock CSVs (Date, Open, High, Low, Close, Volume, [VWAP, %Deliverble, ...])
#    into data/raw/

# 2. Clean, engineer features, and label every stock
python preprocess.py

# 3. Train the pooled, walk-forward-validated LightGBM model
python train.py

# 4. Independently evaluate on the out-of-time holdout
python test_eval.py

# 5. Generate predictions
python predict.py                 # batch, all stocks
python predict.py ADANIPORTS      # single stock
```

**Required raw input schema:** `Date, Open, High, Low, Close, Volume` at
minimum; `VWAP` and `%Deliverble` are optional and unlock the corresponding
microstructure features when present.

---

## 6. Output artifacts

| Location | Contents |
|---|---|
| `data/processed/<SYMBOL>.csv` | Cleaned, featurized, labeled per-stock dataset |
| `metadata/<SYMBOL>.json` | Per-stock cleaning/labeling audit log |
| `models/lightgbm_generalized_50stocks.txt` | Raw LightGBM booster |
| `models/lightgbm_generalized_50stocks.joblib` | Sklearn-wrapper model, used by `test_eval.py` / `predict.py` |
| `reports/latest_batch_predictions_lightgbm.csv` | Latest-row prediction + confidence per stock |

---

## 7. Feature summary

| Family | Count | Features |
|---|---|---|
| Returns & Volatility | 3 | `Log_Return`, `Vol_10D`, `Vol_20D` |
| Trend Distance | 3 | `Dist_SMA_10`, `Dist_SMA_20`, `Dist_SMA_50` |
| Momentum | 4 | `RSI_14`, `MACD`, `MACD_Signal`, `MACD_Diff` |
| Volatility Bands & Range | 2 | `BB_Pband`, `ATR_14` |
| Volume & Microstructure | 2–3 | `Volume_Ratio_20`, `VWAP_Dist`*, `Deliverable_Pct`* |

`*` optional, present only when the raw feed includes `VWAP` / `%Deliverble`.

---

## 8. Design principles

1. **No look-ahead, anywhere.** Backward-only rolling windows, a shifted
   target, `TimeSeriesSplit` cross-validation, and a 1-day-offset backtest all
   move forward in time only.
2. **Stationarity over raw price.** Every feature is a return, a ratio, a
   percentage distance, or a bounded oscillator — never a raw rupee price —
   which is what makes pooling 50 stocks into a single generalized model valid.
3. **Realistic, not optimistic, backtests.** A 1-day execution lag and 10 bps
   transaction friction are charged on every simulated trade, not just netted
   out of a gross return figure.
4. **Config-driven and fault-tolerant.** `config.py` is the single source of
   truth for paths, target definition, and leakage-column exclusion; per-file
   failures in preprocessing and inference are logged and skipped rather than
   fatal.

---

## 9. Known limitations & notes for future project phases

Since this is the **baseline** subsystem, later phases should be aware of the
following simplifications made here:

- **Single pooled model, not per-stock or per-sector models.** All 50 stocks
  share one LightGBM classifier; regime or sector-specific behavior is not
  separately modeled.
- **Binary direction only** — no magnitude, no confidence-calibrated position
  sizing beyond the fixed 0.46/0.54 high-conviction cutoff.
- **Long/cash only** in the backtest — no short leg, no leverage, no
  portfolio-level position sizing or correlation control across the 50 names.
- **Fixed 1-day horizon** (`TARGET_SHIFT`) and **fixed 10 bps cost** —
  both are easy to vary via `config.py` but are not currently swept or tuned.
- **No hyperparameter search** — the LightGBM parameters in `train.py` are
  manually chosen defaults for low signal-to-noise data, not the output of a
  tuning process.
- **Feature set is hand-engineered technical/microstructure indicators only**
  — no fundamental, alternative, or cross-sectional (relative-strength-vs-index)
  data sources are included yet.

These are natural extension points for subsequent phases of the project
rather than defects in the baseline itself — the goal of this module is a
correct, leakage-free, realistically-costed reference point to measure future
improvements against.
