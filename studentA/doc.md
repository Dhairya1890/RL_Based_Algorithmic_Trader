# Generalized Multi-Stock Directional Prediction Engine

An end-to-end Machine Learning pipeline built using **XGBoost Classifier** to predict whether tomorrow's closing price of a stock will be **HIGHER (1)** or **LOWER (0)** than today's closing price. 

The engine processes OHLCV stock market data, engineers stationary technical indicators and historical sequence lags, pools multi-stock datasets, and trains a single generalized model across all stocks while strictly avoiding look-ahead data leakage.

---

## 🛠️ Project Structure

### 1. Before Running the Pipeline
Before running any scripts, your repository contains the source code modules and raw input CSV files:

```text
ProjectFolder/
├── config.py                 # Central configuration file (Paths, Parameters, Constants)
├── feature_engineering.py    # Technical indicator & multi-day lag calculation engine
├── preprocess.py             # Data cleaning, validation, multi-core processing pipeline
├── train_xgboost.py          # Generalized multi-stock model training script
├── test_eval.py              # Out-of-sample evaluation & feature importance metrics
├── predict.py                # Batch & single-stock inference script
├── requirements.txt          # Environment dependencies
└── data/
    └── raw/                  # Place all 50+ raw stock CSV files here (e.g., ASIANPAINT.csv)

```

---

### 2. After Running the Pipeline

As you execute preprocessing, training, and prediction scripts, the system automatically builds output directories to store processed data, metadata, saved models, and prediction reports:

```text
ProjectFolder/
├── config.py
├── feature_engineering.py
├── preprocess.py
├── train_xgboost.py
├── test_eval.py
├── predict.py
├── requirements.txt
├── data/
│   ├── raw/                  # Input raw stock CSV files
│   └── processed/            # [NEW] Preprocessed CSVs with technical features & targets
├── metadata/                 # [NEW] JSON metadata files detailing cleaning stats per stock
├── models/                   # [NEW] Saved trained XGBoost model artifacts (.json)
└── reports/                  # [NEW] Output batch prediction CSV reports & metrics
    ├── feature_importance/   # [NEW] Feature importance plots/data
    ├── metrics/              # [NEW] Validation metrics logs
    ├── shap/                 # [NEW] Model explainability outputs
    └── latest_batch_predictions.csv  # Generated output from predict.py

```

---

## 📂 Purpose of Newly Created Folders & Files

| New Directory / File | Created By | Purpose & Contents |
| --- | --- | --- |
| `data/processed/` | `preprocess.py` | Stores preprocessed CSVs containing cleaned OHLCV data, computed technical indicators (RSI, MACD, Bollinger Bands, ATR), 1/2/3/5-day lag memory features, and 1-day forward target labels (`Target`). |
| `metadata/` | `preprocess.py` | Contains individual `{SYMBOL}.json` metadata files tracking data cleaning statistics (e.g., raw rows count, invalid OHLC rows removed, duplicate drops, positive/negative class balance, execution time). |
| `models/` | `train_xgboost.py` | Stores the trained generalized model file (`xgboost_generalized_50stocks.json`). This serialized XGBoost tree model is reused by `test_eval.py` and `predict.py` without retraining. |
| `reports/` | `config.py` & `predict.py` | Holds generated analysis logs, evaluation charts, and `latest_batch_predictions.csv` containing directional predictions for all raw stocks in `data/raw/`. |

---

## 🚀 Execution Instructions

Follow these steps in sequence to run the pipeline:

### Step 1: Environment Setup

Install all required Python dependencies:

```bash
pip install -r requirements.txt

```

### Step 2: Data Preprocessing & Feature Engineering

Run multi-core parallel processing across all raw CSV files in `data/raw/`. This calculates technical features, lag sequences, and forward target labels, saving processed outputs to `data/processed/`:

```bash
python preprocess.py

```

### Step 3: Train the Generalized XGBoost Model

Pool all processed CSV files into a unified dataset, split data chronologically to prevent time-series leakage, and train the regularized XGBoost classifier:

```bash
python train_xgboost.py

```

* **Output Artifact:** `models/xgboost_generalized_50stocks.json`

### Step 4: Model Evaluation & Feature Diagnostics

Evaluate model performance on unseen out-of-time test data (Accuracy, ROC-AUC, Log-Loss, Confusion Matrix) and inspect feature importances:

```bash
python test_eval.py

```

### Step 5: Run Batch / Single Stock Predictions

Predict tomorrow's price direction (UP / DOWN) for all stock CSVs in `data/raw/`:

```bash
python predict.py

```

To run inference on a specific single stock file:

```bash
python predict.py data/raw/ASIANPAINT.csv

```

---
