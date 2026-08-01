# XGBoost Baseline Model for NIFTY 50 Price Movement Prediction

## 1. Core Task Breakdown

### Sub-Problem
- **Objective:** Predict the next-day price movement direction (Up/Down or Buy/Sell signal) for stocks in the NIFTY 50 dataset.
- **Model:** XGBoost (Extreme Gradient Boosting).
- **Role in Team Project:** Establish the benchmark. You are proving what a traditional, non-RL supervised model can achieve based purely on price and technical features.

### Technique
- Supervised learning classification or regression mapped to trading decisions.
- Feature engineering derived from price and volume (OHLCV).

### App Component
- **Baseline Performance Panel:** A UI component (typically integrated into the team's Streamlit app) that displays your XGBoost model's metrics, predictions, backtest returns, Sharpe Ratio, and maximum drawdown alongside standard baselines like Buy-and-Hold.

---

## 2. Technical Roadmap for Implementation

### Step 1: Data Preparation & Feature Engineering
Using the NIFTY 50 dataset (`rohanrao/nifty50-stock-market-data` on Kaggle or via `yfinance`):

- **Clean & Handle Data:** Account for missing values, corporate actions, and align trading dates.
- **Engineer Technical Indicators:** Calculate features using libraries like `ta` or `pandas_ta`:
  - **Momentum:** RSI (Relative Strength Index), MACD, Stochastic Oscillator.
  - **Trend:** Simple and Exponential Moving Averages (SMA 20, 50, 200), ADX.
  - **Volatility:** Bollinger Bands, ATR (Average True Range).
  - **Volume:** OBV (On-Balance Volume), Volume Rate of Change.
- **Define Target Variable ($y_t$):**
  - Binary classification target:
    $$ y_t = 1 \quad \text{if} \quad \text{Close}_{t+1} > \text{Close}_t \quad \text{else} \quad 0 $$

### Step 2: Time-Series Cross-Validation & Model Training
- **Avoid Data Leakage:** Standard random $k$-fold cross-validation will leak future data. Use `TimeSeriesSplit` or expanding/rolling window cross-validation.
- **Model Pipeline:**
  - Train `XGBClassifier` on historical features.
  - Tune hyperparameters (learning rate, max depth, subsample, `colsample_bytree`) to prevent overfitting.

### Step 3: Convert Predictions to Trading Strategy & Backtest
- Map predictions to position states:
  - $1 \rightarrow \text{Long}$
  - $0 \rightarrow \text{Cash/Flat}$ (or $-1 \rightarrow \text{Short}$)
- Calculate strategy returns factoring in realistic transaction costs (e.g., 0.05% to 0.1% per trade).
- Compute core quantitative metrics:
  - Sharpe Ratio
  - Maximum Drawdown (Max DD)
  - Cumulative / Annualized Returns
  - Accuracy / Precision / AUC-ROC

---

## 3. Deliverables Summary

| Deliverable | Details |
| :--- | :--- |
| **Model Code** | Clean, modular Python scripts/notebooks (`train_xgboost.py`, `feature_engineering.py`). |
| **Strategy Metrics** | Summary table with Accuracy, Sharpe Ratio, Max Drawdown, and Cumulative Returns. |
| **Baseline Panel UI** | A modular Streamlit component (e.g., `render_baseline_panel()`) plotting your model's equity curve against Buy-and-Hold. |

---

## 4. Key Considerations for Quant Success

- **Look-ahead Bias:** Ensure all technical indicators calculated at step $t$ use data only up to step $t$.
- **Transaction Costs:** XGBoost can whip-saw (generate frequent buy/sell signals). Be sure to penalize frequent trading with transaction costs in your strategy backtest.
- **Handoff to Team:** Share your feature definitions with **Teammate B** (RL Environment designer) so feature spaces stay aligned across the project.

---

## 5. Pipeline

The following diagram illustrates the end-to-end workflow—from raw data ingestion to final backtest evaluation and UI visualization.

![Pipeline Diagram](/.src/pipline.png)


