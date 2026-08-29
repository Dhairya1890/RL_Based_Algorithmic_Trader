# nifty-rl-env

Single-stock, Gymnasium-compatible reinforcement learning environment for NIFTY 50 equities. This module is **Sub-problem B** of the larger `RL_Based_Algorithmic_Trader` project: it turns a teammate's engineered price features (plus a sentiment placeholder) into a `gym.Env` that a DQN — or any other RL agent — can trade against.

## Contents
- [Quick facts](#quick-facts)
- [Required setup](#required-setup)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Pipeline: run order](#pipeline-run-order)
- [Environment specification](#environment-specification)
- [Data](#data)
- [Running the tests](#running-the-tests)
- [Known limitations and open questions](#known-limitations-and-open-questions)
- [Project context](#project-context)
- [See also](#see-also)

## Quick facts

| | |
|---|---|
| Framework | [Gymnasium](https://gymnasium.farama.org/) (`gym.Env` subclass) |
| Class | `NiftyTradingEnvV2` in `nifty_trading_env_v2.py` |
| Action space | `Discrete(3)` → 0 = Sell, 1 = Hold, 2 = Buy |
| Observation space | `Box(-inf, inf, shape=(32,), float32)` → 31 features + 1 position flag |
| Default ticker | `RELIANCE` (configurable; 49 tickers available) |
| Data range | 2011-06-01 to 2021-04-29 (2,455 trading days), same for every ticker |
| Initial capital | 100,000 (configurable; same units as the `Close` column — INR, since this is NSE data) |
| Transaction cost | 0.10% of portfolio value per position change (configurable) |
| Sentiment features | Placeholder zeros — real pipeline not merged in yet |
| Verified | Passes Gymnasium's `check_env`, a 300-step random rollout, and a buy-and-hold sanity check — see [Running the tests](#running-the-tests) |

## Required setup

> **None of the scripts in this folder will run until you do this.**

`config_env.py` points at:

```python
STUDENT_A_PROCESSED_DIR = "studentA/data/processed"
```

That folder doesn't exist anywhere in this repository — it looks like a local path from the original author's machine. The real feature-engineered CSVs it's looking for actually live in the sibling `baseline_performance/` module, at **`baseline_performance/data/processed/`** — confirmed by pointing the config there and running the full pipeline successfully end to end (see [Data](#data) and [Running the tests](#running-the-tests)). Pick one fix:

**Option 1 — symlink, no source changes.** Run from inside `nifty-rl-env/`:
```bash
mkdir -p studentA/data
ln -s "$(pwd)/../baseline_performance/data/processed" studentA/data/processed
```

**Option 2 — repoint the config directly.** Edit `config_env.py`:
```python
STUDENT_A_PROCESSED_DIR = "../baseline_performance/data/processed"
```
(No extra folders needed, but this edits the shared config file.)

Either way, confirm it worked:
```bash
python load_studentA_features.py
```
Expected output ends with:
```
Loaded RELIANCE: 2455 rows, 28 columns
...
Null check: 0 total nulls
```

## Installation

From the repo root:
```bash
pip install -r requirements.txt
```
This module specifically needs `gymnasium`, `pandas`, and `numpy`, all already listed there (verified against `gymnasium==1.3.0`).

## Quickstart

Run from inside `nifty-rl-env/` (or add it to `sys.path`) after completing [Required setup](#required-setup):

```python
from build_state_dataset import build_state_dataset
from split_dataset import split_train_test
from nifty_trading_env_v2 import NiftyTradingEnvV2

# 1. Merge the engineered price features with sentiment (currently placeholder zeros)
df = build_state_dataset(ticker="RELIANCE")

# 2. Chronological train/test split (see Known limitations re: the test window)
train_df, test_df = split_train_test(df)

# 3. Instantiate and step through the environment
env = NiftyTradingEnvV2(train_df, initial_capital=100_000.0, transaction_cost=0.001)
obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()      # swap in a trained policy
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(f"Final portfolio value: {info['portfolio_value']:,.2f}")
```

`NiftyTradingEnvV2` is a plain Gymnasium environment — it passes `gymnasium.utils.env_checker.check_env` — so it also drops directly into `stable_baselines3` or any other Gymnasium-compatible training loop.

## Pipeline: run order

Running `test_env_v2.py` alone exercises the whole chain (it imports `build_state_dataset`, which imports the two loaders below, plus `split_dataset`). Running each script standalone once, in this order, is a faster way to check each layer individually before training:

| # | Script | What it does |
|---|---|---|
| 1 | `load_studentA_features.py` | Loads one ticker's engineered-feature CSV and validates it against `config_env`'s required columns. Hard-fails rather than silently loading if you ask for the corrupted combined file or a leakage column. |
| 2 | `sentiment_loader.py` | `generate_placeholder_sentiment()` returns zero-filled sentiment columns for now; `load_real_sentiment_features()` is ready for when the real file lands. |
| 3 | `build_state_dataset.py` | Left-joins price/technical features with sentiment on `Date`, fills missing sentiment with 0, and re-validates the final column set. |
| 4 | `split_dataset.py` | Chronological (never random) train/test split. If no window is given, auto-picks the last full year present in the data and prints exactly what it chose. |
| 5 | `test_env_v2.py` | Full validation suite — see [Running the tests](#running-the-tests). |

```bash
python load_studentA_features.py
python sentiment_loader.py
python build_state_dataset.py
python split_dataset.py
python test_env_v2.py
```

## Environment specification

### Observation
A `Box(shape=(32,), dtype=float32)`: the 31 columns in `config_env.FEATURE_COLUMNS` (see [Data](#data)) from row `current_step - 1`, plus the current position flag (`0.0` = flat, `1.0` = long) appended at the end. Using row `t-1` rather than row `t` is the lookahead-bias guard — the agent decides using only information that was available before today's price move.

### Action
`Discrete(3)`:

| Action | Value | If flat (position = 0) | If long (position = 1) |
|---|---|---|---|
| Sell | 0 | No-op | Closes the position, charges transaction cost |
| Hold | 1 | No-op | No-op |
| Buy | 2 | Opens the position, charges transaction cost | No-op |

### Reward
```
daily_return = (Close[t] - Close[t-1]) / Close[t-1]
pnl          = portfolio_value * daily_return   if long after this step's action, else 0
trade_cost   = transaction_cost * portfolio_value   if this step opened/closed a position, else 0
reward       = (pnl - trade_cost) / initial_capital
portfolio_value  += pnl - trade_cost
```

Because the position flag updates *before* `pnl` is computed, a **Buy fills in time to earn that same step's price move**, while a **Sell exits in time to avoid it** — you keep everything accrued through the prior step, pay the cost, and stop there. Confirmed directly: with `transaction_cost=0`, a Buy on the first step reproduces that day's raw return exactly, and a Sell's reward on the following step is exactly `0.0`.

Reward is scaled by `initial_capital` (not the *current* portfolio value), so it stays roughly comparable across an episode even as `portfolio_value` compounds.

### Episode length and termination
`reset()` starts at `current_step = 1`. `terminated` fires once `current_step >= len(df) - 1`. `truncated` is always `False` — there's no separate time limit; wrap the env yourself with `gymnasium.wrappers.TimeLimit` if you want one.

One boundary detail worth knowing: `info['price']` always reports `Close` at the row the environment just advanced *to*. For every step but the last, that row goes on to be used as "today" in the *next* `step()` call — but on the final step, the episode ends before that can happen. So the terminal `info['price']` reflects the dataset's very last row, even though the last *realized* daily return folded into `portfolio_value` was computed through the second-to-last row. It's a single-row edge effect (confirmed by direct inspection). `test_env_v2.py`'s 0.5%-tolerance buy-and-hold check absorbs it, but a stricter external evaluation should measure returns between consecutive `info['price']` values rather than against the raw `Close` column, to stay consistent with what the environment actually simulated.

### Other methods
- `render()` — prints step, position, and portfolio value.
- `get_trade_log()` — returns a `pd.DataFrame` log (action, position, price, portfolio value) of every step taken since the last `reset()`.

## Data

### Feature columns (31 total — `config_env.FEATURE_COLUMNS`)

| Group | Count | Columns |
|---|---|---|
| Raw OHLCV-derived | 12 | `Prev Close, Open, High, Low, Last, Close, VWAP, Volume, Turnover, Trades, Deliverable Volume, %Deliverble` |
| Engineered technical | 15 | `Log_Return, Vol_10D, Vol_20D, Dist_SMA_10, Dist_SMA_20, Dist_SMA_50, RSI_14, MACD, MACD_Signal, MACD_Diff, BB_Pband, ATR_14, Volume_Ratio_20, VWAP_Dist, Deliverable_Pct` |
| Sentiment | 4 | `Sentiment_Score, Sentiment_Magnitude, Article_Count, Sentiment_Rolling_3D` (placeholder zeros until the real pipeline is merged) |

The raw and engineered columns come from the `baseline_performance` module (a teammate's feature-engineering pipeline). `Tomorrow_Close`, `Tomorrow_Return`, and `Target` — that pipeline's own forward-shifted labels — are explicitly excluded everywhere and will raise an error if they're ever detected in a feature set (`config_env.LEAKAGE_COLUMNS_NEVER_USE_AS_FEATURES`).

### Ticker universe
49 tickers are available via `load_studentA_features.list_available_tickers()`, one CSV per NIFTY 50 constituent. `NIFTY50_all.csv` (all 50 stocks concatenated into one file) is explicitly excluded — loading it as a single ticker would silently compute indicators across unrelated companies.

### Date range
Every ticker's engineered-feature data currently runs 2011-06-01 to 2021-04-29 (2,455 rows). See [Known limitations](#known-limitations-and-open-questions) for why this matters for the train/test split.

### About `nifty-rl-env/data/*.csv`
This folder (49 raw, pre-feature-engineering CSVs, byte-identical to `baseline_performance/data/raw/`) is **not read by any script in this pipeline**. It looks like a leftover from before the environment switched to consuming already-engineered features. The path that actually matters is the one in [Required setup](#required-setup).

## Running the tests

```bash
python test_env_v2.py
```

Three checks, run against the real data:

1. **Gymnasium API compliance** (`gymnasium.utils.env_checker.check_env`) — confirms `reset`/`step`/spaces conform to the Gymnasium API. A couple of `UserWarning`s about the observation space being unbounded (`-inf`/`inf`) are expected and non-fatal.
2. **Random rollout**, 300 steps — confirms nothing crashes under arbitrary actions. (No seed is fixed, so the exact reward total will vary run to run.)
3. **Buy-and-hold sanity check** — always-long from the first step should match a plain pandas buy-and-hold return, up to transaction cost. This is the most important test; it's the one that actually catches reward-logic bugs.

Verified output from a real run (deterministic — this one has no randomness involved):
```
Environment buy-and-hold return: 50.858327%
Manual (pandas) buy-and-hold return: 50.839670%
PASSED — difference of 0.018656% is explained by transaction cost.
```

## Known limitations and open questions

1. **Data ends April 2021, not 2023.** The project's original evaluation plan called for a 2023 held-out test window; the current source data doesn't reach it. `split_dataset.py` auto-falls-back to the last full year present (currently 2020-04-29 to 2021-04-29) so nothing breaks silently, but this is a stand-in — either newer source data or a team-agreed redefinition of the test window is still needed.
2. **Sentiment is a zero-filled placeholder**, not real data (`sentiment_loader.py`). Any result produced with the placeholder is equivalent to a price-only agent and shouldn't be read as evidence about whether sentiment helps. Swapping in the real file later is a one-line change: pass `use_real_sentiment_path=...` to `build_state_dataset()`.
3. **Single stock at a time.** The environment trades one ticker per instance (`RELIANCE` by default). Whether to train one agent per stock or have the environment reset to a random ticker each episode — so one agent generalizes across all 49 — is still an open design decision.
4. **Not registered as a formal Gym environment ID.** `NiftyTradingEnvV2` is imported directly as a class today, not via `gym.make("NiftyTrading-v0")`. That's fine for a single script; worth revisiting if multiple training scripts start depending on it.
5. **Not yet wired into the rest of the repo.** `backend/`, `streamlit_app.py`, and both `dqn_nifty50_trading_agent*.ipynb` notebooks at the repo root currently load raw CSVs and define their own environment/state logic rather than importing from this folder — so this module isn't yet the running app's actual source of truth for trading logic.

## Project context

This repo splits the trading-agent build into sub-problems across a few teammates; this folder is the RL-environment piece:

- **Feature engineering and baseline model** (`baseline_performance/`) — cleans raw OHLCV data and computes the technical/microstructure features this environment's `FEATURE_COLUMNS` are built from.
- **This module** (`nifty-rl-env/`) — turns those features (plus sentiment) into a tradeable Gymnasium environment.
- **Sentiment pipeline** (`sentiment_pipeline/`) — will eventually replace the placeholder in `sentiment_loader.py`.
- **Agent training** — the two DQN notebooks at the repo root currently implement this independently rather than building on `NiftyTradingEnvV2` (see limitation 5 above).

## See also
`HANDOFF_NOTES.md` in this folder is the original author's handoff memo: verification notes from the last working session, plus open questions about test-window redefinition, the handoff interface, single-stock-vs-all-49 design, and staging the sentiment rollout. Worth reading before acting on any of the items in [Known limitations](#known-limitations-and-open-questions) above.