# RL Environment v2 — Handoff Notes (Adithyasai, Sub-problem B)

## What changed from v1
- Data source switched from my own yfinance pull to **Nitesh's clean,
  feature-engineered `studentA/data/processed/` output** — real RSI,
  MACD, volatility, etc., not raw OHLCV.
- `NIFTY50_all.csv` (the corrupted, multi-stock file) is explicitly
  excluded in `config_env.py` and `load_studentA_features.py` — it
  will raise an error, not silently load, if anything tries to use it.
- State is now a single feature row (Nitesh's engineered columns +
  Dhairya's sentiment columns) instead of a 30-day raw-price window,
  since the engineered features already encode rolling history.
- `Tomorrow_Close`, `Tomorrow_Return`, `Target` — Nitesh's own XGBoost
  labels — are explicitly excluded from the feature set everywhere,
  with a check that raises an error if they ever leak in.

## How to run it, in order
```bash
python load_studentA_features.py   # sanity-check Nitesh's data loads
python sentiment_loader.py         # sanity-check the placeholder
python build_state_dataset.py      # merge price + sentiment
python split_dataset.py            # check train/test split sizes
python test_env_v2.py              # full validation suite
```

## ✅ Verified — actually run against your real uploaded data
Every file below has been executed against your real `studentA/data/processed/`
data, not just written and assumed correct:

- `load_studentA_features.py` — loads RELIANCE and TCS cleanly, 0 nulls
  in both. Confirmed `NIFTY50_all.csv` and an invalid ticker both raise
  clear errors instead of silently loading bad data.
- `build_state_dataset.py` — merges to a 2455 x 32 dataset with all 31
  expected feature columns present, no leakage columns.
- `split_dataset.py` — **caught and fixed a real bug here**: your data
  ends April 2021, so a hardcoded 2023 test window returned 0 test rows.
  Fixed to auto-detect the last full year of real data as a temporary
  test window (currently 2020-04-29 to 2021-04-29, 251 rows) and print
  exactly what it chose — but this is a stand-in, not a resolution (see
  question 1 below).
- `test_env_v2.py` Test 1 (Gymnasium compliance) — **PASSED**.
- `test_env_v2.py` Test 2 (random rollout, 300 steps) — **PASSED**, no crashes.
- `test_env_v2.py` Test 3 (buy-and-hold correctness) — **initially FAILED**
  at a 0.69% discrepancy. Root cause: an off-by-one bug in the *test
  script itself* (comparing from the wrong starting row against what the
  environment actually captures). Fixed and re-verified — now passes at
  a 0.019% difference, fully explained by the transaction cost. This is
  exactly the kind of bug the buy-and-hold check exists to catch.

## ⚠️ Blocking issue — needs team discussion today
**Nitesh's data ends April 2021.** Your project's evaluation plan uses
2023 as the held-out test window, but that year doesn't exist in this
dataset. The code now auto-falls-back to the last available year so
nothing breaks silently, but this is a workaround, not a fix — it needs
either updated source data reaching 2023, or the whole team agreeing to
formally redefine the test window.

---

## Questions for my mentor

1. **Data range** — since Nitesh's Kaggle data stops in April 2021, should
   we redefine our test window to the most recent available year in the
   data, or source updated data that reaches 2023?

2. **What exactly should I hand off to Harjap?** A Python class he
   imports directly (`NiftyTradingEnvV2`), a factory function
   (`make_env(ticker="RELIANCE")`), or should I register it as a formal
   Gymnasium environment ID (`gym.make("NiftyTrading-v0")`) so it behaves
   like any standard Gym environment? Is there a convention you'd
   recommend for a project this size?

3. **Single stock vs. all 50** — right now the environment trades one
   representative stock (`RELIANCE`, configurable). Should Harjap train
   one DQN per stock, or should the environment support resetting to a
   random ticker each episode so one DQN generalizes across all 50? This
   changes both my environment design and his training loop.

4. **Handing off before sentiment is ready** — Dhairya's real sentiment
   pipeline isn't built yet, so right now sentiment features are a
   placeholder of zeros. Should I hand this version to Harjap now so he
   can start Run A (price-only DQN) immediately, and give him an updated
   version later for Run B once real sentiment is merged in? Or should
   he wait for the complete pipeline before starting any training?
