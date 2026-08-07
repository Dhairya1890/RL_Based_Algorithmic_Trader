# NiftyTradingEnv - Environment Specification
## Subproblem B: RL Environment for NIFTY 50 Trading
### Owner: Myadarapu Adithyasai

---

## Overview

The NiftyTradingEnv is a custom Gymnasium-compatible environment for training reinforcement learning agents on NIFTY 50 stock data. It simulates a paper trading environment with realistic constraints including transaction costs.

---

## Environment Details

| Property | Value |
|----------|-------|
| **Class** | `NiftyTradingEnv` |
| **Framework** | Gymnasium 0.28+ |
| **Action Space** | Discrete(3) |
| **Observation Space** | Box(low=-inf, high=inf, shape=(30, N)) |
| **Reward Type** | Continuous |

---

## Action Space

| Action | Value | Description | Condition |
|--------|-------|-------------|-----------|
| **BUY** | 0 | Buy shares | Only if not holding |
| **HOLD** | 1 | Do nothing | Always valid |
| **SELL** | 2 | Sell shares | Only if holding |

---

## Observation Space

The observation is a 30-day window of features:
