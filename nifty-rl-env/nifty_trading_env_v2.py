"""
nifty_trading_env_v2.py
-------------------------
Version 2 of my Gymnasium environment. Key change from v1: the state is
now built from Nitesh's real engineered features (RSI, MACD, volatility,
etc.) plus Dhairya's sentiment features, for ONE representative stock
selected via config_env.DEFAULT_TICKER — not a 30-day window of raw
OHLCV pulled from yfinance.

Why no more 30-day window: Nitesh's engineered features (Vol_10D,
Dist_SMA_50, RSI_14, etc.) already encode rolling history internally.
Stacking another 30-day window on top would be redundant. The state at
each step is now a single feature row, matching the team's NEEX.txt
state vector spec, plus the position flag.

Lookahead-bias prevention (unchanged principle from v1): the agent's
action at step t is decided using yesterday's feature row (t-1), and
reward is the return earned between yesterday's close and today's close
— never using information from the day it's currently deciding for.
"""

from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

import config_env as cfg


class NiftyTradingEnvV2(gym.Env):
    """
    Gymnasium environment trading ONE stock, using Nitesh's engineered
    features + Dhairya's sentiment features (placeholder until his real
    pipeline lands) as the state.

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_state_dataset.build_state_dataset() — must
        contain 'Date', 'Close', and every column in config_env.FEATURE_COLUMNS.
    initial_capital : float
    transaction_cost : float
        Fractional cost charged on the portfolio value whenever the
        agent changes position.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.001,
    ):
        super().__init__()

        missing = set(cfg.FEATURE_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"df is missing required feature columns: {missing}")
        if df[cfg.FEATURE_COLUMNS].isnull().any().any():
            raise ValueError("df contains nulls in feature columns — clean before use.")

        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.feature_columns = cfg.FEATURE_COLUMNS

        # Actions: 0 = Sell, 1 = Hold, 2 = Buy
        self.action_space = spaces.Discrete(3)

        # Observation: one feature row + position flag
        obs_len = len(self.feature_columns) + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32
        )

        self.current_step: int = 0
        self.position: int = 0
        self.portfolio_value: float = 0.0
        self._trade_log = []

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Start at step 1 (not 0) since step() always looks at
        # current_step - 1 for the observation and reference price.
        self.current_step = 1
        self.position = 0
        self.portfolio_value = self.initial_capital
        self._trade_log = []

        return self._get_observation(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        price_yesterday = self.df.loc[self.current_step - 1, "Close"]
        price_today = self.df.loc[self.current_step, "Close"]

        trade_cost = 0.0
        if action == 2 and self.position == 0:       # Buy
            self.position = 1
            trade_cost = self.transaction_cost * self.portfolio_value
        elif action == 0 and self.position == 1:      # Sell
            self.position = 0
            trade_cost = self.transaction_cost * self.portfolio_value
        # action == 1 (Hold), or a redundant Buy/Sell, is a no-op.

        daily_return = (price_today - price_yesterday) / price_yesterday
        pnl = self.portfolio_value * daily_return if self.position == 1 else 0.0

        self.portfolio_value = self.portfolio_value + pnl - trade_cost
        reward = (pnl - trade_cost) / self.initial_capital

        self._trade_log.append({
            "step": self.current_step,
            "action": action,
            "position": self.position,
            "price": price_today,
            "portfolio_value": self.portfolio_value,
        })

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def render(self):
        print(
            f"Step: {self.current_step:>5} | "
            f"Position: {'LONG' if self.position == 1 else 'FLAT':>4} | "
            f"Portfolio Value: {self.portfolio_value:,.2f}"
        )

    def _get_observation(self) -> np.ndarray:
        # Uses row (current_step - 1): the last row the agent could
        # actually know about before acting on current_step's price move.
        row = self.df.loc[self.current_step - 1, self.feature_columns].values.astype(np.float32)
        obs = np.concatenate([row, [float(self.position)]]).astype(np.float32)
        return obs

    def _get_info(self) -> Dict[str, Any]:
        idx = min(self.current_step, len(self.df) - 1)
        return {
            "step": self.current_step,
            "position": self.position,
            "portfolio_value": self.portfolio_value,
            "price": self.df.loc[idx, "Close"],
        }

    def get_trade_log(self) -> pd.DataFrame:
        return pd.DataFrame(self._trade_log)
