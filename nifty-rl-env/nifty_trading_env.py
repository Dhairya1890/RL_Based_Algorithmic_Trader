import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class NiftyTradingEnv(gym.Env):
    """
    Custom Gymnasium environment for RL-based stock trading on NIFTY 50.

    Actions:
        0 - Sell
        1 - Hold
        2 - Buy

    Observation Space:
        Technical indicators + account metrics (balance, position, net worth, avg buy price)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, initial_capital: float = 100000.0, commission: float = 0.001):
        """
        Args:
            df (pd.DataFrame): Preprocessed stock data with technical features.
            initial_capital (float): Starting cash balance.
            commission (float): Transaction fee percentage.
        """
        super(NiftyTradingEnv, self).__init__()

        # TODO: Set up action space (Discrete 3)
        # TODO: Set up observation space (Box)
        # TODO: Initialize episode variables (balance, shares_held, net_worth, etc.)
        pass

    def reset(self, seed=None, options=None):
        """
        Resets the environment to start a new episode.

        Returns:
            observation (np.ndarray), info (dict)
        """
        super().reset(seed=seed)
        # TODO: Reset all episode state variables
        # TODO: Return initial observation and info dict
        pass

    def step(self, action: int):
        """
        Executes one trading step.

        Args:
            action (int): 0 = Sell, 1 = Hold, 2 = Buy

        Returns:
            obs, reward, terminated, truncated, info
        """
        # TODO: Execute buy/sell/hold logic
        # TODO: Update net worth
        # TODO: Calculate reward
        # TODO: Check termination and truncation
        # TODO: Return next observation and info
        pass

    def _get_observation(self) -> np.ndarray:
        """Builds and returns the current observation vector."""
        # TODO: Concatenate market features and account state
        pass

    def _get_info(self) -> dict:
        """Returns diagnostic info about the current state."""
        # TODO: Return step, balance, shares_held, net_worth, etc.
        pass

    def render(self, mode="human"):
        """Prints current portfolio state to console."""
        # TODO: Print step, price, net_worth, balance, shares, trades
        pass
