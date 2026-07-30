import os
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from data_loader import DataLoader
from nifty_trading_env import NiftyTradingEnv


def train_and_evaluate(ticker: str, total_timesteps: int = 30000):
    """
    Trains a PPO RL agent on stock data and evaluates it against a Buy-and-Hold baseline.

    Args:
        ticker (str): Stock ticker symbol to train and evaluate on.
        total_timesteps (int): Number of training timesteps for the PPO agent.
    """
    # TODO: Load and preprocess data using DataLoader
    # TODO: Chronological train/test split (80/20)
    # TODO: Instantiate train and test NiftyTradingEnv environments
    # TODO: Initialize and train PPO agent
    # TODO: Save trained model
    # TODO: Evaluate agent on test environment
    # TODO: Compute Buy-and-Hold baseline performance
    # TODO: Print performance metrics (final return %, trades taken)
    # TODO: Generate and save comparison plot (PPO vs Buy-and-Hold)
    pass


if __name__ == "__main__":
    # TODO: Get available tickers and run train_and_evaluate()
    pass
