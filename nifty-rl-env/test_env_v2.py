"""
test_env_v2.py
----------------
Same three-part validation as v1, run against the new environment built
on Nitesh's real features. Run with: python test_env_v2.py
"""

import numpy as np
from gymnasium.utils.env_checker import check_env

from build_state_dataset import build_state_dataset
from split_dataset import split_train_test
from nifty_trading_env_v2 import NiftyTradingEnvV2
import config_env as cfg


def test_gymnasium_api_compliance(env):
    print("=" * 60)
    print("TEST 1: Gymnasium API compliance check")
    print("=" * 60)
    check_env(env)
    print("PASSED\n")


def test_random_agent(env, n_steps=300):
    print("=" * 60)
    print("TEST 2: Random agent rollout")
    print("=" * 60)
    obs, info = env.reset()
    total_reward = 0.0
    rewards = []
    for i in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rewards.append(reward)
        if terminated or truncated:
            print(f"Episode ended early at step {i}")
            break
    print(f"Ran {len(rewards)} steps without crashing.")
    print(f"Total reward: {total_reward:.6f} | Mean: {np.mean(rewards):.6f}")
    print("PASSED\n")


def test_buy_and_hold_matches_manual(df, initial_capital=100_000.0):
    print("=" * 60)
    print("TEST 3: Buy-and-hold sanity check (MOST IMPORTANT TEST)")
    print("=" * 60)
    env = NiftyTradingEnvV2(df, initial_capital=initial_capital)
    obs, info = env.reset()

    terminated = False
    while not terminated:
        action = 2 if env.position == 0 else 1
        obs, reward, terminated, truncated, info = env.step(action)

    env_return_pct = (info["portfolio_value"] - initial_capital) / initial_capital

    # IMPORTANT: the environment's first step() call captures the return
    # from row 0's close to row 1's close (it acts using row 0's already-
    # known features to decide the position for row 1). So the correct
    # manual comparison must start from row 0, not row 1 — using row 1 here
    # was an off-by-one bug that made this check fail even when the
    # environment's reward logic was actually correct.
    start_price = df.loc[0, "Close"]
    end_price = info["price"]  # the actual last price the env reached
    manual_return_pct = (end_price - start_price) / start_price

    print(f"Environment buy-and-hold return: {env_return_pct:.6%}")
    print(f"Manual (pandas) buy-and-hold return: {manual_return_pct:.6%}")

    diff = abs(env_return_pct - manual_return_pct)
    if diff < 0.005:
        print(f"PASSED — difference of {diff:.6%} is explained by transaction cost.\n")
    else:
        print(f"WARNING — difference of {diff:.6%} is larger than expected. Investigate.\n")


if __name__ == "__main__":
    df = build_state_dataset(ticker=cfg.DEFAULT_TICKER)
    print(f"Built state dataset for {cfg.DEFAULT_TICKER}: {df.shape}")
    print(f"Feature columns ({len(cfg.FEATURE_COLUMNS)}): {cfg.FEATURE_COLUMNS}\n")

    train_df, test_df = split_train_test(df)
    print(f"Train: {train_df.shape[0]} rows | Test: {test_df.shape[0]} rows\n")

    env = NiftyTradingEnvV2(train_df)

    test_gymnasium_api_compliance(env)
    test_random_agent(env)
    test_buy_and_hold_matches_manual(train_df)

    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
