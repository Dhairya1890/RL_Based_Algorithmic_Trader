"""
train_demo.py
Demo: Training DQN Agent on NiftyTradingEnv
Shows integration with Subproblem C (Harjap's work)
Owner: Myadarapu Adithyasai (Subproblem B)
"""

import numpy as np
import pandas as pd
from nifty_trading_env import NiftyTradingEnv
import warnings
warnings.filterwarnings('ignore')


def demo_training():
    """Demonstrate training workflow."""
    print("=" * 60)
    print("DQN TRAINING DEMO")
    print("Integration with Subproblem C (Harjap)")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # Generate realistic price movement
    returns = np.random.randn(1000) * 0.01
    price = 10000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'date': dates,
        'ticker': ['NIFTY50'] * 1000,
        'Open': price * 0.99,
        'High': price * 1.01,
        'Low': price * 0.98,
        'Close': price,
        'Volume': 1000000 + np.random.randint(0, 500000, 1000),
        'RSI': 50 + np.random.randn(1000) * 10,
        'MACD': np.cumsum(np.random.randn(1000) * 0.05),
    })
    
    # Create environment
    print("✓ Creating environment...")
    env = NiftyTradingEnv(
        df=df,
        window_size=30,
        initial_balance=100000,
        transaction_cost=0.001,
        sentiment_cols=['RSI', 'MACD']
    )
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space}")
    
    # Try importing Stable-Baselines3 (required for DQN)
    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import EvalCallback
        
        print("✓ Stable-Baselines3 found")
        
        # Create vectorized environment
        vec_env = make_vec_env(lambda: env, n_envs=1)
        
        # Create DQN model
        print("✓ Creating DQN model...")
        model = DQN(
            'MlpPolicy',
            vec_env,
            learning_rate=0.001,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            verbose=1
        )
        
        # Train for a small number of steps (demo only)
        print("✓ Training for 5000 steps (demo)...")
        model.learn(total_timesteps=5000)
        model.save('demo_dqn_trader')
        print("✓ Model saved as 'demo_dqn_trader.zip'")
        
        # Evaluate the trained model
        print("✓ Evaluating trained model...")
        obs, info = env.reset()
        total_reward = 0
        
        for i in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                break
        
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Final portfolio value: ${info['portfolio_value']:,.2f}")
        print(f"  Total return: {(info['portfolio_value'] - 100000) / 100000 * 100:.2f}%")
        
        # Get performance metrics
        metrics = env.get_performance_metrics()
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"  Trades: {metrics['num_trades']}")
        
    except ImportError:
        print("⚠️ Stable-Baselines3 not installed.")
        print("  Install with: pip install stable-baselines3")
        print("  Running environment test only...")
        
        # Run random policy as fallback
        obs, info = env.reset()
        total_reward = 0
        
        for i in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                break
        
        print(f"  Random policy - Final value: ${info['portfolio_value']:,.2f}")
        print(f"  Random policy - Trades: {len(env.trades)}")
    
    env.close()
    print("=" * 60)


def compare_strategies():
    """Compare different trading strategies on the same environment."""
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)
    
    # Create data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    price = 10000 + np.cumsum(np.random.randn(500) * 5)
    
    df = pd.DataFrame({
        'date': dates,
        'ticker': ['NIFTY50'] * 500,
        'Open': price * 0.99,
        'High': price * 1.01,
        'Low': price * 0.98,
        'Close': price,
        'Volume': 1000000 + np.random.randint(0, 500000, 500),
    })
    
    env = NiftyTradingEnv(df=df, window_size=30)
    
    # Strategy 1: Buy and Hold
    print("\n📊 Buy and Hold Strategy:")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(0)  # BUY
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(1)  # HOLD
    metrics = env.get_performance_metrics()
    print(f"  Return: {metrics['total_return']*100:.2f}%")
    print(f"  Sharpe: {metrics['sharpe_ratio']:.4f}")
    print(f"  Drawdown: {metrics['max_drawdown']*100:.2f}%")
    
    # Strategy 2: Random Trading
    print("\n🎲 Random Strategy:")
    obs, info = env.reset()
    for i in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
    metrics = env.get_performance_metrics()
    print(f"  Return: {metrics['total_return']*100:.2f}%")
    print(f"  Sharpe: {metrics['sharpe_ratio']:.4f}")
    print(f"  Drawdown: {metrics['max_drawdown']*100:.2f}%")
    
    env.close()


if __name__ == "__main__":
    demo_training()
    compare_strategies()