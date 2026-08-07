"""
test_env.py
Test script for NiftyTradingEnv
Owner: Myadarapu Adithyasai (Subproblem B)
"""

import numpy as np
import pandas as pd
from nifty_trading_env import NiftyTradingEnv, create_env_from_csv
import warnings
warnings.filterwarnings('ignore')


def test_with_sample_data():
    """Test environment with generated sample data."""
    print("=" * 60)
    print("TEST: NiftyTradingEnv with Sample Data")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'ticker': ['NIFTY50'] * 500,
        'Open': 10000 + np.cumsum(np.random.randn(500) * 5),
        'High': 10000 + np.cumsum(np.random.randn(500) * 5) + 10,
        'Low': 10000 + np.cumsum(np.random.randn(500) * 5) - 10,
        'Close': 10000 + np.cumsum(np.random.randn(500) * 5),
        'Volume': 1000000 + np.random.randint(0, 500000, 500),
        # Optional: add some technical indicators as features
        'RSI': 50 + np.random.randn(500) * 10,
        'MACD': np.cumsum(np.random.randn(500) * 0.1),
    })
    
    # Create environment
    env = NiftyTradingEnv(
        df=df,
        window_size=30,
        initial_balance=100000,
        transaction_cost=0.001,
        sentiment_cols=['RSI', 'MACD']  # Treat these as extra features
    )
    
    print(f"✓ Environment created successfully")
    print(f"  Observation shape: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space}")
    print(f"  Number of features: {env.num_features}")
    
    # Test reset
    obs, info = env.reset()
    print(f"✓ Reset successful")
    print(f"  Initial portfolio value: ${info['portfolio_value']:,.2f}")
    print(f"  Observation shape: {obs.shape}")
    
    # Test random actions
    rewards = []
    portfolio_values = []
    
    for i in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        portfolio_values.append(info['portfolio_value'])
        
        if terminated:
            break
    
    # Results
    print(f"✓ Random trading completed")
    print(f"  Steps: {len(portfolio_values)}")
    print(f"  Final value: ${portfolio_values[-1]:,.2f}")
    print(f"  Total return: {(portfolio_values[-1] - 100000) / 100000 * 100:.2f}%")
    print(f"  Average reward: {np.mean(rewards):.6f}")
    print(f"  Trades: {len(env.trades)}")
    
    # Performance metrics
    metrics = env.get_performance_metrics()
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    
    env.close()
    print()


def test_with_your_data():
    """Test environment with your actual NIFTY 50 data."""
    print("=" * 60)
    print("TEST: NiftyTradingEnv with Your Data")
    print("=" * 60)
    
    try:
        # Try to load your data
        # Your data is in the 'data' folder with ticker names
        import os
        
        data_folder = 'data'
        csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        
        if not csv_files:
            print("No CSV files found in 'data' folder")
            return
        
        # Load the first CSV file as sample
        sample_file = os.path.join(data_folder, csv_files[0])
        print(f"Loading: {sample_file}")
        
        df = pd.read_csv(sample_file)
        print(f"✓ Data loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"  Columns: {list(df.columns)}")
        
        # Check if required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            print(f"⚠️ Missing columns: {missing}")
            print("  Checking for alternative column names...")
            
            # Try to find alternative names (lowercase)
            for col in df.columns:
                if col.lower() in ['open', 'o']:
                    df['Open'] = df[col]
                elif col.lower() in ['high', 'h']:
                    df['High'] = df[col]
                elif col.lower() in ['low', 'l']:
                    df['Low'] = df[col]
                elif col.lower() in ['close', 'c']:
                    df['Close'] = df[col]
                elif col.lower() in ['volume', 'v']:
                    df['Volume'] = df[col]
        
        # Ensure date column exists
        if 'date' not in df.columns and 'Date' in df.columns:
            df['date'] = df['Date']
        elif 'date' not in df.columns:
            df['date'] = range(len(df))
        
        # Create environment
        env = NiftyTradingEnv(
            df=df,
            window_size=30,
            initial_balance=100000,
            transaction_cost=0.001
        )
        
        print(f"✓ Environment created with your data")
        print(f"  Observation shape: {env.observation_space.shape}")
        
        # Run a quick test
        obs, info = env.reset()
        print(f"  Initial portfolio value: ${info['portfolio_value']:,.2f}")
        
        # Run 50 steps
        for i in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        
        print(f"  Final portfolio value: ${info['portfolio_value']:,.2f}")
        print(f"  Total return: {(info['portfolio_value'] - 100000) / 100000 * 100:.2f}%")
        print(f"  Trades: {len(env.trades)}")
        
        env.close()
        
    except Exception as e:
        print(f"Error: {e}")


def test_state_vector():
    """Test that state vector contains correct data."""
    print("=" * 60)
    print("TEST: State Vector Validation")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'ticker': ['NIFTY50'] * 100,
        'Open': np.random.randn(100) * 10 + 1000,
        'High': np.random.randn(100) * 10 + 1005,
        'Low': np.random.randn(100) * 10 + 995,
        'Close': np.random.randn(100) * 10 + 1000,
        'Volume': np.random.randint(0, 1000, 100),
        'Sentiment': np.random.uniform(-1, 1, 100),
    })
    
    env = NiftyTradingEnv(
        df=df,
        window_size=10,
        sentiment_cols=['Sentiment']
    )
    
    obs, info = env.reset()
    print(f"✓ State vector shape: {obs.shape}")
    print(f"  Window size: 10")
    print(f"  Features per step: {obs.shape[1]}")
    print(f"  Expected: 6 (OHLCV + Volume + Sentiment + Position)")
    print(f"  Actual: {obs.shape[1]}")
    
    # Check values
    print(f"  Feature names: {env.all_features + ['Position']}")
    print(f"  First observation sample: {obs[0, :5]}...")
    
    env.close()
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("NIFTY TRADING ENVIRONMENT - TEST SUITE")
    print("Owner: Myadarapu Adithyasai (Subproblem B)")
    print("=" * 60 + "\n")
    
    test_with_sample_data()
    test_state_vector()
    test_with_your_data()
    
    print("=" * 60)
    print("TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()