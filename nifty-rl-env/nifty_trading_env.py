"""
nifty_trading_env.py
Subproblem B: Custom Gymnasium RL Environment for NIFTY 50 Trading
Owner: Myadarapu Adithyasai (Subproblem B)
Team: K_Means_Kuch_Bhi
Date: August 2026

This environment simulates trading on NIFTY 50 stocks using historical data.
It supports both price-only and sentiment-augmented state spaces.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')


class NiftyTradingEnv(gym.Env):
    """
    Custom Gymnasium environment for NIFTY 50 algorithmic trading.
    
    The environment simulates trading a single stock from the NIFTY 50 index
    using historical OHLCV data. It supports optional sentiment features
    as additional state variables.
    
    State (Observation):
        - Window of OHLCV data (default: 30 days)
        - Current position (0 = no position, 1 = holding)
        - Optional sentiment features (if provided)
    
    Actions (Discrete 3):
        - 0: BUY (only valid if not holding)
        - 1: HOLD (do nothing, maintain position)
        - 2: SELL (only valid if holding)
    
    Reward:
        rₜ = (Pₜ - Pₜ₋₁)/Pₜ₋₁ - c·|Δposition|
        where c = transaction cost rate
        
    Performance Metrics:
        - Sharpe Ratio: Risk-adjusted return
        - Maximum Drawdown: Worst peak-to-trough loss
        - Annualized Return: Compounded yearly growth
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 30,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        sentiment_cols: Optional[List[str]] = None,
        ticker_col: str = 'ticker',
        date_col: str = 'date',
        price_col: str = 'Close',  # Your data uses 'Close' with capital C
        ohclv_cols: Optional[List[str]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the trading environment.
        
        Args:
            df: DataFrame with OHLCV and optional sentiment data
            window_size: Number of days to look back (default: 30)
            initial_balance: Starting capital (default: 100,000)
            transaction_cost: Cost per trade as fraction (default: 0.001 = 0.1%)
            sentiment_cols: List of sentiment feature column names
            ticker_col: Column name for ticker symbol
            date_col: Column name for date
            price_col: Column name for price (your data uses 'Close')
            ohclv_cols: List of OHLCV column names
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        # Validate input
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")
        
        # Store parameters
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = float(initial_balance)
        self.transaction_cost = float(transaction_cost)
        self.ticker_col = ticker_col
        self.date_col = date_col
        self.price_col = price_col
        
        # Set OHLCV columns (your data uses capital letters)
        if ohclv_cols is None:
            self.ohclv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        else:
            self.ohclv_cols = ohclv_cols
        
        # Ensure price column is in OHLCV
        if price_col not in self.ohclv_cols:
            self.ohclv_cols.append(price_col)
        
        # Identify sentiment columns
        self.sentiment_cols = sentiment_cols if sentiment_cols else []
        
        # Build feature list
        self.price_features = self.ohclv_cols
        self.all_features = self.price_features + self.sentiment_cols
        self.all_features = list(dict.fromkeys(self.all_features))  # Remove duplicates
        
        # Check for missing columns
        missing = [col for col in self.all_features if col not in self.df.columns]
        if missing:
            print(f"Warning: Missing columns: {missing}")
            # Filter out missing columns
            self.all_features = [col for col in self.all_features if col in self.df.columns]
        
        # If no features left, use only price column
        if not self.all_features:
            self.all_features = [self.price_col]
            print(f"Warning: Using only {self.price_col} column")
        
        # Calculate number of features
        self.num_features = len(self.all_features) + 1  # +1 for position
        
        # Define action space
        self.action_space = spaces.Discrete(3)
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, self.num_features),
            dtype=np.float32
        )
        
        # Set seed
        self.seed(seed)
        
        # Initialize state
        self.reset()
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set random seed."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Returns:
            observation: Initial state
            info: Additional information
        """
        if seed is not None:
            self.seed(seed)
        
        # Reset trading state
        self.current_step = self.window_size
        self.position = 0  # 0 = no position, 1 = holding
        self.balance = self.initial_balance
        self.shares_held = 0
        self.portfolio_value = self.balance
        self.trades = []
        self.returns = []
        self.portfolio_history = [self.balance]
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            'step': self.current_step,
            'position': self.position,
            'balance': self.balance,
            'portfolio_value': self.portfolio_value,
            'date': self._get_current_date()
        }
        
        return observation, info
    
    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0=BUY, 1=HOLD, 2=SELL
            
        Returns:
            observation: Next state
            reward: Reward for this action
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional information
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Get current and next prices
        current_price = self.df.iloc[self.current_step][self.price_col]
        
        # Check if episode would end
        if self.current_step + 1 >= len(self.df):
            observation = self._get_observation()
            info = {
                'step': self.current_step,
                'position': self.position,
                'balance': self.balance,
                'portfolio_value': self.portfolio_value,
                'terminated': True,
                'date': self._get_current_date()
            }
            return observation, 0.0, True, False, info
        
        next_price = self.df.iloc[self.current_step + 1][self.price_col]
        
        # Calculate price return
        price_return = (next_price - current_price) / current_price
        
        # Store previous position
        previous_position = self.position
        
        # Execute action
        if action == 0:  # BUY
            if self.position == 0 and self.balance > 0:
                self.shares_held = self.balance / current_price
                self.balance = 0
                self.position = 1
                self.trades.append({
                    'step': self.current_step,
                    'type': 'BUY',
                    'price': current_price,
                    'shares': self.shares_held,
                    'date': self._get_current_date()
                })
        elif action == 2:  # SELL
            if self.position == 1 and self.shares_held > 0:
                self.balance = self.shares_held * current_price
                self.shares_held = 0
                self.position = 0
                self.trades.append({
                    'step': self.current_step,
                    'type': 'SELL',
                    'price': current_price,
                    'shares': self.shares_held,
                    'date': self._get_current_date()
                })
        # action == 1: HOLD - do nothing
        
        # Calculate transaction cost
        position_change = abs(self.position - previous_position)
        transaction_cost = self.transaction_cost * position_change
        
        # Calculate reward
        reward = price_return - transaction_cost
        
        # Update portfolio value
        if self.position == 1:
            self.portfolio_value = self.shares_held * next_price
        else:
            self.portfolio_value = self.balance
        
        # Store return
        self.returns.append(price_return)
        self.portfolio_history.append(self.portfolio_value)
        
        # Move to next step
        self.current_step += 1
        
        # Get next observation
        observation = self._get_observation()
        
        # Check termination
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        # Prepare info
        info = {
            'step': self.current_step,
            'position': self.position,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'portfolio_value': self.portfolio_value,
            'price_return': price_return,
            'transaction_cost': transaction_cost,
            'total_return': (self.portfolio_value - self.initial_balance) / self.initial_balance,
            'date': self._get_current_date(),
            'action_taken': ['BUY', 'HOLD', 'SELL'][action]
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation window."""
        start = self.current_step - self.window_size
        end = self.current_step
        
        # Get feature values
        feature_values = self.df.iloc[start:end][self.all_features].values
        
        # Add position as feature
        position_array = np.full((self.window_size, 1), self.position, dtype=np.float32)
        
        # Combine
        observation = np.concatenate([feature_values, position_array], axis=1)
        
        return observation.astype(np.float32)
    
    def _get_current_date(self) -> str:
        """Get current date string."""
        if self.date_col in self.df.columns:
            return str(self.df.iloc[self.current_step][self.date_col])
        return str(self.current_step)
    
    def render(self, mode: str = 'human'):
        """Render the environment."""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Position: {'HOLDING' if self.position == 1 else 'NOT HOLDING'}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Total Return: {(self.portfolio_value - self.initial_balance) / self.initial_balance * 100:.2f}%")
            print(f"Trades: {len(self.trades)}")
            print("-" * 50)
    
    def close(self):
        """Clean up."""
        pass
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary with metrics: total_return, sharpe_ratio, max_drawdown, num_trades
        """
        if not self.returns:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'num_trades': 0
            }
        
        returns_array = np.array(self.returns)
        
        # Total return
        total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(returns_array) > 1 and np.std(returns_array) > 1e-6:
            sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.trades)
        }


def create_env_from_csv(
    price_csv_path: str,
    sentiment_csv_path: Optional[str] = None,
    ticker: Optional[str] = None,
    **kwargs
) -> NiftyTradingEnv:
    """
    Factory function to create environment from CSV files.
    
    Args:
        price_csv_path: Path to price features CSV
        sentiment_csv_path: Optional path to sentiment features CSV
        ticker: Optional ticker symbol to filter
        **kwargs: Additional arguments for NiftyTradingEnv
        
    Returns:
        NiftyTradingEnv instance
    """
    # Load price data
    price_df = pd.read_csv(price_csv_path)
    
    # Filter by ticker if specified
    if ticker is not None and 'ticker' in price_df.columns:
        price_df = price_df[price_df['ticker'] == ticker]
    
    # Load sentiment data if provided
    if sentiment_csv_path:
        sentiment_df = pd.read_csv(sentiment_csv_path)
        
        # Filter by ticker
        if ticker is not None and 'ticker' in sentiment_df.columns:
            sentiment_df = sentiment_df[sentiment_df['ticker'] == ticker]
        
        # Merge on date and ticker
        merged_df = pd.merge(
            price_df,
            sentiment_df,
            on=['date', 'ticker'],
            how='left'
        )
        # Fill NaN with 0
        merged_df = merged_df.fillna(0)
        
        # Get sentiment columns
        sentiment_cols = [col for col in sentiment_df.columns 
                         if col not in ['date', 'ticker']]
    else:
        merged_df = price_df
        sentiment_cols = []
    
    # Sort by date
    if 'date' in merged_df.columns:
        merged_df['date'] = pd.to_datetime(merged_df['date'])
        merged_df = merged_df.sort_values('date').reset_index(drop=True)
    
    # Create environment
    env = NiftyTradingEnv(
        df=merged_df,
        sentiment_cols=sentiment_cols,
        **kwargs
    )
    
    return env