"""
Backtester logic to simulate trading strategies.
"""
import pandas as pd
import numpy as np

def run_strategy(model, df: pd.DataFrame, ticker: str, capital: float, strategy_name: str) -> pd.DataFrame:
    """
    Runs a backtest for a given model on a given dataframe of features.
    strategy_name: 'buyhold', 'xgb', 'dqn_run_a', 'dqn_run_b'
    """
    # Filter for ticker
    df_ticker = df[df['ticker'] == ticker].copy()
    if df_ticker.empty:
        return pd.DataFrame()
        
    df_ticker = df_ticker.sort_values('date').reset_index(drop=True)
    
    # Initialize portfolio tracking
    portfolio_values = []
    cash = capital
    shares = 0
    returns = []
    actions = []
    
    for i in range(len(df_ticker)):
        row = df_ticker.iloc[i]
        # Fallback to 0 if close price is missing or not a column
        price = row.get('close', 100.0)
        
        if strategy_name == 'buyhold':
            if i == 0:
                shares = cash / price
                cash = 0
            action = "HOLD"
        elif strategy_name == 'xgb':
            if model is not None:
                # XGBoost inference
                # Assuming basic numeric features as per the dataset spec
                feature_cols = ['open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD', 'returns', 'volatility']
                features = row[feature_cols].fillna(0).values.reshape(1, -1)
                pred = model.predict(features)[0]
                # Map standard classification 0=Buy, 1=Hold, 2=Sell
                action = "BUY" if pred == 0 else "SELL" if pred == 2 else "HOLD"
            else:
                action = "HOLD"
        elif strategy_name in ['dqn_run_a', 'dqn_run_b']:
            if model is not None:
                # DQN Inference
                if strategy_name == 'dqn_run_b':
                    cols = ['open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD', 'returns', 'volatility', 
                            'sentiment_score', 'magnitude', 'article_count', 'rolling_3d_avg', 'sentiment_available']
                else:
                    cols = ['open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD', 'returns', 'volatility']
                    
                # The model expects a specific observation vector shape. 
                # Since the user requested no mocks, we directly pass the features to model.predict
                features = row[cols].fillna(0).values.astype(np.float32)
                
                # Predict action
                action_idx, _ = model.predict(features, deterministic=True)
                action = "BUY" if action_idx == 0 else "SELL" if action_idx == 2 else "HOLD"
            else:
                action = "HOLD"
        else:
            action = "HOLD"
            
        actions.append(action)
        
        # Execute action (0.1% transaction cost)
        if action == "BUY" and cash > 0:
            shares += (cash * 0.999) / price
            cash = 0
        elif action == "SELL" and shares > 0:
            cash += (shares * price) * 0.999
            shares = 0
            
        current_value = cash + (shares * price)
        portfolio_values.append(current_value)
        
        if i == 0:
            returns.append(0.0)
        else:
            prev_val = portfolio_values[-2]
            returns.append((current_value - prev_val) / prev_val)
            
    df_ticker['portfolio_value'] = portfolio_values
    df_ticker['daily_return'] = returns
    df_ticker['action'] = actions
    
    return df_ticker
