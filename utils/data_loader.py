"""
Data and model loading utilities.
"""
import os
import pandas as pd
import streamlit as st
import pickle

@st.cache_data
def load_master_df() -> pd.DataFrame:
    """Loads and merges price and sentiment features."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    price_path = os.path.join(base_dir, "data", "processed", "price_features.csv")
    sentiment_path = os.path.join(base_dir, "data", "processed", "sentiment_features.csv")
    
    # Create empty df as fallback
    df_price = pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume", "RSI", "MACD", "returns", "volatility"])
    df_sent = pd.DataFrame(columns=["date", "ticker", "sentiment_score", "magnitude", "article_count", "rolling_3d_avg", "sentiment_available"])
    
    if os.path.exists(price_path):
        try:
            df_price = pd.read_csv(price_path)
            df_price["date"] = pd.to_datetime(df_price["date"])
        except Exception as e:
            st.warning(f"Failed to read price features: {e}")
            
    if os.path.exists(sentiment_path):
        try:
            df_sent = pd.read_csv(sentiment_path)
            df_sent["date"] = pd.to_datetime(df_sent["date"])
        except Exception as e:
            st.warning(f"Failed to read sentiment features: {e}")
            
    if len(df_price) > 0 and len(df_sent) > 0:
        master_df = pd.merge(df_price, df_sent, on=["date", "ticker"], how="left")
    elif len(df_price) > 0:
        master_df = df_price.copy()
        # Fill missing sentiment columns
        for col in ["sentiment_score", "magnitude", "article_count", "rolling_3d_avg", "sentiment_available"]:
            master_df[col] = 0.0
    else:
        master_df = df_price.copy()
        for col in ["sentiment_score", "magnitude", "article_count", "rolling_3d_avg", "sentiment_available"]:
            master_df[col] = 0.0
            
    # Ensure date is sorted
    if not master_df.empty:
        master_df = master_df.sort_values(by=["ticker", "date"]).reset_index(drop=True)
        
    return master_df

@st.cache_resource
def load_models() -> dict:
    """Loads the RL and XGBoost models."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    
    models = {
        "dqn_run_a": None,
        "dqn_run_b": None,
        "xgb": None
    }
    
    try:
        from stable_baselines3 import DQN
        path_a = os.path.join(models_dir, "dqn_no_sentiment.zip")
        if os.path.exists(path_a):
            models["dqn_run_a"] = DQN.load(path_a, custom_objects={"buffer_size": 1})
            
        path_b = os.path.join(models_dir, "dqn_with_sentiment.zip")
        if os.path.exists(path_b):
            models["dqn_run_b"] = DQN.load(path_b, custom_objects={"buffer_size": 1})
    except ImportError:
        pass
    except Exception as e:
        print(f"Error loading DQN models: {e}")
        
    path_xgb = os.path.join(models_dir, "xgb_model.pkl")
    if os.path.exists(path_xgb):
        try:
            with open(path_xgb, "rb") as f:
                models["xgb"] = pickle.load(f)
        except Exception as e:
            print(f"Error loading XGB model: {e}")
            
    return models
