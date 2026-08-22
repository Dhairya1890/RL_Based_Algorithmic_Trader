import os
import warnings
from stable_baselines3 import DQN
import joblib
import numpy as np

# Suppress harmless warnings for cleaner console output
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message="This system does not have apparently enough memory.*")

model = None
scaler = None

def load_models():
    global model, scaler
    model_path = os.path.join(os.path.dirname(__file__), "models", "dqn_nifty50_generic_agent")
    scaler_path = os.path.join(os.path.dirname(__file__), "models", "dqn_nifty50_scaler.pkl")
    
    if os.path.exists(model_path + ".zip") and os.path.exists(scaler_path):
        # We pass buffer_size=1 to prevent SB3 from allocating a massive replay buffer for inference
        model = DQN.load(model_path, custom_objects={"buffer_size": 1})
        scaler = joblib.load(scaler_path)
        return True
    return False

def predict_action(features_df, sentiment_dict, current_position):
    """
    Features DF has 30 rows, 14 engineered features + 12 raw features = 26 columns.
    We need to add the 4 sentiment columns to make it 30 columns.
    """
    if model is None or scaler is None:
        raise RuntimeError("Model or scaler not loaded")
        
    df = features_df.copy()
    df['Sentiment_Score'] = sentiment_dict['Sentiment_Score']
    df['Sentiment_Magnitude'] = sentiment_dict['Sentiment_Magnitude']
    df['Article_Count'] = sentiment_dict['Article_Count']
    df['Sentiment_Rolling_3D'] = sentiment_dict['Sentiment_Rolling_3D']
    
    # Ensure correct order
    STATE_FEATURES = [
        # Raw (12)
        'Prev Close', 'Open', 'High', 'Low', 'Last', 'Close',
        'VWAP', 'Volume', 'Turnover', 'Trades',
        'Deliverable Volume', 'Deliverable_Pct',
        # Engineered (14)
        'Log_Return', 'Vol_10D', 'Vol_20D',
        'Dist_SMA_10', 'Dist_SMA_20', 'Dist_SMA_50',
        'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Diff',
        'BB_Pband', 'ATR_14', 'Volume_Ratio_20', 'VWAP_Dist',
        # Sentiment (4)
        'Sentiment_Score', 'Sentiment_Magnitude',
        'Article_Count', 'Sentiment_Rolling_3D',
    ]
    
    # Reorder columns and keep as DataFrame to retain feature names
    obs_df = df[STATE_FEATURES]  # shape: (30, 30)
    
    # Apply scaler.transform() on the (30, 30) DataFrame
    scaled_obs = scaler.transform(obs_df)
    flat_obs = scaled_obs.flatten()
    
    # Append the current position to match the expected (901,) shape
    final_obs = np.append(flat_obs, float(current_position))
    
    action, _ = model.predict(final_obs, deterministic=True)
    return int(action)
