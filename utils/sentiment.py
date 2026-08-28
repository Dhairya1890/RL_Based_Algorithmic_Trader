"""
Sentiment utility to fetch today's sentiment for a given ticker.
"""
import random
import pandas as pd

def fetch_today_sentiment(ticker: str) -> dict:
    """
    Fetches the most recent sentiment data for a ticker from the processed dataset.
    """
    from utils.data_loader import load_master_df
    master_df = load_master_df()
    
    if master_df.empty:
        return {}
        
    df_ticker = master_df[master_df['ticker'] == ticker]
    if df_ticker.empty:
        return {}
        
    latest_row = df_ticker.iloc[-1]
    
    return {
        "sentiment_score": float(latest_row.get("sentiment_score", 0.0)),
        "magnitude": float(latest_row.get("magnitude", 0.0)),
        "article_count": int(latest_row.get("article_count", 0)),
        "rolling_3d_avg": float(latest_row.get("rolling_3d_avg", 0.0)),
        "sentiment_available": int(latest_row.get("sentiment_available", 0)),
        "headlines": [] # Headlines require the live pipeline, omitted for static dataset
    }
