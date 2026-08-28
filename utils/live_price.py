"""
Live price fetching utility using yfinance.
"""
import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(ttl=300)
def fetch_live_price(ticker: str) -> dict:
    """Fetches the latest live price and day's change for a NIFTY 50 ticker."""
    yf_symbol = f"{ticker}.NS"
    try:
        stock = yf.Ticker(yf_symbol)
        hist = stock.history(period="5d")
        if hist.empty:
            return {"error": "No data found for ticker"}
        
        latest_close = hist['Close'].iloc[-1]
        if len(hist) > 1:
            prev_close = hist['Close'].iloc[-2]
            change_pct = (latest_close - prev_close) / prev_close
        else:
            change_pct = 0.0
            
        return {
            "price": latest_close,
            "change_pct": change_pct,
            "volume": hist['Volume'].iloc[-1]
        }
    except Exception as e:
        return {"error": str(e)}

def fetch_live_features(ticker: str) -> pd.Series:
    """
    Fetches required features to make a live prediction.
    In a real app, this would re-calculate RSI, MACD, etc. from live data.
    """
    # For now, return a dummy series that the backtester can consume,
    # or fetch recent data and compute.
    # To keep this focused on UX, we will just return empty for now,
    # and the app will handle it or pass the last known row from master_df.
    pass
