from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import datetime
import uvicorn
import math
import numpy as np

from .db import init_db, get_portfolio_history
from .portfolio import apply_trade, get_last_portfolio_state
from .features import fetch_and_prepare_features
from .sentiment import get_live_sentiment
from .model import load_models, predict_action

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_models()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NIFTY_50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", 
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL", 
    "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY", 
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", 
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC", 
    "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LTIM", 
    "LT", "M&M", "MARUTI", "NTPC", "NESTLEIND", "ONGC", "POWERGRID", 
    "RELIANCE", "SBILIFE", "SBIN", "SUNPHARMA", "TCS", "TATACONSUM", 
    "TATAMOTORS", "TATASTEEL", "TECHM", "TITAN", "UPL", "ULTRACEMCO", "WIPRO"
]

@app.get("/health")
def health_check():
    from .model import model
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/nifty50/symbols")
def get_symbols():
    return NIFTY_50

@app.get("/sentiment/{symbol}")
def get_sentiment_endpoint(symbol: str):
    if symbol not in NIFTY_50:
        raise HTTPException(status_code=404, detail="Symbol not found")
        
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Simple logic for rolling 3d average
    last_state = get_last_portfolio_state(symbol)
    rolling_3d = last_state["sentiment_score"] if last_state else 0.0
    
    sentiment = get_live_sentiment(symbol, date_str, rolling_3d)
    sentiment["date"] = date_str
    sentiment["symbol"] = symbol
    # Keys should be lower case according to spec
    return {
        "symbol": symbol,
        "date": date_str,
        "sentiment_score": sentiment["Sentiment_Score"],
        "sentiment_magnitude": sentiment["Sentiment_Magnitude"],
        "article_count": sentiment["Article_Count"],
        "sentiment_rolling_3d": sentiment["Sentiment_Rolling_3D"],
        "headlines": sentiment["headlines"]
    }

@app.get("/portfolio/{symbol}")
def get_portfolio(symbol: str):
    if symbol not in NIFTY_50:
        raise HTTPException(status_code=404, detail="Symbol not found")
    
    history = get_portfolio_history(symbol)
    if not history:
        return {
            "symbol": symbol,
            "initial_value": 1000000,
            "current_value": 1000000.0,
            "total_return_pct": 0.0,
            "history": []
        }
        
    current_value = history[-1]["portfolio_value"]
    initial_value = 1000000
    total_return = ((current_value / initial_value) - 1) * 100
    
    return {
        "symbol": symbol,
        "initial_value": initial_value,
        "current_value": current_value,
        "total_return_pct": round(total_return, 2),
        "history": history
    }

@app.post("/trade/{symbol}")
def run_trade(symbol: str):
    if symbol not in NIFTY_50:
        raise HTTPException(status_code=404, detail="Symbol not found")
        
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # 1-2. Fetch OHLCV & Features
    try:
        features_df = fetch_and_prepare_features(symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch features: {e}")
        
    if len(features_df) != 30:
        raise HTTPException(status_code=500, detail="Insufficient feature data (expected 30 days)")
    
    # Compute daily return from today's close vs yesterday's close
    # The last row in features_df represents "today"
    today_close = features_df.iloc[-1]['Close']
    yesterday_close = features_df.iloc[-1]['Prev Close']
    daily_return = (today_close - yesterday_close) / yesterday_close if yesterday_close else 0.0
    
    # 3. Sentiment
    last_state = get_last_portfolio_state(symbol)
    rolling_3d = last_state["sentiment_score"] if last_state else 0.0
    current_position = last_state["position"] if last_state else 0
    sentiment = get_live_sentiment(symbol, date_str, rolling_3d)
    
    # 4-6. Predict Action
    try:
        action_idx = predict_action(features_df, sentiment, current_position)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
        
    # 7-10. Apply Trade and Save State
    new_state = apply_trade(
        symbol=symbol, 
        date=date_str, 
        action_idx=action_idx, 
        daily_return=daily_return, 
        sentiment_score=sentiment["Sentiment_Score"],
        article_count=sentiment["Article_Count"]
    )
    
    # 11. Return JSON
    return {
        "symbol": new_state["symbol"],
        "date": new_state["date"],
        "action": new_state["action"],
        "position": new_state["position"],
        "portfolio_value": new_state["portfolio_value"],
        "daily_return_pct": new_state["daily_return_pct"],
        "sentiment_score": new_state["sentiment_score"],
        "article_count": new_state["article_count"]
    }
    
if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
