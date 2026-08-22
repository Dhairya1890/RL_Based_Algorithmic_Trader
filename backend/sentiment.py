import requests
import json
import time
import os
from dotenv import load_dotenv
import google.generativeai as genai
import re
import yfinance as yf

# Load environment variables from .env file
load_dotenv()

SYSTEM_PROMPT = """You are a financial sentiment scorer for Indian equity markets.

Score news headlines based on their likely impact on the mentioned company's stock price.

SCORING RULES:

sentiment_score (float, -1.0 to +1.0):
  Positive = news likely to push stock UP
  Negative = news likely to push stock DOWN
  Zero     = no clear price impact

magnitude (float, 0.0 to 1.0):
  How strong and certain is the signal?
  0.7-1.0 : Specific figures mentioned, confirmed major event
  0.0-0.3 : Vague, speculative, routine

Return ONLY a valid JSON array with one object per headline, in the same order.
Each object must have exactly two keys: sentiment_score and magnitude.
No explanation. No markdown. No extra text."""

def build_user_prompt(headlines: list[str]) -> str:
    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
    return f"Score these {len(headlines)} headlines:\n\n{numbered}"

def parse_response(text: str, expected_count: int) -> list[dict]:
    text = re.sub(r"```json|```", "", text).strip()
    try:
        data = json.loads(text)
        if not isinstance(data, list) or len(data) != expected_count:
            return []
        for item in data:
            item["sentiment_score"] = max(-1.0, min(1.0, float(item.get("sentiment_score", 0.0))))
            item["magnitude"] = max(0.0, min(1.0, float(item.get("magnitude", 0.0))))
        return data
    except Exception:
        return []

def fetch_live_announcements(ticker: str, date: str) -> list[str]:
    try:
        yf_symbol = f"{ticker}.NS"
        stock = yf.Ticker(yf_symbol)
        news = stock.news
        headlines = []
        for item in news:
            if "content" in item:
                title = item["content"].get("title")
            else:
                title = item.get("title")
                
            if title:
                headlines.append(title)
        # Limit to the top 10 most recent headlines to conserve Gemini API tokens
        return headlines[:10]
    except Exception as e:
        print(f"Failed to fetch news from yfinance for {ticker}: {e}")
        return []

def score_headlines(headlines: list[str]) -> tuple[float, float]:
    if not headlines:
        return 0.0, 0.0
    
    try:
        # We assume GEMINI_API_KEY is in environment or loaded from .env
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=SYSTEM_PROMPT)
        resp = model.generate_content(
            build_user_prompt(headlines),
            generation_config=genai.types.GenerationConfig(temperature=0.1)
        )
        results = parse_response(resp.text, len(headlines))
        if not results:
            return 0.0, 0.0
            
        avg_score = sum(r['sentiment_score'] for r in results) / len(results)
        avg_mag = sum(r['magnitude'] for r in results) / len(results)
        return avg_score, avg_mag
    except Exception as e:
        print(f"Error scoring sentiment: {e}")
        return 0.0, 0.0

def get_live_sentiment(ticker: str, date: str, last_3d_rolling: float = 0.0):
    headlines = fetch_live_announcements(ticker, date)
    if not headlines:
        return {
            "Sentiment_Score": 0.0,
            "Sentiment_Magnitude": 0.0,
            "Article_Count": 0,
            "Sentiment_Rolling_3D": last_3d_rolling,
            "headlines": []
        }
        
    score, mag = score_headlines(headlines)
    new_rolling = (last_3d_rolling * 2 + score) / 3 if last_3d_rolling != 0.0 else score
    
    return {
        "Sentiment_Score": score,
        "Sentiment_Magnitude": mag,
        "Article_Count": len(headlines),
        "Sentiment_Rolling_3D": new_rolling,
        "headlines": headlines
    }
