import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="NIFTY 50 Paper Trading", layout="wide")

def fetch_symbols():
    try:
        r = requests.get(f"{API_BASE}/nifty50/symbols")
        return r.json() if r.status_code == 200 else []
    except:
        return []

def fetch_portfolio(symbol):
    try:
        r = requests.get(f"{API_BASE}/portfolio/{symbol}")
        return r.json() if r.status_code == 200 else None
    except:
        return None

def fetch_sentiment(symbol):
    try:
        r = requests.get(f"{API_BASE}/sentiment/{symbol}")
        return r.json() if r.status_code == 200 else None
    except:
        return None

def run_trade(symbol):
    try:
        r = requests.post(f"{API_BASE}/trade/{symbol}")
        return r.json() if r.status_code == 200 else None
    except:
        return None

st.title("🤖 NIFTY 50 Live Paper Trading Platform")
st.markdown("Agent autonomously trades using DQN model. Watch its performance below.")

# Sidebar
st.sidebar.header("Configuration")
symbols = fetch_symbols()
if not symbols:
    st.sidebar.error("Could not connect to backend.")
    st.stop()

selected_symbol = st.sidebar.selectbox("Select NIFTY 50 Stock", symbols)
st.sidebar.metric("Starting Capital", "₹10,00,000")

if st.sidebar.button("Run Today's Trade", type="primary"):
    with st.spinner(f"Agent is analyzing {selected_symbol} and deciding..."):
        trade_res = run_trade(selected_symbol)
        if trade_res:
            st.sidebar.success(f"Trade executed for {selected_symbol}")
        else:
            st.sidebar.error("Trade failed.")

# Main app
portfolio = fetch_portfolio(selected_symbol)
sentiment = fetch_sentiment(selected_symbol)

if portfolio and portfolio.get("history"):
    history = portfolio["history"]
    latest = history[-1]
    
    st.header(f"Section 1: Today's Decision ({selected_symbol})")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Action
    action = latest["action"]
    action_color = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "🟡"
    col1.metric("Today's Action", f"{action_color} {action}")
    
    # Position
    pos = "IN MARKET" if latest["position"] == 1 else "SITTING OUT"
    col2.metric("Current Position", pos)
    
    # Return
    ret = latest["daily_return_pct"]
    col3.metric("Today's Return", f"{ret}%", delta=ret)
    
    # Sentiment Gauge
    score = latest["sentiment_score"]
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Sentiment Score"},
        gauge = {'axis': {'range': [-1, 1]},
                 'bar': {'color': "darkblue"},
                 'steps' : [
                     {'range': [-1, -0.2], 'color': "pink"},
                     {'range': [-0.2, 0.2], 'color': "lightgray"},
                     {'range': [0.2, 1], 'color': "lightgreen"}]}
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
    col4.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.header("Section 2: Portfolio Performance")
    
    df = pd.DataFrame(history)
    df["date"] = pd.to_datetime(df["date"])
    
    # Ensure starting at 1,000,000
    df_chart = df[["date", "portfolio_value", "bah_value"]].set_index("date")
    
    st.line_chart(df_chart, use_container_width=True)
    
    # Key Metrics
    trades_made = len(df[df["action"] != "HOLD"])
    # Win rate: % of days where position=1 and daily_return > 0
    trading_days = df[df["position"] == 1]
    win_days = trading_days[trading_days["daily_return_pct"] > 0]
    win_rate = (len(win_days) / len(trading_days) * 100) if len(trading_days) > 0 else 0.0
    
    c1, c2, c3, c4 = st.columns(4)
    total_ret = portfolio["total_return_pct"]
    bah_ret = ((latest["bah_value"] / portfolio["initial_value"]) - 1) * 100
    
    c1.metric("Total Return", f"{total_ret}%")
    c2.metric("vs Buy-and-Hold", f"{round(total_ret - bah_ret, 2)}%")
    c3.metric("Trades Made", trades_made)
    c4.metric("Win Rate", f"{round(win_rate, 1)}%")
    
else:
    st.info("No trading history for this stock. Click 'Run Today's Trade' to start.")

st.divider()

st.header("Section 3: Sentiment Panel")
if sentiment:
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Today's Score", round(sentiment["sentiment_score"], 2))
    sc2.metric("Magnitude", round(sentiment["sentiment_magnitude"], 2))
    sc3.metric("Article Count", sentiment["article_count"])
    
    st.subheader("Today's Headlines")
    if sentiment["headlines"]:
        for h in sentiment["headlines"]:
            st.markdown(f"- {h}")
    else:
        st.write("No headlines found.")
        
    if portfolio and portfolio.get("history"):
        st.subheader("7-Day Sentiment Trend")
        df_hist = pd.DataFrame(portfolio["history"]).tail(7)
        st.line_chart(df_hist.set_index("date")["sentiment_score"], height=150)
else:
    st.write("No sentiment data available.")
