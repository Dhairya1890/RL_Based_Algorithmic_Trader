"""
NIFTY 50 RL Trading Platform.
Main Streamlit application entry point.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Utility imports
from utils.data_loader import load_master_df, load_models
from utils.metrics import sharpe_ratio, max_drawdown, annual_return, win_rate
from utils.sentiment import fetch_today_sentiment
from utils.live_price import fetch_live_price
from utils.backtester import run_strategy
from components.metric_card import metric_card
from components.headline_card import headline_card

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
st.set_page_config(page_title="NIFTY 50 RL Trading", layout="wide")

COLORS = {
    "buy":       "#00C853",   # green
    "sell":      "#D32F2F",   # red
    "hold":      "#F9A825",   # amber
    "run_a":     "#1565C0",   # blue (DQN no sentiment)
    "run_b":     "#00897B",   # teal (DQN with sentiment — our contribution)
    "xgb":       "#6A1B9A",   # purple (XGBoost)
    "buyhold":   "#546E7A",   # gray (Buy and Hold)
    "positive":  "#00C853",
    "negative":  "#D32F2F",
    "neutral":   "#F9A825",
}

ALL_TICKERS = [
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

# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
def page_overview():
    st.title("NIFTY 50 RL Trading Platform")
    st.markdown("Evaluating Reinforcement Learning (DQN) with Sentiment Analysis against Baseline Strategies.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        ticker = st.selectbox("Select NIFTY 50 Stock", ALL_TICKERS, index=37) # Default RELIANCE
    with col2:
        capital = st.number_input("Starting Capital (₹)", value=1000000, step=100000)

    # Load Data & Models
    with st.spinner("Loading data..."):
        master_df = load_master_df()
        models = load_models()
        
    if master_df.empty:
        st.info("Awaiting Data Pipelines. Please run the feature engineering scripts to generate data/processed/ CSV files.")
        return

    # Filter to ticker
    df_ticker = master_df[master_df['ticker'] == ticker]
    if df_ticker.empty:
        st.warning(f"No data available for {ticker} in the dataset.")
        return

    # Run backtests for all strategies to generate overview numbers
    # To keep it fast, we only run the last 1 year by default on overview
    df_recent = df_ticker.tail(252)
    
    res_bh = run_strategy(None, df_recent, ticker, capital, "buyhold")
    res_xgb = run_strategy(models["xgb"], df_recent, ticker, capital, "xgb")
    res_run_a = run_strategy(models["dqn_run_a"], df_recent, ticker, capital, "dqn_run_a")
    res_run_b = run_strategy(models["dqn_run_b"], df_recent, ticker, capital, "dqn_run_b")
    
    # 4 Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    
    # Action (Run B latest)
    latest_action = res_run_b['action'].iloc[-1]
    action_color = COLORS.get(latest_action.lower(), "#FFF")
    with c1:
        st.markdown(f"**Today's Action**<br><span style='color:{action_color}; font-size:1.5rem; font-weight:bold;'>{latest_action}</span>", unsafe_allow_html=True)
        st.caption("DQN Run B (With Sentiment) recommendation.")
        
    # Sharpe Ratio
    sharpe_a = sharpe_ratio(res_run_a['daily_return'])
    sharpe_b = sharpe_ratio(res_run_b['daily_return'])
    with c2:
        metric_card("Sharpe Ratio (Run B)", f"{sharpe_b:.2f}", "Measures return per unit of risk. Above 1.0 is good. Shown for Run B vs Run A.", delta=f"{sharpe_b - sharpe_a:+.2f}", good_direction="up")
        
    # Max Drawdown
    dd_b = max_drawdown(res_run_b['portfolio_value'])
    dd_a = max_drawdown(res_run_a['portfolio_value'])
    with c3:
        metric_card("Max Drawdown (Run B)", f"{dd_b*100:.1f}%", "Largest peak-to-trough portfolio decline. Closer to 0% is better.", delta=f"{(dd_b - dd_a)*100:+.1f}%", good_direction="up") # Note: smaller negative is better

    # Sentiment Score
    latest_sentiment = df_recent['sentiment_score'].iloc[-1]
    with c4:
        metric_card("Sentiment Score", f"{latest_sentiment:.2f}", "Today's news sentiment. -1.0 = very negative, +1.0 = very positive, 0 = neutral or no news.", delta_color="off")

    st.divider()

    # Equity Curve
    st.subheader("1-Year Equity Curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res_bh['date'], y=res_bh['portfolio_value'], name="Buy and Hold", line=dict(color=COLORS['buyhold'])))
    fig.add_trace(go.Scatter(x=res_xgb['date'], y=res_xgb['portfolio_value'], name="XGBoost Baseline", line=dict(color=COLORS['xgb'])))
    fig.add_trace(go.Scatter(x=res_run_a['date'], y=res_run_a['portfolio_value'], name="DQN Run A (No Sent)", line=dict(color=COLORS['run_a'])))
    fig.add_trace(go.Scatter(x=res_run_b['date'], y=res_run_b['portfolio_value'], name="DQN Run B (With Sent)", line=dict(color=COLORS['run_b'], width=3)))
    
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                      hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    # Comparison Table
    st.subheader("Strategy Summary")
    summary_data = []
    for name, res in [("Buy and Hold", res_bh), ("XGBoost", res_xgb), ("DQN Run A", res_run_a), ("DQN Run B", res_run_b)]:
        summary_data.append({
            "Strategy": name,
            "Sharpe Ratio": round(sharpe_ratio(res['daily_return']), 2),
            "Max Drawdown (%)": round(max_drawdown(res['portfolio_value']) * 100, 2),
            "Annual Return (%)": round(annual_return(res['portfolio_value']) * 100, 2),
            "Win Rate (%)": round(win_rate(res['daily_return']) * 100, 2)
        })
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary.style.highlight_max(subset=["Sharpe Ratio", "Annual Return (%)", "Win Rate (%)"], color="rgba(0, 200, 83, 0.2)")
                 .highlight_max(subset=["Max Drawdown (%)"], color="rgba(0, 200, 83, 0.2)"), # Max drawdown is a negative number, so max is best
                 use_container_width=True)

# ---------------------------------------------------------------------------
# Page: Live Trading
# ---------------------------------------------------------------------------
def page_live_trading():
    st.title("Live Trading")
    st.markdown("See what the agent would do today based on live market prices.")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        ticker = st.selectbox("Select NIFTY 50 Stock", ALL_TICKERS, index=37, key="live_ticker")
    with col2:
        model_choice = st.radio("Active Model", ["Run B (With Sentiment)", "Run A (Price Only)"], horizontal=True)
        
    master_df = load_master_df()
    models = load_models()
    
    if st.button("Run Today's Trade", type="primary"):
        with st.spinner(f"Fetching live data for {ticker}..."):
            live_data = fetch_live_price(ticker)
            if "error" in live_data:
                st.error(f"Failed to fetch live price: {live_data['error']}")
                return
                
            # Perform prediction
            st.success("Analysis complete!")
            
            # Decision Card
            st.markdown("### Decision")
            d1, d2 = st.columns(2)
            
            # Real decision pipeline
            model_key = "dqn_run_b" if "Run B" in model_choice else "dqn_run_a"
            selected_model = models.get(model_key)
            
            if selected_model is None:
                st.error(f"{model_choice} model is not loaded. Please ensure it exists in the models/ directory.")
                return
                
            df_ticker = master_df[master_df['ticker'] == ticker]
            if df_ticker.empty:
                st.error("No historical feature data available to combine with live price for prediction.")
                return
                
            last_row = df_ticker.iloc[-1]
            
            if model_key == 'dqn_run_b':
                cols = ['open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD', 'returns', 'volatility', 
                        'sentiment_score', 'magnitude', 'article_count', 'rolling_3d_avg', 'sentiment_available']
            else:
                cols = ['open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD', 'returns', 'volatility']
                
            features = last_row[cols].fillna(0).values.astype(np.float32)
            
            try:
                action_idx, _ = selected_model.predict(features, deterministic=True)
                action = "BUY" if action_idx == 0 else "SELL" if action_idx == 2 else "HOLD"
            except Exception as e:
                st.error(f"Model prediction failed (likely due to observation shape mismatch): {e}")
                return
                
            action_color = COLORS.get(action.lower(), "#FFF")
            
            with d1:
                st.markdown(f"<div style='text-align: center; padding: 20px; border-radius: 10px; background-color: rgba(255,255,255,0.05); border: 2px solid {action_color};'>"
                            f"<h1 style='color: {action_color}; margin: 0;'>{action}</h1>"
                            f"<p style='color: #a0aab2; margin-top: 10px;'>Recommended Action ({model_choice})</p>"
                            f"</div>", unsafe_allow_html=True)
                            
            with d2:
                st.metric("Live Price (₹)", f"{live_data['price']:,.2f}", f"{live_data['change_pct']*100:+.2f}%")
                st.info("The DQN agent outputs one of three actions each day based on the current market state.")

# ---------------------------------------------------------------------------
# Page: Backtest & Compare
# ---------------------------------------------------------------------------
def page_backtest():
    st.title("Strategy Comparison — Ablation Study")
    st.markdown("Does news sentiment improve the DQN agent? Run A vs Run B answers this.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        ticker = st.selectbox("Select NIFTY 50 Stock", ALL_TICKERS, index=37, key="bt_ticker")
    with col2:
        capital = st.number_input("Starting Capital (₹)", value=1000000, step=100000, key="bt_cap")
        
    if "backtest_results" not in st.session_state:
        st.session_state.backtest_results = None
        
    if st.button("Run Full Backtest", type="primary"):
        with st.spinner(f"Running simulation for {ticker}..."):
            master_df = load_master_df()
            models = load_models()
            df_ticker = master_df[master_df['ticker'] == ticker]
            
            if not df_ticker.empty:
                res_bh = run_strategy(None, df_ticker, ticker, capital, "buyhold")
                res_xgb = run_strategy(models["xgb"], df_ticker, ticker, capital, "xgb")
                res_run_a = run_strategy(models["dqn_run_a"], df_ticker, ticker, capital, "dqn_run_a")
                res_run_b = run_strategy(models["dqn_run_b"], df_ticker, ticker, capital, "dqn_run_b")
                
                st.session_state.backtest_results = {
                    "bh": res_bh, "xgb": res_xgb, "run_a": res_run_a, "run_b": res_run_b
                }
            else:
                st.info("Awaiting Data Pipelines. Please ensure data is generated.")

    results = st.session_state.backtest_results
    if results:
        res_bh = results["bh"]
        res_xgb = results["xgb"]
        res_run_a = results["run_a"]
        res_run_b = results["run_b"]
        
        # Calculate metrics
        def get_metrics(res):
            return {
                "sharpe": sharpe_ratio(res['daily_return']),
                "dd": max_drawdown(res['portfolio_value']),
                "ret": annual_return(res['portfolio_value']),
                "win": win_rate(res['daily_return'])
            }
            
        m_bh = get_metrics(res_bh)
        m_xgb = get_metrics(res_xgb)
        m_a = get_metrics(res_run_a)
        m_b = get_metrics(res_run_b)
        
        # Strategy Cards
        st.subheader("Performance Metrics")
        sc1, sc2, sc3, sc4 = st.columns(4)
        
        def display_strat_card(col, title, color, metrics):
            with col:
                st.markdown(f"<div style='border-top: 4px solid {color}; padding-top: 10px;'>", unsafe_allow_html=True)
                st.markdown(f"#### {title}")
                metric_card("Sharpe Ratio", f"{metrics['sharpe']:.2f}", "Risk-adjusted return", delta_color="off")
                metric_card("Annual Return", f"{metrics['ret']*100:.1f}%", "Annualised return", delta_color="off")
                metric_card("Max Drawdown", f"{metrics['dd']*100:.1f}%", "Peak-to-trough drop", delta_color="off")
                metric_card("Win Rate", f"{metrics['win']*100:.1f}%", "Profitable days", delta_color="off")
                st.markdown("</div>", unsafe_allow_html=True)
                
        display_strat_card(sc1, "Buy & Hold", COLORS['buyhold'], m_bh)
        display_strat_card(sc2, "XGBoost", COLORS['xgb'], m_xgb)
        display_strat_card(sc3, "DQN Run A (Price)", COLORS['run_a'], m_a)
        display_strat_card(sc4, "DQN Run B (Sent)", COLORS['run_b'], m_b)
        
        # Key Finding
        pct_improvement = ((m_b['sharpe'] - m_a['sharpe']) / abs(m_a['sharpe'])) * 100 if m_a['sharpe'] != 0 else 0
        st.info(f"**Key Finding:** DQN Run B (with sentiment) achieved a Sharpe Ratio of {m_b['sharpe']:.2f} vs {m_a['sharpe']:.2f} for Run A, an improvement of {pct_improvement:.1f}%.")
        
        # Chart
        st.subheader("Historical Equity Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res_bh['date'], y=res_bh['portfolio_value'], name="Buy and Hold", line=dict(color=COLORS['buyhold'])))
        fig.add_trace(go.Scatter(x=res_xgb['date'], y=res_xgb['portfolio_value'], name="XGBoost Baseline", line=dict(color=COLORS['xgb'])))
        fig.add_trace(go.Scatter(x=res_run_a['date'], y=res_run_a['portfolio_value'], name="DQN Run A (No Sent)", line=dict(color=COLORS['run_a'])))
        fig.add_trace(go.Scatter(x=res_run_b['date'], y=res_run_b['portfolio_value'], name="DQN Run B (With Sent)", line=dict(color=COLORS['run_b'], width=3)))
        
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                          hovermode="x unified", height=500, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Page: Sentiment Feed
# ---------------------------------------------------------------------------
def page_sentiment():
    st.title("Sentiment Feed")
    st.markdown("Explore today's news and historical sentiment trends.")
    
    ticker = st.selectbox("Select NIFTY 50 Stock", ALL_TICKERS, index=37, key="sent_ticker")
    
    # Fetch today's sentiment
    sentiment = fetch_today_sentiment(ticker)
    
    st.subheader("Today's Sentiment Summary")
    c1, c2, c3, c4 = st.columns(4)
    
    score = sentiment.get("sentiment_score", 0.0)
    score_color = "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral"
    
    with c1:
        st.markdown(f"**Score**<br><span style='color:{COLORS[score_color]}; font-size:2rem; font-weight:bold;'>{score:+.2f}</span>", unsafe_allow_html=True)
    with c2:
        metric_card("Magnitude", f"{sentiment.get('magnitude', 0.0):.2f}", "Confidence in the score. 0 = uncertain, 1 = highly confident.", delta_color="off")
    with c3:
        metric_card("Article Count", f"{sentiment.get('article_count', 0)}", "Number of news items scored today.", delta_color="off")
    with c4:
        if sentiment.get("sentiment_available", 1) == 0:
            st.info("No news data available for this period.")
        else:
            st.success("Data active and processed.")
            
    st.divider()
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("Today's Headlines")
        headlines = sentiment.get("headlines", [])
        if not headlines:
            st.info("No headlines found for today.")
        else:
            # Sort by magnitude descending
            headlines.sort(key=lambda x: x.get('magnitude', 0), reverse=True)
            for h in headlines:
                headline_card(h['text'], h['score'], h['magnitude'], h['source'], h['category'])
                
    with col_right:
        st.subheader("7-Day Sentiment Trend")
        # Load historical to show chart
        master_df = load_master_df()
        df_ticker = master_df[master_df['ticker'] == ticker].tail(7)
        if not df_ticker.empty and 'sentiment_score' in df_ticker.columns:
            fig = px.bar(df_ticker, x='date', y='sentiment_score', 
                         color='sentiment_score', 
                         color_continuous_scale=[COLORS['negative'], COLORS['neutral'], COLORS['positive']],
                         range_color=[-1, 1])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                              coloraxis_showscale=False, height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough historical data to show 7-day trend.")

# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
def page_about():
    st.title("About the Project")
    
    st.markdown("""
    ### Team K_Means_Kuch_Bhi
    
    * **Nitesh** — XGBoost supervised baseline & Feature Engineering
    * **Adithya** — Custom Gymnasium Trading Environment (`NiftyTradingEnv`)
    * **Harjap** — DQN Agent Training (Price vs. Sentiment Ablation)
    * **Dhairya** — News Sentiment Pipeline & Full-Stack Deployment
    
    ---
    
    ### How it works
    
    1. **Historical NIFTY 50 Data:** Cleaned end-of-day OHLCV data from the Kaggle dataset (2000-2021).
    2. **News Sentiment Pipeline:** BSE filings and news articles are scored by Google Gemini for price impact.
    3. **DQN Training:** A reinforcement learning agent learns to maximise risk-adjusted returns through trial and error.
       * **Run A:** Trained exclusively on technical indicators (RSI, MACD, etc.).
       * **Run B:** Trained on technical indicators + Gemini sentiment scores.
    4. **Ablation Study:** This platform evaluates whether the addition of NLP-derived sentiment (Run B) provides a statistically significant improvement in Sharpe Ratio and Max Drawdown over price-action alone (Run A).
    
    ### Technologies Used
    Python, Streamlit, Stable-Baselines3, Plotly, Pandas, yfinance, Google Gemini.
    """)

# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", [
    "Overview", 
    "Live Trading", 
    "Backtest & Compare", 
    "Sentiment Feed", 
    "About"
])

if page == "Overview":
    page_overview()
elif page == "Live Trading":
    page_live_trading()
elif page == "Backtest & Compare":
    page_backtest()
elif page == "Sentiment Feed":
    page_sentiment()
elif page == "About":
    page_about()
