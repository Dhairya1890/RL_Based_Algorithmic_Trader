import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests

import os

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

# --- API FUNCTIONS ---
@st.cache_data(ttl=60)
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

def run_backtest(symbol):
    try:
        r = requests.post(f"{API_BASE}/backtest/{symbol}?days=30")
        return r.json() if r.status_code == 200 else None
    except:
        return None

# Helpers for metrics
def calc_sharpe(returns, risk_free_rate=0.0):
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return np.sqrt(252) * (returns.mean() - risk_free_rate) / returns.std()

def calc_max_drawdown(portfolio_values):
    peak = portfolio_values.expanding(min_periods=1).max()
    drawdown = (portfolio_values / peak) - 1
    return drawdown.min()

# --- PAGE CONFIG ---
st.set_page_config(page_title="Algorithmic Trading Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
<style>
/* Custom sidebar padding and font sizing */
.css-1544g2n {
    padding-top: 2rem;
}
.sidebar-title {
    font-size: 12px;
    color: #888;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 5px;
    margin-top: 20px;
}
.metric-card {
    background-color: #1c1f26;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 15px;
    display: flex;
    flex-direction: column;
}
.metric-title {
    color: #aaa;
    font-size: 14px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
}
.metric-value {
    font-size: 32px;
    font-weight: bold;
    margin-bottom: 5px;
}
.metric-subtitle {
    font-size: 14px;
}
.text-green { color: #00d284; }
.text-red { color: #ff4b4b; }
.text-yellow { color: #ffd700; }
.text-gray { color: #888; }

/* Tooltip */
.tooltip-icon {
    position: relative;
    display: inline-block;
    cursor: pointer;
    color: #888;
    font-size: 14px;
    margin-left: 5px;
}
.tooltip-icon .tooltiptext {
    visibility: hidden;
    width: 200px;
    background-color: #333;
    color: #fff;
    text-align: left;
    border-radius: 6px;
    padding: 8px;
    font-size: 12px;
    font-weight: normal;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    margin-left: -100px;
    opacity: 0;
    transition: opacity 0.3s;
    border: 1px solid #555;
    box-shadow: 0px 4px 6px rgba(0,0,0,0.3);
}
.tooltip-icon .tooltiptext::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -5px;
    border-width: 5px;
    border-style: solid;
    border-color: #333 transparent transparent transparent;
}
.tooltip-icon:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}
.tooltip-icon:hover {
    color: #fff;
}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### Navigation")
    nav = st.radio("Go to", ["Live trading", "Backtest", "Sentiment feed", "About"], label_visibility="collapsed")
    
    symbols = fetch_symbols()
    if not symbols:
        st.error("Could not fetch symbols from API. Is backend running?")
        st.stop()
        
    st.markdown('<div class="sidebar-title">STOCK</div>', unsafe_allow_html=True)
    selected_symbol = st.selectbox("Stock", symbols, label_visibility="collapsed", index=symbols.index("RELIANCE") if "RELIANCE" in symbols else 0)
    
    portfolio = fetch_portfolio(selected_symbol)
    init_cap = portfolio.get("initial_value", 1000000) if portfolio else 1000000
    
    st.markdown('<div class="sidebar-title">CAPITAL</div>', unsafe_allow_html=True)
    st.markdown(f"## ₹{init_cap:,.0f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Run today's trade", use_container_width=True, type="secondary"):
        with st.spinner("Analyzing and executing..."):
            if run_trade(selected_symbol):
                st.success("Trade executed successfully!")
                st.rerun()
            else:
                st.error("Failed to execute trade.")
        
    if st.button("Run 30-day backtest", use_container_width=True, type="secondary"):
        with st.spinner("Running 30-day simulation..."):
            if run_backtest(selected_symbol):
                st.success("Backtest completed!")
                st.rerun()
            else:
                st.error("Failed to run backtest.")


# --- MAIN CONTENT ---
if nav == "Live trading":
    # Tabs
    tab1, tab2, tab3 = st.tabs(["trading", "compare", "feed"])
    
    if portfolio and portfolio.get("history") and len(portfolio["history"]) > 0:
        history = portfolio["history"]
        latest = history[-1]
        df = pd.DataFrame(history)
        df["date"] = pd.to_datetime(df["date"])
        
        sentiment = fetch_sentiment(selected_symbol)
        
        # --- Calculations ---
        df['daily_return_decimal'] = df['daily_return_pct'] / 100.0
        sharpe = calc_sharpe(df['daily_return_decimal'])
        max_dd = calc_max_drawdown(df['portfolio_value'])
        
        # --- TAB 1: TRADING ---
        with tab1:
            # --- Metrics Row ---
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                action = latest.get("action", "HOLD")
                action_color = "text-green" if action == "BUY" else "text-red" if action == "SELL" else "text-yellow"
                position_text = "IN MARKET" if latest.get('position') == 1 else "SITTING OUT"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Today's action <div class="tooltip-icon">ⓘ<span class="tooltiptext">The trading action predicted by the RL agent based on today's closing data.</span></div></div>
                    <div class="metric-value {action_color}">{action}</div>
                    <div class="metric-subtitle {action_color}">Position: {position_text}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Sharpe ratio <div class="tooltip-icon">ⓘ<span class="tooltiptext">Risk-adjusted return metric. Measures excess return per unit of risk.</span></div></div>
                    <div class="metric-value">{sharpe:.2f}</div>
                    <div class="metric-subtitle text-gray">Agent performance</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Max drawdown <div class="tooltip-icon">ⓘ<span class="tooltiptext">The maximum observed loss from a peak to a trough of the portfolio.</span></div></div>
                    <div class="metric-value">{max_dd*100:.1f}%</div>
                    <div class="metric-subtitle text-gray">Agent performance</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col4:
                sent_score = latest.get("sentiment_score", 0)
                sent_color = "text-green" if sent_score > 0 else "text-red" if sent_score < 0 else "text-gray"
                articles_count = sentiment.get("article_count", 0) if sentiment else 0
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Sentiment score <div class="tooltip-icon">ⓘ<span class="tooltiptext">Aggregated daily sentiment score from analyzed news headlines (-1.0 to 1.0).</span></div></div>
                    <div class="metric-value {sent_color}">{sent_score:+.2f}</div>
                    <div class="metric-subtitle text-gray">{articles_count} articles analyzed</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- Chart section ---
            st.markdown("### Equity curve — all strategies")
            
            df_chart = df[["date", "portfolio_value", "bah_value"]].copy()
            df_chart.rename(columns={"portfolio_value": "RL Agent", "bah_value": "Buy and hold"}, inplace=True)
            df_chart = df_chart.melt(id_vars="date", var_name="Strategy", value_name="Value")
            
            color_map = {
                "RL Agent": "#00d284",
                "Buy and hold": "#758a9e"
            }
            
            fig = px.line(df_chart, x="date", y="Value", color="Strategy", color_discrete_map=color_map)
            
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title=None,
                yaxis_title=None,
                xaxis=dict(showgrid=False, showticklabels=True, zeroline=False),
                yaxis=dict(showgrid=False, showticklabels=True, zeroline=False),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.1,
                    xanchor="center",
                    x=0.5,
                    title=None
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)

        # --- TAB 2: COMPARE ---
        with tab2:
            st.markdown("### Strategy comparison <span style='color:#888; font-size:16px; font-weight:normal;'>— ablation study</span>", unsafe_allow_html=True)
            
            # Calculations for Buy and Hold
            df['bah_return'] = df['bah_value'].pct_change().fillna(0)
            bah_sharpe = calc_sharpe(df['bah_return'])
            bah_max_dd = calc_max_drawdown(df['bah_value'])
            
            total_ret = portfolio.get("total_return_pct", 0)
            bah_total_ret = ((latest["bah_value"] / portfolio.get("initial_value", 1000000)) - 1) * 100
            
            trading_days = df[df["position"] == 1]
            win_days = trading_days[trading_days["daily_return_pct"] > 0]
            win_rate = (len(win_days) / len(trading_days) * 100) if len(trading_days) > 0 else 0.0
            
            bah_win_days = df[df['bah_return'] > 0]
            bah_win_rate = (len(bah_win_days) / len(df) * 100) if len(df) > 0 else 0.0
            
            data = {
                "Strategy": ["RL Agent", "Buy and hold"],
                "Sharpe": [f"{sharpe:.2f}", f"{bah_sharpe:.2f}"],
                "Max drawdown": [f"{max_dd*100:.1f}%", f"{bah_max_dd*100:.1f}%"],
                "Total return": [f"{total_ret:+.1f}%", f"{bah_total_ret:+.1f}%"],
                "Win rate": [f"{win_rate:.0f}%", f"{bah_win_rate:.0f}%"]
            }
            df_table = pd.DataFrame(data)
            st.dataframe(df_table, use_container_width=True, hide_index=True)

        # --- TAB 3: FEED ---
        with tab3:
            st.markdown("### Sentiment Feed")
            if sentiment:
                st.write(f"**Score:** {sentiment.get('sentiment_score', 0)}")
                st.write(f"**Magnitude:** {sentiment.get('sentiment_magnitude', 0)}")
                st.write(f"**Articles Processed:** {sentiment.get('article_count', 0)}")
                
                headlines = sentiment.get("headlines", [])
                if headlines:
                    st.write("**Top Headlines:**")
                    for h in headlines:
                        st.markdown(f"- {h}")
                else:
                    st.info("No headlines found for today.")
            else:
                st.warning("No sentiment data available for this stock currently.")

    else:
        with tab1:
            st.info(f"No trading history for {selected_symbol}. Click 'Run Today's Trade' or 'Run 30-Day Backtest' to start.")

elif nav == "Backtest":
    st.header("Backtest")
    st.info("Navigate to Live Trading to run a backtest and view the results.")
    
elif nav == "Sentiment feed":
    st.header("Sentiment Feed Overview")
    sentiment = fetch_sentiment(selected_symbol)
    if sentiment and sentiment.get("headlines"):
        st.write(f"**Stock:** {selected_symbol}")
        st.write(f"**Score:** {sentiment.get('sentiment_score', 0)}")
        for h in sentiment.get("headlines", []):
            st.markdown(f"- {h}")
    else:
        st.write("No headlines found.")
        
else:
    st.header(nav)
    st.write(f"Content for {nav} is under construction.")
