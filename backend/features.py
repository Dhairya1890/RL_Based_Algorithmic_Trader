import numpy as np
import pandas as pd
import yfinance as yf

def compute_rsi(series, window=14):
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def compute_engineered_features(df):
    """
    Compute all 15 engineered features for a single stock DataFrame.
    """
    df = df.copy()

    # 1. Log Return
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # 2-3. Rolling Volatility
    df['Vol_10D'] = df['Log_Return'].rolling(window=10).std()
    df['Vol_20D'] = df['Log_Return'].rolling(window=20).std()

    # 4-6. Distance from Simple Moving Averages
    for w in [10, 20, 50]:
        sma = df['Close'].rolling(window=w).mean()
        df[f'Dist_SMA_{w}'] = (df['Close'] - sma) / sma

    # 7. RSI
    df['RSI_14'] = compute_rsi(df['Close'], window=14)

    # 8-10. MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']

    # 11. Bollinger Band %B
    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    df['BB_Pband'] = (df['Close'] - lower_band) / (upper_band - lower_band)

    # 12. Average True Range
    high_low = df['High'] - df['Low']
    high_close_prev = (df['High'] - df['Close'].shift(1)).abs()
    low_close_prev = (df['Low'] - df['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df['ATR_14'] = true_range.rolling(window=14).mean()

    # 13. Volume Ratio
    vol_sma_20 = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio_20'] = df['Volume'] / vol_sma_20.replace(0, np.nan)

    # 14. VWAP Distance
    df['VWAP_Dist'] = (df['Close'] - df['VWAP']) / df['VWAP'].replace(0, np.nan)

    # 15. Deliverable Percentage
    if '%Deliverble' in df.columns:
        df['Deliverable_Pct'] = df['%Deliverble']
    else:
        df['Deliverable_Pct'] = df['Deliverable Volume'] / df['Volume'].replace(0, np.nan)

    return df

def fetch_and_prepare_features(symbol: str, return_days: int = 30):
    # Fetch enough historical data from yfinance
    # We need 'return_days' final rows. We also need 50 days of warmup for SMA_50. 
    # For backtesting (e.g. 60 days), we need 60 + 50 = 110 trading days. 6 months gives ~126 trading days.
    # 1 year gives ~252 trading days.
    yf_symbol = f"{symbol}.NS"
    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(period="1y")
    
    # Format columns as expected
    df['Prev Close'] = df['Close'].shift(1)
    df['Last'] = df['Close']
    df['VWAP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Turnover'] = df['Volume'] * df['VWAP']
    df['Trades'] = 0
    df['Deliverable Volume'] = 0
    df['Deliverable_Pct'] = 0

    df['VWAP'] = df['VWAP'].replace(0, np.nan)
    
    df = compute_engineered_features(df)
    
    df.dropna(subset=[
        'Log_Return', 'Vol_10D', 'Vol_20D', 'Dist_SMA_10', 'Dist_SMA_20',
        'Dist_SMA_50', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Diff',
        'BB_Pband', 'ATR_14', 'Volume_Ratio_20', 'VWAP_Dist', 'Deliverable_Pct'
    ], inplace=True)
    
    return df.tail(return_days).copy()

STATE_FEATURES = [
    # Raw (12)
    'Prev Close', 'Open', 'High', 'Low', 'Last', 'Close',
    'VWAP', 'Volume', 'Turnover', 'Trades',
    'Deliverable Volume', 'Deliverable_Pct',
    # Engineered (15)
    'Log_Return', 'Vol_10D', 'Vol_20D',
    'Dist_SMA_10', 'Dist_SMA_20', 'Dist_SMA_50',
    'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Diff',
    'BB_Pband', 'ATR_14', 'Volume_Ratio_20', 'VWAP_Dist',
    # Sentiment (4)
    'Sentiment_Score', 'Sentiment_Magnitude',
    'Article_Count', 'Sentiment_Rolling_3D',
]
