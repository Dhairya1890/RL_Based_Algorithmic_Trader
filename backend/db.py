import sqlite3

DB_PATH = "portfolio.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS portfolio (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      symbol TEXT NOT NULL,
      date TEXT NOT NULL,
      action TEXT NOT NULL,           
      position INTEGER NOT NULL,      
      portfolio_value REAL NOT NULL,
      bah_value REAL NOT NULL,
      daily_return_pct REAL,
      sentiment_score REAL,
      article_count INTEGER,
      shares INTEGER DEFAULT 0,
      cash REAL DEFAULT 1000000.0,
      bah_shares INTEGER DEFAULT 0,
      bah_cash REAL DEFAULT 0.0,
      UNIQUE(symbol, date)
    );
    """)
    conn.commit()
    conn.close()

def get_portfolio_history(symbol: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM portfolio WHERE symbol = ? ORDER BY date ASC", (symbol,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_portfolio_history(symbol: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM portfolio WHERE symbol = ?", (symbol,))
    conn.commit()
    conn.close()

def get_last_portfolio_state(symbol: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM portfolio WHERE symbol = ? ORDER BY date DESC LIMIT 1", (symbol,))
    row = c.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def save_portfolio_state(state: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    INSERT OR REPLACE INTO portfolio 
    (symbol, date, action, position, portfolio_value, bah_value, daily_return_pct, sentiment_score, article_count, shares, cash, bah_shares, bah_cash)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        state["symbol"], state["date"], state["action"], state["position"], 
        state["portfolio_value"], state["bah_value"], state["daily_return_pct"],
        state["sentiment_score"], state["article_count"],
        state.get("shares", 0), state.get("cash", 0), state.get("bah_shares", 0), state.get("bah_cash", 0)
    ))
    conn.commit()
    conn.close()
