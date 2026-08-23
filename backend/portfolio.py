from .db import get_last_portfolio_state, save_portfolio_state

INITIAL_CAPITAL = 1000000.0
TRANSACTION_COST = 0.001

def apply_trade(symbol: str, date: str, action_idx: int, daily_return: float, sentiment_score: float, article_count: int, current_price: float):
    # Action: 0=BUY, 1=HOLD, 2=SELL
    actions = ["BUY", "HOLD", "SELL"]
    action_str = actions[action_idx]
    
    last_state = get_last_portfolio_state(symbol)
    if not last_state:
        # First day of trading
        portfolio_value = INITIAL_CAPITAL
        bah_value = INITIAL_CAPITAL
        position = 0
        shares = 0
        cash = INITIAL_CAPITAL
        # Buy and hold initially buys the stock completely
        bah_shares = (INITIAL_CAPITAL * (1 - TRANSACTION_COST)) // current_price
        bah_cash = INITIAL_CAPITAL - (bah_shares * current_price) - (bah_shares * current_price * TRANSACTION_COST)
    else:
        portfolio_value = last_state["portfolio_value"]
        bah_value = last_state["bah_value"]
        position = last_state["position"]
        shares = last_state.get("shares", 0)
        cash = last_state.get("cash", portfolio_value)
        bah_shares = last_state.get("bah_shares", 0)
        bah_cash = last_state.get("bah_cash", 0)
        
    # Update Buy-and-Hold value based on actual holdings
    if last_state:
        bah_value = bah_cash + (bah_shares * current_price)

    # Calculate new position and transaction costs
    if action_idx == 0 and position == 0:
        position = 1
        # Buy as many shares as possible with available cash
        max_investment = cash * (1 - TRANSACTION_COST)
        shares_to_buy = max_investment // current_price
        cost = shares_to_buy * current_price * TRANSACTION_COST
        cash = cash - (shares_to_buy * current_price) - cost
        shares = shares_to_buy
    elif action_idx == 2 and position == 1:
        position = 0
        # Sell all shares
        cost = shares * current_price * TRANSACTION_COST
        cash = cash + (shares * current_price) - cost
        shares = 0
        
    # Update portfolio value
    portfolio_value = cash + (shares * current_price)
    
    new_state = {
        "symbol": symbol,
        "date": date,
        "action": action_str,
        "position": position,
        "portfolio_value": round(portfolio_value, 2),
        "bah_value": round(bah_value, 2),
        "daily_return_pct": round(daily_return * 100, 4),
        "sentiment_score": sentiment_score,
        "article_count": article_count,
        "shares": shares,
        "cash": round(cash, 2),
        "bah_shares": bah_shares,
        "bah_cash": round(bah_cash, 2)
    }
    
    save_portfolio_state(new_state)
    return new_state
