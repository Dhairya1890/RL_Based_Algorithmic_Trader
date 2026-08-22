from .db import get_last_portfolio_state, save_portfolio_state

INITIAL_CAPITAL = 1000000.0
TRANSACTION_COST = 0.001

def apply_trade(symbol: str, date: str, action_idx: int, daily_return: float, sentiment_score: float, article_count: int):
    # Action: 0=BUY, 1=HOLD, 2=SELL
    actions = ["BUY", "HOLD", "SELL"]
    action_str = actions[action_idx]
    
    last_state = get_last_portfolio_state(symbol)
    if not last_state:
        # First day of trading
        portfolio_value = INITIAL_CAPITAL
        bah_value = INITIAL_CAPITAL
        position = 0
    else:
        portfolio_value = last_state["portfolio_value"]
        bah_value = last_state["bah_value"]
        position = last_state["position"]
        
    # Update Buy-and-Hold
    # Buy and hold means we are always in the market (position=1) with no transaction cost after initial entry
    if not last_state:
        bah_value = INITIAL_CAPITAL * (1 - TRANSACTION_COST)
    else:
        bah_value *= (1 + daily_return)

    # Calculate new position and transaction costs
    cost = 0.0
    if action_idx == 0 and position == 0:
        position = 1
        cost = TRANSACTION_COST
    elif action_idx == 2 and position == 1:
        position = 0
        cost = TRANSACTION_COST
        
    # Update portfolio value
    portfolio_value *= (1 + (daily_return * position) - cost)
    
    new_state = {
        "symbol": symbol,
        "date": date,
        "action": action_str,
        "position": position,
        "portfolio_value": round(portfolio_value, 2),
        "bah_value": round(bah_value, 2),
        "daily_return_pct": round(daily_return * 100, 4),
        "sentiment_score": sentiment_score,
        "article_count": article_count
    }
    
    save_portfolio_state(new_state)
    return new_state
