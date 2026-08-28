"""
Metrics calculation utilities for portfolio evaluation.
Includes functions for Sharpe Ratio, Max Drawdown, Annual Return, and Win Rate.
"""
import numpy as np
import pandas as pd

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.06) -> float:
    """Annualised Sharpe Ratio. risk_free_rate is annual (6% for India)."""
    excess = returns - risk_free_rate / 252
    if excess.std() == 0:
        return 0.0
    return (excess.mean() / excess.std()) * np.sqrt(252)

def max_drawdown(portfolio_values: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a negative percentage."""
    if len(portfolio_values) == 0:
        return 0.0
    peak = portfolio_values.cummax()
    drawdown = (portfolio_values - peak) / peak
    return drawdown.min()

def annual_return(portfolio_values: pd.Series) -> float:
    """Annualised return from first to last portfolio value."""
    n_days = len(portfolio_values)
    if n_days == 0:
        return 0.0
    total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
    return (1 + total_return) ** (252 / n_days) - 1

def win_rate(returns: pd.Series) -> float:
    """Fraction of trading days with positive return."""
    if len(returns) == 0:
        return 0.0
    return (returns > 0).mean()
