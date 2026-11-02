"""Investor Risk Lab: transparent portfolio risk, cost and position-sizing analytics.
No trading, custody, wallet keys, or investment recommendations.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

SUPPORTED = {"BTC", "ETH", "SOL"}


def _finite(x):
    return float(x) if np.isfinite(x) else None


def _var_cvar(r: pd.Series, confidence: float):
    x = pd.to_numeric(r, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 20:
        return {"var": None, "cvar": None, "observations": len(x)}
    q = float(np.quantile(x, 1-confidence))
    tail = x[x <= q]
    return {"var": _finite(-q), "cvar": _finite(-tail.mean()) if len(tail) else None, "observations": len(x)}


def risk_lab(holdings, frames, fee_bps=10, slippage_bps=10, risk_budget=100, stop_loss_pct=8):
    """Compute empirical portfolio risk from aligned daily returns and manual quantities."""
    if not isinstance(holdings, list) or not 1 <= len(holdings) <= 6:
        raise ValueError("Provide between 1 and 6 holdings.")
    if not (0 <= fee_bps <= 500 and 0 <= slippage_bps <= 500):
        raise ValueError("Fee and slippage must each be between 0 and 500 basis points.")
    if not (0 < risk_budget <= 10_000_000 and 0.1 <= stop_loss_pct <= 100):
        raise ValueError("Risk budget must be positive and stop-loss distance must be 0.1–100%.")

    rows, close_series, value = [], {}, 0.0
    seen = set()
    for h in holdings:
        sym = str(h.get("symbol", "")).upper().strip()
        qty = float(h.get("quantity", 0))
        if sym not in SUPPORTED or sym in seen:
            raise ValueError(f"Unsupported or duplicate asset: {sym}. Supported assets: BTC, ETH, SOL.")
        if not math.isfinite(qty) or qty <= 0 or qty > 1e12:
            raise ValueError(f"Quantity for {sym} must be a positive finite number.")
        seen.add(sym)
        df = frames[sym].copy().sort_values("date")
        close = pd.Series(df.close.to_numpy(dtype=float), index=pd.to_datetime(df.date))
        close = close[~close.index.duplicated(keep="last")]
        if len(close) < 30:
            raise ValueError(f"At least 30 price observations are required for {sym}.")
        price = float(close.iloc[-1]); notional = price * qty
        value += notional
        close_series[sym] = close.pct_change().rename(sym)
        rows.append({"symbol": sym, "quantity": qty, "price": price, "value": notional})

    if value <= 0:
        raise ValueError("Portfolio value must be positive.")
    for r in rows:
        r["weight"] = r["value"] / value
    returns = pd.concat(close_series.values(), axis=1, join="inner").dropna()
    if len(returns) < 20:
        raise ValueError("Selected assets need at least 20 overlapping daily returns.")
    weights = pd.Series({r["symbol"]: r["weight"] for r in rows})
    port_ret = returns[weights.index].mul(weights, axis=1).sum(axis=1)
    daily_vol = float(port_ret.std(ddof=1))
    annual_vol = daily_vol * math.sqrt(365)
    peak = (1 + port_ret).cumprod().cummax()
    wealth = (1 + port_ret).cumprod()
    max_dd = float((wealth / peak - 1).min())
    var95, var99 = _var_cvar(port_ret, .95), _var_cvar(port_ret, .99)
    corr = returns[weights.index].corr().fillna(0)
    w = weights.to_numpy()
    variance = float(w @ returns[weights.index].cov().to_numpy() @ w)
    hhi = float(np.sum(w*w))
    entropy = -sum(float(x)*math.log(float(x)) for x in w if x > 0)
    entropy_norm = entropy / math.log(len(w)) if len(w) > 1 else 0.0
    off_diag = corr.to_numpy()[np.triu_indices(len(w), 1)]
    avg_corr = float(np.mean(off_diag)) if len(off_diag) else 1.0
    # A transparent heuristic, not a credit rating or investment signal.
    diversification = float(np.clip(100 * (0.55*entropy_norm + 0.45*(1-max(0, avg_corr))), 0, 100)) if len(w) > 1 else 0.0

    round_trip_rate = 2 * (fee_bps + slippage_bps) / 10_000
    est_round_trip_cost = value * round_trip_rate
    breakeven_move = round_trip_rate
    stop_fraction = stop_loss_pct / 100
    risk_per_unit_fraction = stop_fraction + round_trip_rate
    position_notional = risk_budget / risk_per_unit_fraction

    # Explicit hypothetical shocks. These are scenarios, not forecasts/probabilities.
    shocks = {
        "Broad risk-off": {"BTC": -.20, "ETH": -.28, "SOL": -.35},
        "Altcoin liquidity shock": {"BTC": -.12, "ETH": -.30, "SOL": -.45},
        "Severe drawdown": {"BTC": -.45, "ETH": -.55, "SOL": -.65},
        "BTC-led rally / altcoin lag": {"BTC": .15, "ETH": .02, "SOL": -.12},
    }
    scenarios = []
    for name, shock in shocks.items():
        impact = sum(r["value"] * shock[r["symbol"]] for r in rows)
        scenarios.append({"scenario": name, "pnl": impact, "return": impact/value, "portfolio_after": value+impact})

    return {
        "portfolio": {"value": value, "holdings": rows, "asset_count": len(rows), "as_of": str(max(pd.to_datetime(f.date).max() for f in frames.values()).date())},
        "risk": {"annualized_volatility": annual_vol, "daily_volatility": daily_vol, "max_drawdown": max_dd,
                 "var_95_1d": var95["var"], "cvar_95_1d": var95["cvar"], "var_99_1d": var99["var"], "cvar_99_1d": var99["cvar"],
                 "observations": len(returns), "correlation": {"labels": list(weights.index), "matrix": corr.to_numpy().tolist()},
                 "concentration_hhi": hhi, "effective_holdings": (1/hhi if hhi else None), "average_pairwise_correlation": avg_corr if len(w)>1 else None,
                 "diversification_score": diversification,
                 "method_note": "Historical, equal-date daily returns; portfolio weights held constant for the historical risk estimate. VaR/CVaR are empirical one-day loss estimates, not worst-case bounds."},
        "costs": {"fee_bps_each_side": fee_bps, "slippage_bps_each_side": slippage_bps, "estimated_round_trip_cost": est_round_trip_cost,
                  "round_trip_cost_pct": round_trip_rate, "breakeven_move_pct": breakeven_move,
                  "note": "Illustrative proportional fee/slippage assumptions. Actual venue, spread, market impact, taxes and funding can differ."},
        "scenarios": scenarios,
        "position_sizing": {"risk_budget": risk_budget, "stop_loss_distance_pct": stop_loss_pct, "estimated_round_trip_cost_pct": round_trip_rate,
                            "max_position_notional": position_notional, "note": "Arithmetic sizing illustration: risk budget divided by stop distance plus estimated round-trip friction. Not a recommendation; gaps and slippage can exceed assumptions."},
        "limitations": ["Historical risk can change abruptly and does not predict future returns.", "Only the supported assets and supplied holdings are included.", "Scenario shocks are hypothetical and not forecasts.", "No wallet connection, exchange execution, custody, or investment advice is provided."]
    }
