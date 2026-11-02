import numpy as np, pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def max_drawdown(c): return float((c/c.cummax() - 1).min())

def _f(o):
    if isinstance(o, dict): return {k: _f(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_f(v) for v in o]
    if isinstance(o, (pd.Series, np.ndarray)): return _f(list(o))
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o

def eda(df, ann=365):
    c = df["close"]; r = c.pct_change().dropna()
    def desc(s):
        if len(s) == 0:
            return {k: None for k in ("mean", "median", "min", "max", "range", "variance", "std", "q1", "q3", "iqr", "skew", "kurtosis")}
        q1, q3 = s.quantile(.25), s.quantile(.75)
        return {"mean": s.mean(), "median": s.median(), "min": s.min(), "max": s.max(), "range": s.max()-s.min(),
                "variance": s.var() if len(s) > 1 else None, "std": s.std() if len(s) > 1 else None,
                "q1": q1, "q3": q3, "iqr": q3-q1,
                "skew": stats.skew(s) if len(s) >= 3 else None,
                "kurtosis": stats.kurtosis(s) if len(s) >= 4 else None}
    k = stats.kurtosis(r) if len(r) >= 4 else None
    period_move = c.iloc[-1]/c.iloc[0]-1 if len(c) > 1 else None
    obs = []
    obs.append(f"Price moved {period_move:+.1%} over the period; the worst peak-to-trough fall was {max_drawdown(c):.1%}." if period_move is not None else "Only one closing-price observation is available; a period return cannot be calculated.")
    if k is None:
        obs.append("There are too few daily returns to describe the return distribution's tail behaviour.")
    else:
        obs.append(f"Daily-return excess kurtosis is {k:.2f}" + (": fat tails, so extreme days are more common than a normal distribution implies." if k > 1 else "."))
    obs.append(f"{(r > 0).mean():.0%} of days were up days; the largest one-day move was {r.abs().max():.1%}." if len(r) else "No daily return observations are available in this selected window.")
    return _f({"kpi": {"latest_close": c.iloc[-1], "latest_date": str(df.date.iloc[-1].date()),
                       "period_return": period_move, "high": c.max(), "low": c.min(),
                       "avg_volume": df.volume.mean(), "volatility_annualized": r.std()*np.sqrt(ann) if len(r) > 1 else None,
                       "annualization": f"std(daily simple return) x sqrt({ann}); a convention, since crypto trades every day",
                       "max_drawdown": max_drawdown(c)},
               "price_stats": desc(c), "return_stats": desc(r), "observations": obs})

def series(df, win=20):
    r = df.close.pct_change()
    return _f({"date": [str(d.date()) for d in df.date], "close": df.close, "open": df.open, "high": df.high, "low": df.low, "volume": df.volume, "ret": r,
               "sma": df.close.rolling(win).mean(), "roll_vol": r.rolling(win).std()*np.sqrt(365)})

def compare(frames):
    cl = pd.concat({k: v.set_index("date")["close"] for k, v in frames.items()}, axis=1, join="inner").dropna()
    if len(cl) < 10: raise ValueError("Fewer than 10 overlapping dates between the selected assets.")
    r = cl.pct_change().dropna(); norm = cl/cl.iloc[0]*100
    perf = {k: {"return": cl[k].iloc[-1]/cl[k].iloc[0]-1, "vol_ann": r[k].std()*np.sqrt(365), "max_drawdown": max_drawdown(cl[k]),
                "sharpe_like": (r[k].mean()/r[k].std()*np.sqrt(365)) if r[k].std() else None} for k in cl}
    return _f({"dates": [str(d.date()) for d in cl.index], "normalized": {k: norm[k] for k in norm}, "performance": perf,
               "corr": {"labels": list(r.columns), "matrix": r.corr().values.tolist()}, "overlap_rows": len(cl),
               "note": "Correlation is not causation; past performance does not predict future performance. sharpe_like assumes a 0% risk-free rate."})

def anomalies(df, method="zscore", threshold=3.0, var="ret"):
    if var not in ("ret", "volume_chg"): raise ValueError("var must be ret or volume_chg")
    s = df.close.pct_change() if var == "ret" else df.volume.pct_change().replace([np.inf, -np.inf], np.nan)
    ok = s.dropna()
    if threshold <= 0: raise ValueError("threshold must be positive")
    if len(ok) < 30: raise ValueError("Need at least 30 observations for anomaly detection.")
    if ok.std() == 0: raise ValueError("Selected variable is constant; nothing to detect.")
    if method == "zscore": sc = (ok - ok.mean())/ok.std(); flag = sc.abs() > threshold
    elif method == "modified_z":
        mad = (ok - ok.median()).abs().median()
        if mad == 0: raise ValueError("MAD is zero for this variable; use another method.")
        sc = 0.6745*(ok - ok.median())/mad; flag = sc.abs() > threshold
    elif method == "iqr":
        q1, q3 = ok.quantile(.25), ok.quantile(.75); i = q3 - q1
        if i == 0: raise ValueError("IQR is zero; use another method.")
        flag = (ok < q1 - threshold*i) | (ok > q3 + threshold*i); sc = (ok - ok.median())/i
    elif method == "isolation_forest":
        m = IsolationForest(contamination=min(0.5, max(0.001, threshold/100)), random_state=42).fit(ok.to_frame())
        flag = pd.Series(m.predict(ok.to_frame()) == -1, index=ok.index); sc = pd.Series(-m.score_samples(ok.to_frame()), index=ok.index)
    else: raise ValueError("method must be zscore, modified_z, iqr or isolation_forest")
    hits = [{"date": str(df.date[i].date()), "value": float(ok[i]), "score": float(sc[i]), "method": method} for i in ok.index[flag]]
    return _f({"count": len(hits), "n": len(ok), "anomalies": hits, "var": var, "series": {"date": [str(df.date[i].date()) for i in ok.index], "values": list(ok.values)},
               "note": "Statistically unusual is not the same as erroneous: extreme days can be real market events. Nothing is removed. For isolation_forest the threshold is the contamination percent."})

def clusters(df, k=3):
    r = df.close.pct_change()
    X = pd.DataFrame({"ret": r, "vol10": r.rolling(10).std(), "volchg": df.volume.pct_change().replace([np.inf, -np.inf], np.nan), "mom10": df.close.pct_change(10)})
    ok = X.dropna()
    if len(ok) < k*5: raise ValueError("Not enough data for this many clusters.")
    lab = KMeans(k, n_init=10, random_state=42).fit_predict(StandardScaler().fit_transform(ok))
    prof = ok.assign(cluster=lab).groupby("cluster").agg(n=("ret", "size"), avg_ret=("ret", "mean"), avg_vol=("vol10", "mean"), avg_mom=("mom10", "mean"))
    return _f({"profile": prof.reset_index().to_dict("records"), "timeline": [{"date": str(df.date[i].date()), "cluster": int(c)} for i, c in zip(ok.index, lab)],
               "note": "Clusters are algorithmic groups of similar days, not proven economic regimes, and are not tested as predictors. Scaler fit on the full period (descriptive use only)."})
