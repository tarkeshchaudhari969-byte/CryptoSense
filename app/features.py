"""Leakage-safe features. Feature at row t uses data up to and including day t only.
Targets at row t look at day t+1 and are never predictors."""
import numpy as np, pandas as pd

def rsi(close, n=14):
    d = close.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + up/dn.replace(0, np.nan))

def build_features(df, win=(5, 10, 20)):
    f = pd.DataFrame({"date": df["date"]}); c, v = df["close"], df["volume"]
    f["ret"] = c.pct_change(); f["logret"] = np.log(c).diff()
    for k in range(1, 6): f[f"ret_lag{k}"] = f["ret"].shift(k)
    for w in win:
        f[f"sma_ratio_{w}"] = c/c.rolling(w).mean() - 1
        f[f"ema_ratio_{w}"] = c/c.ewm(span=w, adjust=False).mean() - 1
        f[f"vol_{w}"] = f["ret"].rolling(w).std(); f[f"mom_{w}"] = c.pct_change(w)
    f["rsi14"] = rsi(c)
    e12, e26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    f["macd"] = (e12 - e26)/c; f["macd_signal"] = f["macd"].ewm(span=9, adjust=False).mean()
    f["volume_chg"] = v.pct_change().replace([np.inf, -np.inf], np.nan); f["volume_ratio_20"] = v/v.rolling(20).mean()
    if df[["high", "low"]].notna().all().all(): f["hl_range"] = (df["high"] - df["low"])/c
    nxt = c.shift(-1)
    f["target_ret"] = nxt/c - 1; f["target_up"] = np.where(nxt.isna(), np.nan, (nxt > c).astype(float))
    return f

def feature_cols(f): return [c for c in f.columns if c not in ("date", "target_ret", "target_up")]
