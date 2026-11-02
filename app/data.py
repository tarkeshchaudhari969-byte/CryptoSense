"""Data acquisition, caching, validation and cleaning. Never fabricates data."""
import json, time, io, os, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE, UPLOADS = ROOT/"data"/"cache", ROOT/"data"/"uploads"
SYMBOLS = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}
COLS = ["date", "open", "high", "low", "close", "volume"]
BUNDLED = ROOT/"data"/"bundled"
ALLOWED = ["BTC", "ETH", "SOL"] + [x for x in os.environ.get("CRYPTO_SYMBOLS", "").upper().split(",") if re.fullmatch(r"[A-Z0-9]{2,10}", x) and x not in SYMBOLS]
MAX_UPLOAD = 5_000_000
_FAIL = {}   # symbol -> (time, error): skips repeated 15s timeouts while offline

def valid_symbol(s):
    s = str(s or "").strip().upper()
    if s not in ALLOWED: raise ValueError(f"Unsupported symbol {s[:12]!r}. Allowed: {', '.join(ALLOWED)}")
    return s

def _binance(sym, limit_days=2000):
    """Daily OHLCV from Binance public endpoints, trying alternate hosts (no API key)."""
    import requests
    # Some networks cannot reach one Binance hostname. Try the public market-data
    # hostname and documented alternatives; keep connect timeout short on weak networks.
    hosts = (
        "https://data-api.binance.vision",
        "https://api1.binance.com",
        "https://api-gcp.binance.com",
        "https://api.binance.com",
    )
    rows, end = [], None
    while len(rows) < limit_days:
        params = {"symbol": f"{sym}USDT", "interval": "1d", "limit": min(1000, limit_days - len(rows))}
        if end is not None:
            params["endTime"] = end
        last_error = None
        k = None
        for host in hosts:
            try:
                response = requests.get(f"{host}/api/v3/klines", params=params, timeout=(4, 12))
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list):
                    raise ValueError(f"Unexpected Binance response from {host}")
                k = payload
                break
            except Exception as exc:
                last_error = exc
        if k is None:
            raise RuntimeError(f"All Binance public endpoints failed for {sym}: {type(last_error).__name__}: {last_error}") from last_error
        if not k:
            break
        rows = k + rows
        end = k[0][0] - 1
        if len(k) < params["limit"]:
            break
    if not rows:
        raise ValueError(f"Binance returned no daily candles for {sym}.")
    df = pd.DataFrame(rows)
    out = pd.DataFrame({"date": pd.to_datetime(df[0], unit="ms").dt.normalize()})
    for name, i in [("open", 1), ("high", 2), ("low", 3), ("close", 4), ("volume", 5)]:
        out[name] = pd.to_numeric(df[i], errors="coerce")
    return out

def _coingecko(sym):
    """Fallback provider: close/volume only (free tier has no OHLC); missing fields stay NaN, not zero."""
    import requests
    r = requests.get(f"https://api.coingecko.com/api/v3/coins/{SYMBOLS[sym]}/market_chart",
                     params={"vs_currency": "usd", "days": 365, "interval": "daily"}, timeout=15)
    r.raise_for_status(); j = r.json()
    df = pd.DataFrame({"date": pd.to_datetime([p[0] for p in j["prices"]], unit="ms").normalize(),
                       "close": [p[1] for p in j["prices"]], "volume": [p[1] for p in j["total_volumes"]]})
    for c in ("open", "high", "low"): df[c] = np.nan
    return df[COLS]

def clean(df):
    """Returns (clean_df, report). Every decision is recorded in the report."""
    rep = {"original_rows": int(len(df))}
    df = df.copy(); df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    for c in COLS[1:]:
        if c not in df: df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    rep["bad_dates_removed"] = int(df["date"].isna().sum()); df = df.dropna(subset=["date"])
    rep["duplicates_removed"] = int(df.duplicated("date").sum())
    df = df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    rep["missing_values"] = {c: int(df[c].isna().sum()) for c in COLS[1:]}
    bad = (df["close"] <= 0) | df["close"].isna()
    rep["invalid_close_removed"] = int(bad.sum()); df = df[~bad].reset_index(drop=True)
    ohlc = df[["open", "high", "low", "close"]].notna().all(axis=1)
    viol = ohlc & ((df.high < df[["open", "close", "low"]].max(axis=1)) | (df.low > df[["open", "close", "high"]].min(axis=1)))
    rep["ohlc_violations_flagged"] = int(viol.sum()); df["ohlc_suspect"] = viol   # flagged, not altered
    if len(df):
        full = pd.date_range(df.date.min(), df.date.max(), freq="D")
        rep["missing_day_count"] = int(len(full) - len(df)); rep["missing_days"] = [str(d.date()) for d in full.difference(df.date)][:50]
        rep["has_ohlc"] = bool(df[["open", "high", "low"]].notna().all().all())
    rep["final_rows"] = int(len(df)); rep["final_cols"] = int(df.shape[1])
    rep["note"] = "Missing days are reported, never interpolated into the price series."
    return df, rep

def _window(raw, days):
    """Select the date window BEFORE cleaning so the quality report describes the selected range."""
    d = pd.to_datetime(raw["date"], errors="coerce", format="mixed")
    if d.notna().sum() == 0: raise ValueError("No valid dates in the data.")
    if days and days > 0: raw = raw[d >= d.max().normalize() - pd.Timedelta(days=days - 1)]
    return raw

def load(sym, days=365, refresh=False):
    """Priority: uploaded CSV > live fetch > cache > bundled snapshot. Always labels provenance."""
    sym = valid_symbol(sym); meta = {"symbol": sym, "currency": "USD"}
    up, cf, mf, bf = UPLOADS/f"{sym}.csv", CACHE/f"{sym}.csv", CACHE/f"{sym}.json", BUNDLED/f"{sym}.csv"
    raw = None
    if up.exists():
        raw = pd.read_csv(up); meta.update(source="User-uploaded CSV", retrieved_at=time.ctime(up.stat().st_mtime), fallback=False)
    else:
        stale = (not cf.exists()) or time.time() - cf.stat().st_mtime > 6*3600
        recent_fail = sym in _FAIL and time.time() - _FAIL[sym][0] < 120
        if (refresh or stale) and (refresh or not recent_fail):
            CACHE.mkdir(parents=True, exist_ok=True); errs = []
            for name, fn in (("Binance public API", _binance), ("CoinGecko API", _coingecko)):
                try:
                    raw = fn(sym); raw.to_csv(cf, index=False); _FAIL.pop(sym, None)
                    meta.update(source=name, retrieved_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), fallback=False)
                    mf.write_text(json.dumps(meta)); break
                except Exception as e: errs.append(f"{name}: {type(e).__name__}")
            if raw is None: _FAIL[sym] = (time.time(), "; ".join(errs))
        err = _FAIL.get(sym, (0, None))[1]
        if raw is None and cf.exists():
            raw = pd.read_csv(cf); old = json.loads(mf.read_text()) if mf.exists() else {}
            meta.update(source=old.get("source", "cache"), retrieved_at=old.get("retrieved_at", "unknown"), fallback=bool(err), last_error=err,
                        warning="Live data unavailable. Showing CACHED data from the retrieval time above." if err else "Cached copy (under 6 hours old).")
        if raw is None and bf.exists():
            man = json.loads((BUNDLED/"manifest.json").read_text()) if (BUNDLED/"manifest.json").exists() else {}
            raw = pd.read_csv(bf)
            meta.update(source=f"BUNDLED historical snapshot (offline fallback; originally {man.get('source', 'unknown source')})", retrieved_at=man.get("retrieved_at", "unknown"),
                        fallback=True, last_error=err, warning="Live data unavailable. Showing a BUNDLED historical snapshot that ends on its snapshot date. It is NOT current.")
        if raw is None:
            raise FileNotFoundError(f"No data for {sym}: live fetch failed ({err}) and there is no cache, bundled snapshot or upload. Connect to the internet, run scripts/make_bundled.py, or upload a CSV.")
    df, rep = clean(_window(raw, days))
    if df.empty: raise ValueError("No valid rows in the selected range.")
    rep["scope"] = f"{df.date.min().date()} to {df.date.max().date()} ({days or 'all'}-day window)"
    meta.update(start=str(df.date.min().date()), end=str(df.date.max().date()), rows=len(df))
    return df, rep, meta

def parse_upload(content: bytes):
    """Validate an uploaded CSV. Returns (raw_df_with_known_columns, cleaning_report). Content is only ever parsed as data."""
    if not content: raise ValueError("Empty file.")
    if len(content) > MAX_UPLOAD: raise ValueError("File too large (max 5 MB).")
    try: text = content.decode("utf-8-sig")
    except UnicodeDecodeError: raise ValueError("File must be UTF-8 text.")
    if "\x00" in text: raise ValueError("Binary content rejected.")
    try: df = pd.read_csv(io.StringIO(text), nrows=100_000)
    except Exception: raise ValueError("Could not parse as CSV.")
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not {"date", "close"} <= set(df.columns): raise ValueError("CSV needs at least 'date' and 'close' columns.")
    df = df[[c for c in COLS if c in df.columns]]
    cl, rep = clean(df)
    if rep["bad_dates_removed"] > 0.1*max(rep["original_rows"], 1): raise ValueError("More than 10% of dates could not be parsed.")
    if len(cl) < 30: raise ValueError(f"Only {len(cl)} valid rows after validation; need at least 30.")
    return df, rep
