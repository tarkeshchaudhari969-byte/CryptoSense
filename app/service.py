"""Endpoint logic as plain functions (testable without a web server). main.py only wraps these."""
from collections import namedtuple
import pandas as pd
from . import data, features, ml, analysis, investor

Csv = namedtuple("Csv", "text name")
Pdf = namedtuple("Pdf", "content name")
RANGES = {
    "1d": 1, "1w": 7, "1m": 30, "3m": 90, "6m": 180,
    "1y": 365, "5y": 1825, "all": 0,
    # Backwards-compatible aliases used by earlier builds and saved links.
    "7d": 7, "30d": 30, "90d": 90, "180d": 180, "max": 0,
}
REPORT_PERIODS = {
    "1d": (1, "1 Day"), "1w": (7, "1 Week"), "1m": (30, "1 Month"),
    "3m": (90, "3 Months"), "6m": (180, "6 Months"),
    "1y": (365, "1 Year"), "5y": (1825, "5 Years"), "all": (0, "All Available History"),
}

class ApiError(Exception):
    def __init__(self, status, msg): super().__init__(msg); self.status, self.msg = status, msg

def _sym(s):
    try: return data.valid_symbol(s)
    except ValueError as e: raise ApiError(422, str(e))

def _get(sym, rng, refresh=False):
    if rng not in RANGES: raise ApiError(422, f"range must be one of {list(RANGES)}")
    sym = _sym(sym)
    try: return data.load(sym, RANGES[rng], refresh)
    except FileNotFoundError as e: raise ApiError(503, str(e))
    except ValueError as e: raise ApiError(422, str(e))
    except Exception as e: raise ApiError(500, f"Unexpected {type(e).__name__} while loading data.")

def _guard(fn):
    try: return fn()
    except ValueError as e: raise ApiError(422, str(e))

def get_data(symbol, range, refresh=False):
    df, rep, meta = _get(symbol, range, refresh)
    return {"meta": meta, "quality": rep, **_guard(lambda: {"eda": analysis.eda(df), "series": analysis.series(df)})}

def upload(symbol, content):
    sym = _sym(symbol)
    try: raw, rep = data.parse_upload(content)
    except ValueError as e: raise ApiError(400, f"Invalid CSV: {e}")
    data.UPLOADS.mkdir(parents=True, exist_ok=True); tmp = data.UPLOADS/f"{sym}.csv.tmp"
    raw.to_csv(tmp, index=False); tmp.replace(data.UPLOADS/f"{sym}.csv")
    return {"ok": True, "rows": len(raw), "valid_rows": rep["final_rows"]}

def remove_upload(symbol):
    (data.UPLOADS/f"{_sym(symbol)}.csv").unlink(missing_ok=True); return {"ok": True}

def get_features(symbol, range, fmt="json"):
    df = _get(symbol, range)[0]; f = features.build_features(df)
    if fmt == "csv": return Csv(f.to_csv(index=False), f"{symbol}_features.csv")
    t = f.tail(30).copy(); t["date"] = t.date.astype(str)
    return {"columns": list(f.columns), "rows": t.round(5).astype(object).where(t.notna(), None).to_dict("records")}

def clean_csv(symbol, range): return Csv(_get(symbol, range)[0].to_csv(index=False), f"{symbol}_clean.csv")

def train(symbol, range="max", task="classification", model="random_forest", test_frac=0.25, horizon=1, fmt="json"):
    if not 0.1 <= test_frac <= 0.5: raise ApiError(422, "test_frac must be between 0.1 and 0.5")
    if not 1 <= horizon <= 30: raise ApiError(422, "horizon must be 1-30 days")
    df, _, meta = _get(symbol, range); f = features.build_features(df)
    res = _guard(lambda: ml.run(df, f, task, model, test_frac, horizon)); res["data_source"] = meta
    return Csv(pd.DataFrame(res["predictions"]).to_csv(index=False), f"{symbol}_predictions.csv") if fmt == "csv" else res

def compare(symbols, range):
    syms = list(dict.fromkeys(_sym(s) for s in symbols.split(",")))
    if not 2 <= len(syms) <= 6: raise ApiError(422, "Choose between 2 and 6 different symbols.")
    return _guard(lambda: analysis.compare({s: _get(s, range)[0] for s in syms}))

def anomalies(symbol, range, method="zscore", threshold=3.0, var="ret"):
    return _guard(lambda: analysis.anomalies(_get(symbol, range)[0], method, threshold, var))

def clusters(symbol, range, k=3):
    if not 2 <= k <= 8: raise ApiError(422, "k must be between 2 and 8")
    return _guard(lambda: analysis.clusters(_get(symbol, range)[0], k))


def risk_lab(payload):
    if not isinstance(payload, dict): raise ApiError(422, "Request body must be a JSON object.")
    holdings = payload.get("holdings", [])
    if not isinstance(holdings, list): raise ApiError(422, "holdings must be a list.")
    symbols = list(dict.fromkeys(str(h.get("symbol", "")).upper().strip() for h in holdings if isinstance(h, dict)))
    if len(symbols) != len(holdings): raise ApiError(422, "Each holding must be an object with symbol and quantity.")
    frames = {}
    for sym in symbols:
        if sym not in ("BTC", "ETH", "SOL"): raise ApiError(422, f"Unsupported asset {sym}; supported: BTC, ETH, SOL.")
        frames[sym] = _get(sym, "max")[0]
    try:
        return investor.risk_lab(holdings, frames, float(payload.get("fee_bps", 10)),
                                 float(payload.get("slippage_bps", 10)), float(payload.get("risk_budget", 100)),
                                 float(payload.get("stop_loss_pct", 8)))
    except (ValueError, TypeError, OverflowError) as e: raise ApiError(422, str(e))


def crypto_report(symbol, period):
    """Return a downloadable PDF generated from the selected asset's available history."""
    from .reports import generate_report_pdf
    sym = _sym(symbol)
    if period not in REPORT_PERIODS:
        raise ApiError(422, f"period must be one of {list(REPORT_PERIODS)}")
    days, label = REPORT_PERIODS[period]
    try:
        df, quality, meta = data.load(sym, days)
        pdf = generate_report_pdf(df, quality, meta, sym, label)
        return Pdf(pdf, f"CryptoSense_{sym}_{period}_Research_Report.pdf")
    except FileNotFoundError as e:
        raise ApiError(503, str(e))
    except ValueError as e:
        raise ApiError(422, str(e))
    except Exception as e:
        raise ApiError(500, f"Could not generate report ({type(e).__name__}).")
