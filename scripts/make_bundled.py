"""Create real-data offline snapshots using Binance, then CoinGecko as fallback.
Usage: python scripts/make_bundled.py
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import data

data.BUNDLED.mkdir(parents=True, exist_ok=True)
symbols = {}
errors = {}
for symbol in ("BTC", "ETH", "SOL"):
    raw = None
    source = None
    for name, fetch in (("Binance public API", data._binance), ("CoinGecko API", data._coingecko)):
        try:
            raw = fetch(symbol)
            if raw is not None and not raw.empty and "close" in raw and raw["close"].notna().any():
                source = name
                break
            raw = None
            errors[f"{symbol}:{name}"] = "provider returned no usable close prices"
        except Exception as exc:
            errors[f"{symbol}:{name}"] = f"{type(exc).__name__}: {exc}"
    if raw is None:
        print(f"FAILED {symbol}: no provider returned usable data")
        continue
    raw = raw[data.COLS].copy()
    raw.to_csv(data.BUNDLED / f"{symbol}.csv", index=False)
    symbols[symbol] = {"rows": len(raw), "through": str(raw.date.max().date()), "source": source}
    print(f"OK {symbol}: {len(raw)} rows through {raw.date.max().date()} ({source})")
manifest = {"source": "Per-symbol provider listed below", "retrieved_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "symbols": symbols, "errors": errors}
(data.BUNDLED / "manifest.json").write_text(json.dumps(manifest, indent=2))
if len(symbols) < 3:
    print("WARNING: Some snapshots were not created. Check network access or upload a historical CSV for the missing symbols.")
if not symbols:
    raise SystemExit("No datasets could be downloaded from Binance or CoinGecko. No synthetic data was created.")
