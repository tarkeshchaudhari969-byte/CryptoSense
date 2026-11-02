from fastapi import FastAPI, UploadFile, HTTPException, Body
from fastapi.responses import FileResponse, Response
import os
from . import service as S, data

UPLOADS_ENABLED = os.getenv("ENABLE_CSV_UPLOADS", "true").strip().lower() in {"1", "true", "yes", "on"}
app = FastAPI(
    title="CryptoSense AI",
    version="2.0.0",
    description="Cryptocurrency historical analytics, machine-learning research and portfolio risk intelligence.",
)

@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    return response

def _run(fn, *a, **k):
    try: r = fn(*a, **k)
    except S.ApiError as e: raise HTTPException(e.status, e.msg)
    if isinstance(r, S.Csv): return Response(r.text, media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{r.name}"'})
    if isinstance(r, S.Pdf): return Response(r.content, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{r.name}"', "Cache-Control": "no-store"})
    return r

@app.get("/api/data")
def api_data(symbol: str = "BTC", range: str = "1y", refresh: bool = False): return _run(S.get_data, symbol, range, refresh)

@app.post("/api/upload")
async def upload(symbol: str, file: UploadFile):
    if not UPLOADS_ENABLED:
        raise HTTPException(403, "CSV uploads are disabled on this public deployment. Enable them only after adding authenticated, per-user storage.")
    return _run(S.upload, symbol, await file.read(data.MAX_UPLOAD + 1))

@app.delete("/api/upload")
def remove_upload(symbol: str):
    if not UPLOADS_ENABLED:
        raise HTTPException(403, "CSV uploads are disabled on this public deployment.")
    return _run(S.remove_upload, symbol)

@app.get("/api/features")
def api_features(symbol: str = "BTC", range: str = "1y", fmt: str = "json"): return _run(S.get_features, symbol, range, fmt)

@app.get("/api/clean.csv")
def clean_csv(symbol: str = "BTC", range: str = "1y"): return _run(S.clean_csv, symbol, range)

@app.get("/api/train")
def train(symbol: str = "BTC", range: str = "max", task: str = "classification", model: str = "random_forest", test_frac: float = 0.25, horizon: int = 1, fmt: str = "json"):
    return _run(S.train, symbol, range, task, model, test_frac, horizon, fmt)

@app.get("/api/compare")
def compare(symbols: str = "BTC,ETH,SOL", range: str = "1y"): return _run(S.compare, symbols, range)

@app.get("/api/anomalies")
def anom(symbol: str = "BTC", range: str = "1y", method: str = "zscore", threshold: float = 3.0, var: str = "ret"): return _run(S.anomalies, symbol, range, method, threshold, var)

@app.get("/api/clusters")
def clus(symbol: str = "BTC", range: str = "1y", k: int = 3): return _run(S.clusters, symbol, range, k)

@app.get("/")
def index(): return FileResponse(data.ROOT/"static"/"index.html")


@app.post("/api/risk-lab")
def risk_lab(payload: dict = Body(...)): return _run(S.risk_lab, payload)

@app.get("/api/reports/{symbol}")
def crypto_report(symbol: str, period: str = "1y"): return _run(S.crypto_report, symbol, period)

@app.get("/api/config")
def client_config():
    return {"csv_uploads_enabled": UPLOADS_ENABLED, "reports_enabled": True, "supported_symbols": ["BTC", "ETH", "SOL"]}

@app.get("/health")
def health(): return {"status": "ok", "service": "CryptoSense AI", "version": "2.0.0"}
