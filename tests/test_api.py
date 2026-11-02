"""HTTP-level tests. Skipped automatically if fastapi/httpx are not installed (they are in requirements.txt)."""
import unittest
try:
    from fastapi.testclient import TestClient
    from app.main import app; OK = True
except Exception: OK = False
try: from tests.test_service import Base, fixture
except ImportError: from test_service import Base, fixture

@unittest.skipUnless(OK, "fastapi/httpx not installed")
class API(Base):
    def setUp(self): super().setUp(); self.c = TestClient(app)
    def test_public_upload_guard(self):
        import app.main as main_module
        old = main_module.UPLOADS_ENABLED
        main_module.UPLOADS_ENABLED = False
        try:
            r = self.c.post("/api/upload?symbol=BTC", files={"file": ("blocked.csv", b"date,close\n2024-01-01,100\n")})
            self.assertEqual(r.status_code, 403)
            self.assertIn("per-user storage", r.json()["detail"])
        finally:
            main_module.UPLOADS_ENABLED = old

    def test_flow(self):
        self.assertEqual(self.c.get("/").status_code, 200); self.assertEqual(self.c.get("/api/data?symbol=../x").status_code, 422)
        self.assertEqual(self.c.get("/api/data?symbol=BTC").status_code, 503)
        csv = fixture(300).to_csv(index=False).encode()
        self.assertEqual(self.c.post("/api/upload?symbol=BTC", files={"file": ("a.csv", csv)}).status_code, 200)
        self.assertEqual(self.c.post("/api/upload?symbol=BTC", files={"file": ("a.csv", b"junk")}).status_code, 400)
        self.assertEqual(self.c.get("/api/data?symbol=BTC&range=30d").json()["meta"]["rows"], 30)
        r = self.c.get("/api/train?symbol=BTC&fmt=csv"); self.assertEqual((r.status_code, r.headers["content-type"].split(";")[0]), (200, "text/csv"))
        self.assertEqual(self.c.get("/api/train?symbol=BTC&horizon=99").status_code, 422)
        self.assertEqual(self.c.get("/api/anomalies?symbol=BTC&var=volume_chg").json()["var"], "volume_chg")
        self.assertEqual(self.c.get("/api/clean.csv?symbol=BTC").status_code, 200)
        report = self.c.get("/api/reports/BTC?period=1m")
        self.assertEqual(report.status_code, 200)
        self.assertEqual(report.headers["content-type"].split(";")[0], "application/pdf")
        self.assertTrue(report.content.startswith(b"%PDF-"))
        self.assertEqual(self.c.get("/api/reports/BTC?period=3y").status_code, 422)
        self.assertEqual(self.c.get("/api/config").json()["csv_uploads_enabled"], True)
        self.assertEqual(self.c.get("/health").json()["status"], "ok")
        for sym in ("ETH", "SOL"):
            self.assertEqual(self.c.post(f"/api/upload?symbol={sym}", files={"file": (f"{sym}.csv", csv)}).status_code, 200)
        risk = self.c.post("/api/risk-lab", json={"holdings":[{"symbol":"BTC","quantity":0.1},{"symbol":"ETH","quantity":1}],"risk_budget":100,"stop_loss_pct":8})
        self.assertEqual(risk.status_code, 200, risk.text)
        self.assertIn("scenarios", risk.json())
if __name__ == "__main__": unittest.main()
