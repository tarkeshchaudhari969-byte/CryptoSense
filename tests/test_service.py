"""Offline tests of endpoint logic, upload safety, window-scoped quality report, fallbacks, exports and alignment.
Fixtures are SYNTHETIC test data only."""
import os, tempfile, unittest, time
from pathlib import Path
from unittest import mock
import numpy as np, pandas as pd
try: from tests.test_core import fixture
except ImportError: from test_core import fixture
from app import data, service as S

class Base(unittest.TestCase):
    def setUp(self):
        t = Path(tempfile.mkdtemp())
        for n in ("uploads", "cache", "bundled"): (t/n).mkdir()
        self.t = t; self.binance = mock.Mock(side_effect=OSError("offline")); self.gecko = mock.Mock(side_effect=OSError("offline"))
        for p in (mock.patch.object(data, "UPLOADS", t/"uploads"), mock.patch.object(data, "CACHE", t/"cache"), mock.patch.object(data, "BUNDLED", t/"bundled"),
                  mock.patch.object(data, "_binance", self.binance), mock.patch.object(data, "_coingecko", self.gecko), mock.patch.dict(data._FAIL, clear=True)):
            p.start(); self.addCleanup(p.stop)
    def up(self, sym="BTC", n=600, seed=0): return S.upload(sym, fixture(n, seed).to_csv(index=False).encode())

class TestData(Base):
    def test_quality_report_matches_selected_range(self):
        d = fixture(200); d = pd.concat([d, d.iloc[[10]], d.iloc[[190]]]); (self.t/"uploads"/"BTC.csv").write_text(d.to_csv(index=False))
        df, rep, meta = data.load("BTC", 30)
        self.assertEqual((rep["original_rows"], rep["duplicates_removed"], rep["final_rows"], meta["rows"]), (31, 1, 30, 30))   # old duplicate is outside the window
        _, rep_all, _ = data.load("BTC", 0); self.assertEqual((rep_all["original_rows"], rep_all["duplicates_removed"]), (202, 2))
        self.assertIn("30-day", rep["scope"])
    def test_symbol_validation(self):
        for bad in ("../etc/passwd", "DOGE", "", "BTC;rm", "btc/../x"):
            with self.assertRaises(S.ApiError) as c: S.get_data(bad, "1y")
            self.assertEqual(c.exception.status, 422)
        with self.assertRaises(S.ApiError) as c: S.get_data("BTC", "3y")
        self.assertEqual(c.exception.status, 422)
    def test_upload_rejections(self):
        good = fixture(60).to_csv(index=False)
        cases = [b"", b"\x00\x01\x02binary", b"date,open\n2022-01-01,1", good.encode()[:300], b"date,close\n" + b"x,y\n"*50,
                 b"a" * (data.MAX_UPLOAD + 1), "\ufffd".encode("utf-16")]
        for c in cases:
            with self.assertRaises(S.ApiError, msg=c[:20]) as e: S.upload("BTC", c)
            self.assertEqual(e.exception.status, 400)
        with self.assertRaises(S.ApiError): S.upload("../x", good.encode())
        self.assertEqual(list((self.t/"uploads").iterdir()), [])     # nothing saved from rejected files
        self.assertTrue(S.upload("BTC", good.encode())["ok"])
        self.assertEqual([p.name for p in (self.t/"uploads").iterdir()], ["BTC.csv"])
    def test_live_success_then_cache(self):
        self.binance.side_effect = None; self.binance.return_value = fixture(100)
        _, _, m = data.load("BTC", 30); self.assertEqual((m["source"], m["fallback"]), ("Binance public API", False))
        os.utime(self.t/"cache"/"BTC.csv", (time.time() - 99999,) * 2); self.binance.side_effect = OSError("offline")
        _, _, m = data.load("BTC", 30, refresh=True); self.assertTrue(m["fallback"]); self.assertIn("CACHED", m["warning"])
    def test_bundled_fallback_labeled_and_backoff(self):
        fixture(300).to_csv(self.t/"bundled"/"BTC.csv", index=False)
        (self.t/"bundled"/"manifest.json").write_text('{"source": "test", "retrieved_at": "2020-01-01"}')
        _, _, m = data.load("BTC", 90); self.assertTrue(m["fallback"]); self.assertIn("BUNDLED", m["source"]); self.assertIn("NOT current", m["warning"])
        data.load("BTC", 90); self.assertEqual(self.binance.call_count, 1)    # second call skips the slow live attempt
        data.load("BTC", 90, refresh=True); self.assertEqual(self.binance.call_count, 2)  # explicit refresh retries
    def test_no_data_is_503_not_fabricated(self):
        with self.assertRaises(S.ApiError) as c: S.get_data("SOL", "1y")
        self.assertEqual(c.exception.status, 503)

class TestEndpoints(Base):
    def setUp(self): super().setUp(); self.up("BTC", 600, 0); self.up("ETH", 600, 1)
    def test_data_and_ranges(self):
        for r, n in (("7d", 7), ("30d", 30), ("90d", 90), ("1y", 365)):
            j = S.get_data("BTC", r); self.assertEqual((j["meta"]["rows"], j["quality"]["final_rows"], len(j["series"]["date"])), (n, n, n))
    def test_anomaly_series_matches_selected_variable(self):
        df = S._get("BTC", "1y")[0]
        for var, s in (("ret", df.close.pct_change()), ("volume_chg", df.volume.pct_change())):
            j = S.anomalies("BTC", "1y", "zscore", 2.5, var); self.assertEqual(j["var"], var)
            np.testing.assert_allclose(j["series"]["values"], s.dropna().values); self.assertEqual(len(j["series"]["date"]), len(j["series"]["values"]))
            self.assertTrue(all(a["date"] in set(j["series"]["date"]) for a in j["anomalies"]))
        for bad in (dict(threshold=0), dict(var="close"), dict(method="nope")):
            with self.assertRaises(S.ApiError): S.anomalies("BTC", "1y", **{"method": "zscore", **bad})
    def test_prediction_alignment_and_metrics(self):
        df = S._get("BTC", "max")[0]; c = df.set_index("date").close; nxt = c.shift(-1)
        r = S.train("BTC", "max", "classification", "logistic")
        for p in r["predictions"]:
            d = pd.Timestamp(p["date"]); self.assertEqual(p["actual"], int(nxt[d] > c[d])); self.assertEqual(p["pred"], int(p["prob_up"] >= .5))
        y = [p["actual"] for p in r["predictions"]]; yh = [p["pred"] for p in r["predictions"]]
        self.assertAlmostEqual(r["metrics"]["accuracy"], np.mean(np.array(y) == np.array(yh)), places=5)
        self.assertEqual(sum(map(sum, r["confusion_matrix"])), r["n_test"]); self.assertEqual(len(r["predictions"]), r["n_test"])
        self.assertEqual(r["latest_forecast"]["as_of"], str(df.date.iloc[-1].date())); self.assertLess(r["train_period"][1], r["test_period"][0])
        g = S.train("BTC", "max", "regression", "ridge")
        for p in g["predictions"]:
            d = pd.Timestamp(p["date"]); self.assertAlmostEqual(p["actual"], nxt[d]/c[d] - 1, places=5)
        e = np.array([p["pred"] - p["actual"] for p in g["predictions"]]); self.assertAlmostEqual(g["metrics"]["rmse"], float(np.sqrt((e**2).mean())), places=4)
    def test_report_periods_and_pdf_export(self):
        for period, expected in (("1d", 1), ("1w", 7), ("1m", 30), ("3m", 90), ("6m", 180)):
            report = S.crypto_report("BTC", period)
            self.assertTrue(report.content.startswith(b"%PDF-"))
            self.assertIn(f"_{period}_", report.name)
        all_report = S.crypto_report("BTC", "all")
        self.assertTrue(all_report.content.startswith(b"%PDF-"))
        with self.assertRaises(S.ApiError) as err:
            S.crypto_report("BTC", "3y")
        self.assertEqual(err.exception.status, 422)

    def test_new_dashboard_ranges(self):
        for rng, expected in (("1d", 1), ("1w", 7), ("1m", 30), ("3m", 90), ("6m", 180), ("5y", 600), ("all", 600)):
            data_response = S.get_data("BTC", rng)
            self.assertEqual(data_response["meta"]["rows"], expected)
            if rng == "1d":
                self.assertIsNone(data_response["eda"]["kpi"]["period_return"])
                self.assertIsNone(data_response["eda"]["kpi"]["volatility_annualized"])

    def test_exports(self):
        r = S.train("BTC", "max", "classification", "random_forest"); c = S.train("BTC", "max", "classification", "random_forest", fmt="csv")
        self.assertEqual((c.text.count("\n"), c.text.split("\n")[0]), (r["n_test"] + 1, "date,actual,pred,prob_up"))
        df = S._get("BTC", "1y")[0]; self.assertEqual(S.get_features("BTC", "1y", "csv").text.count("\n"), len(df) + 1)
        self.assertEqual(S.clean_csv("BTC", "1y").text.count("\n"), len(df) + 1); self.assertEqual(len(S.get_features("BTC", "1y")["rows"]), 30)
    def test_errors(self):
        for kw in (dict(test_frac=0.9), dict(horizon=0), dict(model="xgb"), dict(task="x")):
            with self.assertRaises(S.ApiError) as c: S.train("BTC", "max", **kw)
            self.assertEqual(c.exception.status, 422)
        with self.assertRaises(S.ApiError): S.train("BTC", "7d")          # too little data
        with self.assertRaises(S.ApiError): S.compare("BTC,BTC", "1y")
        with self.assertRaises(S.ApiError) as c: S.clusters("BTC", "1y", 99)
        self.assertEqual(c.exception.status, 422)
        self.assertEqual(len(S.compare("BTC,ETH", "1y")["normalized"]), 2); self.assertEqual(len(S.clusters("BTC", "1y", 3)["profile"]), 3)
        self.assertTrue(S.remove_upload("BTC")["ok"]); self.assertFalse((self.t/"uploads"/"BTC.csv").exists())
if __name__ == "__main__": unittest.main()
