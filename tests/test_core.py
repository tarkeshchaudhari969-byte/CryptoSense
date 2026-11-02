"""Run: python -m unittest discover tests.  Uses a SYNTHETIC random walk as a test fixture only; never shown in the app."""
import unittest, numpy as np, pandas as pd
from app import data, features, ml, analysis

def fixture(n=600, seed=0):
    r = np.random.default_rng(seed); c = 100*np.exp(np.cumsum(r.normal(0, .02, n)))
    d = pd.DataFrame({"date": pd.date_range("2022-01-01", periods=n), "close": c})
    d["open"] = d.close.shift(1).fillna(c[0]); d["high"] = d[["open", "close"]].max(axis=1)*1.01; d["low"] = d[["open", "close"]].min(axis=1)*.99
    d["volume"] = r.uniform(1e3, 1e4, n); return d

class T(unittest.TestCase):
    def test_clean(self):
        d = fixture(50); d = pd.concat([d, d.iloc[[3]]]); d.loc[5, "close"] = -1
        out, rep = data.clean(d)
        self.assertEqual((rep["duplicates_removed"], rep["invalid_close_removed"]), (1, 1)); self.assertTrue(out.date.is_monotonic_increasing)
    def test_no_future_leakage(self):
        d = fixture(); f1 = features.build_features(d); d2 = d.copy(); d2.loc[400:, "close"] *= 3; d2.loc[400:, "volume"] *= 5
        f2 = features.build_features(d2); cols = features.feature_cols(f1)
        pd.testing.assert_frame_equal(f1.loc[:399, cols], f2.loc[:399, cols])
    def test_target_alignment(self):
        d = fixture(); f = features.build_features(d)
        self.assertEqual(f.target_up[10], float(d.close[11] > d.close[10])); self.assertTrue(np.isnan(f.target_up.iloc[-1]))
    def test_models(self):
        d = fixture(); f = features.build_features(d)
        for task, reg in (("classification", ml.CLS), ("regression", ml.REG)):
            for m in reg:
                with self.subTest(task=task, model=m):
                    r = ml.run(d, f, task, m); self.assertLess(r["train_period"][1], r["test_period"][0])
                    self.assertGreater(r["n_test"], 20); self.assertTrue(r["baselines"] and r["predictions"] and r["importance"]["values"])
    def test_horizon_gap(self):
        d = fixture(); r = ml.run(d, features.build_features(d), horizon=5)
        self.assertLess(r["train_period"][1], r["test_period"][0])
    def test_reproducible(self):
        d = fixture(); f = features.build_features(d); self.assertEqual(ml.run(d, f)["metrics"], ml.run(d, f)["metrics"])
    def test_too_little_data(self):
        d = fixture(60)
        with self.assertRaises(ValueError): ml.run(d, features.build_features(d))
    def test_analysis(self):
        d = fixture(); self.assertLessEqual(analysis.eda(d)["kpi"]["max_drawdown"], 0)
        for m in ("zscore", "modified_z", "iqr", "isolation_forest"): analysis.anomalies(d, m, 3 if m != "isolation_forest" else 2)
        self.assertEqual(len(analysis.clusters(d)["profile"]), 3)
        self.assertEqual(analysis.compare({"A": d, "B": fixture(seed=1)})["overlap_rows"], 600)
if __name__ == "__main__": unittest.main()
