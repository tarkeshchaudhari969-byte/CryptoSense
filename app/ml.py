"""Chronological ML pipeline with baselines. Imputer/scaler are fit on training rows only (sklearn Pipeline)."""
import time, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn import metrics as M
from .features import feature_cols

SEED = 42
CLS = {"logistic": lambda p: LogisticRegression(max_iter=1000, C=p.get("C", 1.0)),
       "decision_tree": lambda p: DecisionTreeClassifier(max_depth=p.get("max_depth", 4), random_state=SEED),
       "random_forest": lambda p: RandomForestClassifier(n_estimators=p.get("n_estimators", 200), max_depth=p.get("max_depth", 6), random_state=SEED, n_jobs=1)}
REG = {"linear": lambda p: LinearRegression(), "ridge": lambda p: Ridge(alpha=p.get("alpha", 1.0)),
       "random_forest": lambda p: RandomForestRegressor(n_estimators=p.get("n_estimators", 200), max_depth=p.get("max_depth", 6), random_state=SEED, n_jobs=1)}
WARMUP = ("ret_lag5", "sma_ratio_20", "vol_20", "rsi14", "macd")

def prepare(df, f, task, horizon=1):
    f = f.copy(); nxt = df["close"].shift(-horizon)
    f["target_ret"] = nxt/df["close"] - 1; f["target_up"] = np.where(nxt.isna(), np.nan, (nxt > df["close"]).astype(float))
    tgt = "target_up" if task == "classification" else "target_ret"
    d = f.dropna(subset=[tgt]).dropna(subset=[c for c in WARMUP if c in f])   # drop indicator warm-up rows
    return d, feature_cols(f), tgt, f

def run(df, f, task="classification", model="random_forest", test_frac=0.25, horizon=1, params=None):
    params = params or {}
    if task not in ("classification", "regression"): raise ValueError("task must be classification or regression")
    d, cols, tgt, f = prepare(df, f, task, horizon)
    if len(d) < 80: raise ValueError(f"Only {len(d)} usable rows; need at least 80. Choose a longer history.")
    cut = int(len(d)*(1 - test_frac)); gap = horizon - 1          # gap stops overlapping labels leaking into the test period
    tr, te = d.iloc[:cut - gap], d.iloc[cut:]
    if len(te) < 20: raise ValueError("Test set too small; lower the test fraction or use more history.")
    Xtr, ytr, Xte, yte = tr[cols], tr[tgt].values, te[cols], te[tgt].values
    reg = CLS if task == "classification" else REG
    if model not in reg: raise ValueError(f"Unknown model '{model}'. Options: {list(reg)}")
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", reg[model](params))])
    t0 = time.time(); pipe.fit(Xtr, ytr); dur = time.time() - t0
    out = {"task": task, "model": model, "horizon": horizon,
           "target_definition": f"1 if close[t+{horizon}] > close[t] else 0" if task == "classification" else f"close[t+{horizon}]/close[t] - 1 (simple return)",
           "n_train": len(tr), "n_test": len(te), "train_period": [str(tr.date.min().date()), str(tr.date.max().date())],
           "test_period": [str(te.date.min().date()), str(te.date.max().date())], "train_seconds": round(dur, 3)}
    dates = [str(x.date()) for x in te.date]
    if task == "classification":
        proba = pipe.predict_proba(Xte)[:, 1]; pred = (proba >= params.get("threshold", 0.5)).astype(int); yte = yte.astype(int)
        base = np.full(len(yte), int(round(ytr.mean()))); naive = (te["ret"].fillna(0).values > 0).astype(int)
        def sc(p): return {"accuracy": M.accuracy_score(yte, p), "balanced_accuracy": M.balanced_accuracy_score(yte, p), "precision": M.precision_score(yte, p, zero_division=0),
                           "recall": M.recall_score(yte, p, zero_division=0), "f1": M.f1_score(yte, p, zero_division=0)}
        out["metrics"] = sc(pred); out["metrics"]["roc_auc"] = float(M.roc_auc_score(yte, proba)) if len(set(yte)) > 1 else None
        out["metrics"]["brier"] = float(M.brier_score_loss(yte, proba)); out["confusion_matrix"] = M.confusion_matrix(yte, pred, labels=[0, 1]).tolist()
        out["baselines"] = {"majority_class": sc(base), "naive_previous_direction": sc(naive)}
        out["beats_baseline"] = bool(out["metrics"]["balanced_accuracy"] > out["baselines"]["majority_class"]["balanced_accuracy"] + 0.02)
        out["class_distribution_test"] = {"up": int(yte.sum()), "down": int(len(yte) - yte.sum())}
        out["predictions"] = [{"date": a, "actual": int(b), "pred": int(c), "prob_up": float(p)} for a, b, c, p in zip(dates, yte, pred, proba)]
        scoring = "balanced_accuracy"
    else:
        pred = pipe.predict(Xte); base = np.full(len(yte), ytr.mean())
        def sc(p): return {"mae": M.mean_absolute_error(yte, p), "mse": M.mean_squared_error(yte, p), "rmse": float(np.sqrt(M.mean_squared_error(yte, p))),
                           "r2": M.r2_score(yte, p), "directional_accuracy": float(np.mean(np.sign(p) == np.sign(yte)))}
        out["metrics"] = sc(pred); out["baselines"] = {"constant_train_mean": sc(base)}
        out["beats_baseline"] = bool(out["metrics"]["rmse"] < out["baselines"]["constant_train_mean"]["rmse"]*0.99)
        out["predictions"] = [{"date": a, "actual": float(b), "pred": float(c), "error": float(c - b)} for a, b, c in zip(dates, yte, pred)]
        scoring = "neg_root_mean_squared_error"
    out["verdict"] = ("Model beats the baseline on unseen data. Check stability before trusting it." if out["beats_baseline"]
                      else "Model does NOT meaningfully beat the baseline. Common for daily crypto data; no predictive edge is demonstrated.")
    pi = permutation_importance(pipe, Xte, yte, n_repeats=5, random_state=SEED, scoring=scoring)
    out["importance"] = {"method": "Permutation on the test set: score drop when a feature is shuffled. Measures model reliance, not causation; correlated features share credit.",
                         "values": [{"feature": a, "importance": float(b)} for a, b in sorted(zip(cols, pi.importances_mean), key=lambda x: -x[1])[:15]]}
    inner = pipe.named_steps["m"]
    if hasattr(inner, "coef_"): out["coefficients"] = sorted([{"feature": c, "coef": float(v)} for c, v in zip(cols, np.ravel(inner.coef_))], key=lambda x: -abs(x["coef"]))[:15]
    last = f.dropna(subset=list(WARMUP)).iloc[[-1]]       # latest completed day: outcome not yet known
    lp = pipe.predict_proba(last[cols])[0, 1] if task == "classification" else pipe.predict(last[cols])[0]
    out["latest_forecast"] = {"as_of": str(last.date.iloc[0].date()), "value": float(lp), "kind": "prob_up" if task == "classification" else "predicted_return",
                              "note": "Outcome not yet known. Experimental estimate, not advice."}
    return _clean(out)

def _clean(o):
    if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list): return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.integer): return int(o)
    return o
