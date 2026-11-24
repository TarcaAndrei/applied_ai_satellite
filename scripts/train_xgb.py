"""B1 baseline: XGBoost on hand-crafted spectral features (see sat/features.py).

    python -m scripts.train_xgb                      # all features
    python -m scripts.train_xgb --features index     # spectral indices only (ratios, no raw reflectance)
    python -m scripts.train_xgb --tune               # small grid on val first

Fits on `train`, early-stops on `val`, scores `test` once. Also reports the majority-class
floor (B0) and a logistic regression on the same features for comparison.
Writes runs/<name>/{metrics.json, confusion_test.csv, feature_importance.csv, model.json, test_predictions.csv}.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sat.eurosat import CACHE_DIR, CLASSES, REPO_ROOT, load_cache, load_split
from sat.features import compute_features, feature_names

FEATURE_VERSION = "v1"


def load_features(seed: int) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    split = load_split(seed=seed)
    names = feature_names()
    path = CACHE_DIR / f"eurosat_ms_features_{FEATURE_VERSION}.npy"
    if path.exists():
        X = np.load(path)
    else:
        t = time.time()
        print(f"[xgb] computing features -> {path}")
        X = compute_features(load_cache())
        np.save(path, X)
        print(f"[xgb] features {X.shape} in {time.time() - t:.0f}s")
    assert X.shape == (len(split), len(names)), (X.shape, len(split), len(names))
    return X, names, split


def select(names: list[str], group: str) -> list[int]:
    if group == "all":
        return list(range(len(names)))
    if group == "index":  # ratio features: less sensitive to absolute radiometry (L1C vs L2A)
        prefixes = ("NDVI", "NDWI", "MNDWI", "NDBI", "NDRE")
        return [i for i, n in enumerate(names) if n.startswith(prefixes)]
    if group == "band":
        return [i for i, n in enumerate(names) if n.startswith("B")]
    raise ValueError(group)


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro")),
        "per_class_f1": dict(zip(CLASSES, map(float, f1_score(y, p, average=None, labels=range(len(CLASSES)))))),
    }


def make_xgb(args, **over) -> xgb.XGBClassifier:
    params = dict(
        objective="multi:softprob", tree_method="hist", device=args.device,
        n_estimators=3000, learning_rate=args.lr, max_depth=args.max_depth,
        min_child_weight=args.min_child_weight, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, early_stopping_rounds=100, eval_metric="mlogloss",
        n_jobs=args.n_jobs, random_state=args.seed,
    )
    params.update(over)
    return xgb.XGBClassifier(**params)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["all", "band", "index"], default="all")
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--min-child-weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--tune", action="store_true", help="grid over max_depth x min_child_weight on val")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    name = args.name or f"xgb_b1_{args.features}"
    out = REPO_ROOT / "runs" / name
    out.mkdir(parents=True, exist_ok=True)

    X, names, split = load_features(args.seed)
    cols = select(names, args.features)
    X, names = X[:, cols], [names[i] for i in cols]
    y = split["label"].to_numpy()
    m = {s: (split["split"] == s).to_numpy() for s in ("train", "val", "test")}
    Xtr, ytr, Xva, yva, Xte, yte = X[m["train"]], y[m["train"]], X[m["val"]], y[m["val"]], X[m["test"]], y[m["test"]]
    print(f"[xgb] {name}: {len(names)} features, train {len(ytr)}, val {len(yva)}, test {len(yte)}")

    tuning = []
    if args.tune:
        for d in (4, 6, 8):
            for mcw in (1, 5):
                clf = make_xgb(args, max_depth=d, min_child_weight=mcw).fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
                acc = accuracy_score(yva, clf.predict(Xva))
                tuning.append({"max_depth": d, "min_child_weight": mcw, "val_accuracy": float(acc), "best_iteration": int(clf.best_iteration)})
                print(f"  depth {d} mcw {mcw}: val acc {acc:.4f} @ {clf.best_iteration} trees")
        best = max(tuning, key=lambda r: r["val_accuracy"])
        args.max_depth, args.min_child_weight = best["max_depth"], best["min_child_weight"]

    t = time.time()
    clf = make_xgb(args).fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    fit_s = time.time() - t
    p_val, p_te = clf.predict(Xva), clf.predict(Xte)

    # Comparisons on the same split and features.
    majority = np.bincount(ytr).argmax()
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)).fit(Xtr, ytr)

    metrics = {
        "model": "xgboost", "features": args.features, "n_features": len(names),
        "params": {"max_depth": args.max_depth, "min_child_weight": args.min_child_weight, "learning_rate": args.lr,
                   "subsample": 0.8, "colsample_bytree": 0.8, "best_iteration": int(clf.best_iteration)},
        "fit_seconds": round(fit_s, 1), "seed": args.seed, "feature_version": FEATURE_VERSION,
        "val": scores(yva, p_val), "test": scores(yte, p_te),
        "baselines_test": {
            "B0_majority": scores(yte, np.full_like(yte, majority)),
            "logreg": scores(yte, logreg.predict(Xte)),
        },
        "tuning": tuning,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(confusion_matrix(yte, p_te, labels=range(len(CLASSES))), index=CLASSES, columns=CLASSES) \
        .to_csv(out / "confusion_test.csv")
    imp = clf.get_booster().get_score(importance_type="gain")
    pd.Series({n: imp.get(f"f{i}", 0.0) for i, n in enumerate(names)}, name="gain") \
        .sort_values(ascending=False).to_csv(out / "feature_importance.csv")
    proba = clf.predict_proba(Xte)
    pd.DataFrame({"path": split.loc[m["test"], "path"].to_numpy(), "label": yte, "pred": p_te,
                  **{f"p_{c}": proba[:, i] for i, c in enumerate(CLASSES)}}).to_csv(out / "test_predictions.csv", index=False)
    clf.save_model(out / "model.json")

    print(f"[xgb] {clf.best_iteration} trees, fit {fit_s:.0f}s")
    print(f"  val   acc {metrics['val']['accuracy']:.4f}  macro-F1 {metrics['val']['macro_f1']:.4f}")
    print(f"  test  acc {metrics['test']['accuracy']:.4f}  macro-F1 {metrics['test']['macro_f1']:.4f}")
    for k, v in metrics["baselines_test"].items():
        print(f"  {k:12s} test acc {v['accuracy']:.4f}  macro-F1 {v['macro_f1']:.4f}")
    print("  per-class test F1: " + ", ".join(f"{c} {f:.3f}" for c, f in metrics["test"]["per_class_f1"].items()))
    print(f"[xgb] wrote {out}")


if __name__ == "__main__":
    main()
