"""Train the XGBoost default-probability model and save everything the app needs.

Usage:  python train.py [--data path/to/LoansData_sample.csv.gz]
Outputs: model/xgb_model.json, model/meta.json
"""
import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from pandas.api.types import is_numeric_dtype
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, f1_score, log_loss, mean_absolute_error,
                             mean_squared_error, r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

from loan_features import prepare

SEED = 42
URL = "https://raw.githubusercontent.com/tiwvis/MLBD-Projects/main/data_sets/LoansData_sample.csv.gz"
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading dataset ...")
        urllib.request.urlretrieve(URL, path)
    return pd.read_csv(path, compression="gzip", low_memory=False)


def main(data_path: Path):
    df = load(data_path)
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
    y = (df["loan_status"] == "Charged Off").astype(int).to_numpy()
    print(f"Closed loans: {len(df):,} | charge-off rate {y.mean():.3%}")

    X = prepare(df)
    drop = X.isna().mean()[lambda s: s > 0.30].index.tolist()
    X = X.drop(columns=drop)
    cat_cols = [c for c in X.columns if not is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]

    X_enc = pd.get_dummies(X, columns=cat_cols, prefix_sep="=", dtype=float)

    # Same split recipe as the notebook: 70 / 10 / 20 stratified.
    idx = np.arange(len(X_enc))
    trv, te = train_test_split(idx, test_size=0.20, stratify=y, random_state=SEED)
    tr, va = train_test_split(trv, test_size=0.125, stratify=y[trv], random_state=SEED)

    medians = X_enc.iloc[tr].median()
    X_enc = X_enc.fillna(medians).astype("float32")

    model = xgb.XGBClassifier(
        n_estimators=1500, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_lambda=2.0, eval_metric="auc", early_stopping_rounds=50,
        tree_method="hist", random_state=SEED, n_jobs=-1)
    model.fit(X_enc.iloc[tr], y[tr], eval_set=[(X_enc.iloc[va], y[va])], verbose=False)

    p = np.clip(model.predict_proba(X_enc.iloc[te])[:, 1], 1e-6, 1 - 1e-6)
    yt = y[te]
    pred = (p >= 0.5).astype(int)
    metrics = {
        "ROC-AUC": roc_auc_score(yt, p), "PR-AUC": average_precision_score(yt, p),
        "LogLoss": log_loss(yt, p), "RMSE": float(np.sqrt(mean_squared_error(yt, p))),
        "MAE": mean_absolute_error(yt, p), "R2": r2_score(yt, p),
        "F1@0.5": f1_score(yt, pred), "Recall@0.5": recall_score(yt, pred),
    }
    print({k: round(float(v), 4) for k, v in metrics.items()}, "| best iter", model.best_iteration)

    fpr, tpr, _ = roc_curve(yt, p)
    keep = np.linspace(0, len(fpr) - 1, 150).astype(int)
    frac, mean_pred = calibration_curve(yt, p, n_bins=10, strategy="quantile")

    grade_rate = {}
    if "grade" in X.columns:
        grade_rate = (pd.Series(y, index=X.index).groupby(X["grade"]).mean() * 100).round(2).to_dict()

    cats = {c: sorted(X[c].dropna().astype(str).unique().tolist()) for c in cat_cols}
    ranges = {c: [float(X[c].quantile(0.01)), float(X[c].quantile(0.99))] for c in num_cols}

    meta = {
        "columns": X_enc.columns.tolist(), "num_cols": num_cols, "cat_cols": cat_cols,
        "categories": cats, "medians": {k: float(v) for k, v in medians.items()},
        "ranges": ranges, "dropped_for_missing": drop,
        "n_loans": int(len(df)), "default_rate": float(y.mean()),
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "best_iteration": int(model.best_iteration),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "roc": {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()},
        "calibration": {"predicted": mean_pred.round(4).tolist(), "observed": frac.round(4).tolist()},
        "grade_default_rate": grade_rate,
    }
    MODEL_DIR.mkdir(exist_ok=True)
    model.get_booster().save_model(MODEL_DIR / "xgb_model.json")
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
    print("Saved model/xgb_model.json and model/meta.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "LoansData_sample.csv.gz"))
    main(Path(ap.parse_args().data))
