"""Classical reference classifier on the numeric feature sheet, nested CV.

Outer 5-fold stratified CV gives out-of-fold scores for every eval session so
the reference can enter the same AUROC/bootstrap comparisons as the models.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..manifest import Session


def feature_frame(sessions: list[Session], cache_dir: str | Path) -> pd.DataFrame:
    rows = []
    for s in sessions:
        p = Path(cache_dir) / s.session_id / "inputs.json"
        if not p.exists():
            continue
        feats = json.loads(p.read_text())["features"]
        flat = {f"{task}__{k}": v for task, d in feats.items() for k, v in d.items()}
        rows.append({"session_id": s.session_id, "dataset": s.dataset, "split": s.split, "label": s.label, **flat})
    return pd.DataFrame(rows)


def _pipelines() -> dict[str, tuple[Pipeline, dict]]:
    # SVC decision_function is used for ranking (AUROC); no probability calibration needed
    svm = Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler()), ("clf", SVC())])
    gbt = Pipeline([("imp", SimpleImputer()), ("clf", HistGradientBoostingClassifier(min_samples_leaf=5))])
    return {
        "svm": (svm, {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01]}),
        "gbt": (gbt, {"clf__max_depth": [2, 3], "clf__learning_rate": [0.05, 0.1], "clf__max_iter": [50, 200]}),
    }


def nested_cv_scores(df: pd.DataFrame, dataset: str = "behavioral", outer_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    sub = df[(df.dataset == dataset) & (df.split == "eval")].reset_index(drop=True)
    X = sub.drop(columns=["session_id", "dataset", "split", "label"]).to_numpy(dtype=float)
    y = sub["label"].to_numpy()
    outer = StratifiedKFold(outer_folds, shuffle=True, random_state=seed)
    out = sub[["session_id", "label"]].copy()
    for name, (pipe, grid) in _pipelines().items():
        inner = GridSearchCV(pipe, grid, cv=StratifiedKFold(3, shuffle=True, random_state=seed), scoring="roc_auc")
        method = "decision_function" if name == "svm" else "predict_proba"
        pred = cross_val_predict(inner, X, y, cv=outer, method=method)
        s = pred if pred.ndim == 1 else pred[:, 1]
        # map to 0-100 like the model risk_score (rank-preserving); SVM margins via logistic squash
        out[f"score_{name}"] = (100 / (1 + np.exp(-s))) if name == "svm" else s * 100
    return out
