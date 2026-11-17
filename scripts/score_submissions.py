#!/usr/bin/env python
"""Score every outputs/<model>/predictions.csv against the private test labels.

    python scripts/score_submissions.py --outputs outputs --labels data/test_labels.csv --audit data/audit.csv --out results/arm_b

Produces:
    results/arm_b/summary.csv          per model: n, AUROC + DeLong CI, sens@90%spec, Brier, failures, floor gap
    results/arm_b/strata.csv           per model x stratum variable x tertile: AUROC
    results/arm_b/pairwise.csv         paired-bootstrap AUROC differences between models
    results/arm_b/confound_floor.json  AUROC of a logistic model on audit columns only (train->test)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from apd_bench.analysis import metrics as M  # noqa: E402

REQUIRED = {"session_id", "risk_score", "risk_label"}


def load_submission(d: Path) -> pd.DataFrame | None:
    p = d / "predictions.csv"
    if not p.exists():
        print(f"{d.name}: no predictions.csv")
        return None
    df = pd.read_csv(p, dtype={"session_id": str})
    missing = REQUIRED - set(df.columns)
    if missing:
        print(f"{d.name}: predictions.csv missing columns {missing}")
        return None
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
    bad = df["risk_score"].isna() | (df["risk_score"] < 0) | (df["risk_score"] > 100)
    if bad.any():
        print(f"{d.name}: {bad.sum()} rows with invalid risk_score -> set to 50")
        df.loc[bad, "risk_score"] = 50.0
    return df


def confound_floor(audit: pd.DataFrame, train_labels: pd.DataFrame, test_labels: pd.DataFrame) -> dict:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cols = [c for c in ("duration_s", "mean_luminance", "luminance_std", "face_detect_rate", "audio_snr_db", "speech_fraction") if c in audit]
    a = audit.set_index("session_id")[cols]
    Xtr = a.loc[train_labels.session_id].to_numpy(float)
    ytr = (train_labels.cohort == "elevated").astype(int).to_numpy()
    Xte = a.loc[test_labels.session_id].to_numpy(float)
    yte = (test_labels.cohort == "elevated").astype(int).to_numpy()
    clf = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(Xtr, ytr)
    s = clf.predict_proba(Xte)[:, 1]
    return {"columns": cols, **M.auroc_ci(yte, s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--labels", default="data/test_labels.csv")
    ap.add_argument("--train-labels", default="data/train/train_labels.csv")
    ap.add_argument("--audit", default="data/audit.csv")
    ap.add_argument("--out", default="results/arm_b")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(a.labels, dtype={"session_id": str})
    labels["y"] = (labels.cohort == "elevated").astype(int)
    audit = pd.read_csv(a.audit, dtype={"session_id": str}) if Path(a.audit).exists() else None

    subs: dict[str, pd.DataFrame] = {}
    for d in sorted(Path(a.outputs).iterdir()):
        if d.is_dir():
            df = load_submission(d)
            if df is not None:
                subs[d.name] = df

    floor = None
    if audit is not None and Path(a.train_labels).exists():
        floor = confound_floor(audit, pd.read_csv(a.train_labels, dtype={"session_id": str}), labels)
        (out / "confound_floor.json").write_text(json.dumps(floor, indent=1))

    summary, strata, merged_scores = [], [], {}
    for name, df in subs.items():
        m = labels.merge(df, on="session_id", how="left")
        n_missing = int(m.risk_score.isna().sum())
        m["risk_score"] = m.risk_score.fillna(50.0)
        y, s = m.y.to_numpy(), m.risk_score.to_numpy()
        merged_scores[name] = m.set_index("session_id").risk_score
        auc = M.auroc_ci(y, s)
        sp = M.sensitivity_at_specificity(y, s, 0.90)
        cal = M.calibration(y, s / 100)
        fail = Path(a.outputs) / name / "failures.csv"
        n_fail = len(pd.read_csv(fail)) if fail.exists() else 0
        summary.append({"model": name, "n_test": len(m), "missing_rows": n_missing, "failures_reported": n_fail,
                        "auroc": auc["auroc"], "ci_low": auc["ci_low"], "ci_high": auc["ci_high"],
                        "sens_at_90spec": sp["sensitivity"], "brier": cal["brier"], "ece": cal["ece"],
                        "floor_auroc": floor["auroc"] if floor else np.nan,
                        "gap_over_floor": (auc["auroc"] - floor["auroc"]) if floor else np.nan})
        if audit is not None:
            ma = m.merge(audit, on="session_id", how="left")
            for var in ("duration_tertile", "luminance_tertile", "snr_tertile"):
                if var not in ma:
                    continue
                for lvl, g in ma.groupby(var, observed=True):
                    if g.y.nunique() == 2 and len(g) >= 6:
                        r = M.auroc_ci(g.y, g.risk_score)
                        strata.append({"model": name, "variable": var, "level": lvl, "n": len(g), **r})

    pd.DataFrame(summary).sort_values("auroc", ascending=False).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(strata).to_csv(out / "strata.csv", index=False)

    pair = []
    names = list(merged_scores)
    y_idx = labels.set_index("session_id").y
    for i, x in enumerate(names):
        for z in names[i + 1:]:
            idx = y_idx.index
            r = M.paired_bootstrap_auc_diff(y_idx.loc[idx], merged_scores[x].loc[idx], merged_scores[z].loc[idx])
            pair.append({"model_a": x, "model_b": z, **r})
    pd.DataFrame(pair).to_csv(out / "pairwise.csv", index=False)

    pd.set_option("display.width", 200)
    print(pd.DataFrame(summary).round(3).to_string(index=False))
    if pair:
        print("\npairwise AUROC differences (a - b):")
        print(pd.DataFrame(pair).round(3).to_string(index=False))
    if floor:
        print(f"\nconfound floor (audit columns only): AUROC {floor['auroc']:.3f} [{floor['ci_low']:.3f}, {floor['ci_high']:.3f}]")


if __name__ == "__main__":
    main()
