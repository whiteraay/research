"""Turn runs.jsonl into the result tables the RQs ask for."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M


def load_runs(log_path: str | Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in Path(log_path).read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df["risk_score"] = pd.to_numeric(df.get("risk_score"), errors="coerce")
    return df


def aggregate_repeats(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, dataset, condition, prompt_mode, session): mean score over repeats."""
    ok = df[df["status"] == "ok"]
    g = ok.groupby(["model", "dataset", "condition", "prompt_mode", "session_id"], as_index=False)
    return g.agg(label=("label", "first"), score=("risk_score", "mean"), n_rep=("repeat", "count"))


def main_table(df: pd.DataFrame) -> pd.DataFrame:
    """RQ1/RQ2: AUROC (DeLong CI), sens@90%spec, calibration, refusal rate per cell."""
    agg = aggregate_repeats(df)
    out = []
    for keys, g in agg.groupby(["model", "dataset", "condition", "prompt_mode"]):
        if g["label"].nunique() < 2 or len(g) < 4:
            continue
        y, s = g["label"].to_numpy(), g["score"].to_numpy()
        a = M.auroc_ci(y, s)
        sp = M.sensitivity_at_specificity(y, s)
        cal = M.calibration(y, s / 100.0)
        cell = df[(df.model == keys[0]) & (df.dataset == keys[1]) & (df.condition == keys[2]) & (df.prompt_mode == keys[3])]
        out.append({
            "model": keys[0], "dataset": keys[1], "condition": keys[2], "prompt_mode": keys[3],
            "n_sessions": len(g), "auroc": a["auroc"], "ci_low": a["ci_low"], "ci_high": a["ci_high"],
            "sens_at_90spec": sp["sensitivity"], "brier": cal["brier"], "ece": cal["ece"],
            "refusal_rate": float((cell["status"].isin(["refused", "error"])).mean()),
            "unparsed_rate": float((cell["status"] == "unparsed").mean()),
        })
    return pd.DataFrame(out).sort_values(["dataset", "condition", "prompt_mode", "auroc"], ascending=[True, True, True, False])


def transfer_gap(table: pd.DataFrame) -> pd.DataFrame:
    """RQ2a/b: synthetic − behavioral AUROC per model (per condition, prompt), and rank agreement."""
    piv = table.pivot_table(index=["model", "condition", "prompt_mode"], columns="dataset", values="auroc").reset_index()
    if {"synthetic", "behavioral"} <= set(piv.columns):
        piv["gap"] = piv["synthetic"] - piv["behavioral"]
    return piv


def rank_agreement(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cond, pm), g in table.groupby(["condition", "prompt_mode"]):
        a = g[g.dataset == "behavioral"].set_index("model")["auroc"].to_dict()
        b = g[g.dataset == "synthetic"].set_index("model")["auroc"].to_dict()
        if len(set(a) & set(b)) >= 3:
            rows.append({"condition": cond, "prompt_mode": pm, **M.kendall_tau_rankings(a, b)})
    return pd.DataFrame(rows)


def pairwise_model_diffs(df: pd.DataFrame, dataset: str, condition: str, prompt_mode: str) -> pd.DataFrame:
    """RQ1: paired bootstrap of AUROC differences between every pair of models on the same sessions."""
    agg = aggregate_repeats(df)
    sub = agg[(agg.dataset == dataset) & (agg.condition == condition) & (agg.prompt_mode == prompt_mode)]
    wide = sub.pivot(index="session_id", columns="model", values="score").dropna()
    labels = sub.drop_duplicates("session_id").set_index("session_id").loc[wide.index, "label"].to_numpy()
    models = list(wide.columns)
    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            r = M.paired_bootstrap_auc_diff(labels, wide[a].to_numpy(), wide[b].to_numpy())
            rows.append({"model_a": a, "model_b": b, **r})
    return pd.DataFrame(rows)


def reliability(df: pd.DataFrame) -> pd.DataFrame:
    """RQ3b: test-retest kappa of risk_label across repeats."""
    ok = df[(df["status"] == "ok") & df["risk_label"].notna()]
    rows = []
    for keys, g in ok.groupby(["model", "dataset", "condition", "prompt_mode"]):
        piv = g.pivot_table(index="session_id", columns="repeat", values="risk_label", aggfunc="first").dropna()
        if piv.shape[1] < 2:
            continue
        reps = [piv[c].to_numpy() for c in piv.columns]
        rows.append({"model": keys[0], "dataset": keys[1], "condition": keys[2], "prompt_mode": keys[3],
                     **M.test_retest_kappa(reps), "n_sessions": len(piv)})
    return pd.DataFrame(rows)


def task_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """RQ3a (self-report side): which task each model says was most informative."""
    ok = df[df["status"] == "ok"]
    return (ok.groupby(["model", "dataset", "condition"])["task_most_informative"]
              .value_counts(normalize=True).rename("share").reset_index())


def write_all(log_path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_runs(log_path)
    tab = main_table(df)
    files = {}
    for name, t in {
        "main_table": tab,
        "transfer_gap": transfer_gap(tab) if len(tab) else pd.DataFrame(),
        "rank_agreement": rank_agreement(tab) if len(tab) else pd.DataFrame(),
        "reliability": reliability(df),
        "task_attribution": task_attribution(df),
    }.items():
        p = out_dir / f"{name}.csv"
        t.to_csv(p, index=False)
        files[name] = p
    return files
