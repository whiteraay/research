import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_metrics_basic():
    from apd_bench.analysis import metrics as M

    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
    a = M.auroc_ci(y, s)
    assert 0.5 < a["auroc"] <= 1.0 and a["ci_low"] <= a["auroc"] <= a["ci_high"]
    assert M.cohen_kappa([1, 0, 1], [1, 0, 1]) == 1.0
    assert "brier" in M.calibration(y, s)


def test_parse_json():
    from apd_bench.models.base import parse_json

    assert parse_json('```json\n{"risk_score": 42}\n```')["risk_score"] == 42
    assert parse_json('Sure. {"risk_score": 7, "x": [1,2]} done')["risk_score"] == 7
    assert parse_json("I cannot help with that.") is None


def test_end_to_end_mock(tmp_path: Path):
    from apd_bench.analysis.report import write_all
    from apd_bench.config import load_config
    from apd_bench.manifest import load_manifest
    from apd_bench.runner import run_grid

    subprocess.run([sys.executable, str(ROOT / "examples/make_toy_manifest.py"), str(tmp_path / "toy")], check=True)
    cfg = load_config(ROOT / "configs/models.yaml")
    sessions = load_manifest(tmp_path / "toy/manifest.csv")
    log = run_grid(cfg, sessions, tmp_path / "toy/manifest.csv", tmp_path / "cache", tmp_path / "runs",
                   models=["mock"], conditions=["transcript", "feature_sheet"], repeats=2, skip_media=True)
    files = write_all(log, tmp_path / "results")
    import pandas as pd

    tab = pd.read_csv(files["main_table"])
    assert set(tab.dataset) == {"behavioral", "synthetic"}
    assert len(tab) == 8  # 2 datasets x 2 conditions x 2 prompt modes
    assert (tab[tab.prompt_mode == "zero_shot"].auroc > 0.5).all()
    gap = pd.read_csv(files["transfer_gap"])
    assert "gap" in gap.columns
