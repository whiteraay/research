"""Command line:

    apd-bench prepare  manifest.csv --cache cache/            build inputs & feature sheets
    apd-bench run      manifest.csv --cache cache/ --out runs/ --models mock --repeats 2
    apd-bench analyze  runs/runs.jsonl --out results/
    apd-bench reference manifest.csv --cache cache/ --out results/
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .config import load_config
from .manifest import load_manifest

app = typer.Typer(add_completion=False, no_args_is_help=True)
DEFAULT_CFG = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"


@app.command()
def prepare(manifest: Path, cache: Path = Path("cache"), config: Path = DEFAULT_CFG, skip_media: bool = False):
    """Build transcript, feature sheet and frames for every session (label never read)."""
    from .preprocess.feature_sheet import build_inputs

    cfg = load_config(config)
    for s in load_manifest(manifest):
        build_inputs(s, cache, cfg.frames_per_task, cfg.frame_max_side, skip_media=skip_media)
        typer.echo(f"prepared {s.session_id}")


@app.command()
def run(
    manifest: Path,
    cache: Path = Path("cache"),
    out: Path = Path("runs"),
    config: Path = DEFAULT_CFG,
    models: Optional[str] = typer.Option(None, help="comma-separated subset, e.g. gemini,claude"),
    conditions: Optional[str] = None,
    prompt_modes: Optional[str] = None,
    repeats: Optional[int] = None,
    skip_media: bool = False,
    limit: Optional[int] = typer.Option(None, help="pilot: only the first N eval sessions per dataset"),
):
    """Run the evaluation grid; appends to <out>/runs.jsonl and resumes if interrupted."""
    from .runner import run_grid

    cfg = load_config(config)
    sessions = load_manifest(manifest)
    if limit:
        keep, seen = [], {"behavioral": 0, "synthetic": 0}
        for s in sessions:
            if s.split == "calibration":
                keep.append(s)
            elif seen[s.dataset] < limit:
                keep.append(s); seen[s.dataset] += 1
        sessions = keep
    split = lambda v: v.split(",") if v else None
    log = run_grid(cfg, sessions, manifest, cache, out, split(models), split(conditions), split(prompt_modes), repeats, skip_media)
    typer.echo(f"log: {log}")


@app.command()
def check(config: Path = DEFAULT_CFG, models: Optional[str] = None):
    """Send one tiny request to each configured provider to confirm keys, model IDs and JSON output."""
    import os

    from .models import Request, make_client

    cfg = load_config(config)
    env = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}
    names = models.split(",") if models else [m for m in cfg.models if m != "mock"]
    for name in names:
        spec = cfg.models[name]
        key = env.get(spec.provider)
        if key and not (os.environ.get(key) or (spec.provider == "gemini" and os.environ.get("GOOGLE_API_KEY"))):
            typer.echo(f"{name:8s} SKIP   {key} not set")
            continue
        try:
            client = make_client(spec)
            req = Request(system=cfg.prompts["system"], temperature=0.0, max_tokens=cfg.max_output_tokens,
                          user_text=cfg.prompts["zero_shot"].format(
                              session_id="check000", condition="transcript",
                              payload="[Poetry reading task] Transcript:\nThe light comes slowly through the window.\n\n"
                                      "[Image-description task] Transcript:\nTwo people at a table, talking."))
            r = client.complete(req)
            if r.error:
                typer.echo(f"{name:8s} FAIL   {spec.model_id}: {r.error[:160]}")
            elif r.parsed is None:
                typer.echo(f"{name:8s} WARN   {spec.model_id}: replied but not JSON: {r.raw_text[:120]!r}")
            else:
                typer.echo(f"{name:8s} OK     {spec.model_id}: risk_score={r.risk_score} in {r.latency_s:.1f}s "
                           f"(tokens {r.input_tokens}/{r.output_tokens})")
        except Exception as e:
            typer.echo(f"{name:8s} FAIL   {spec.model_id}: {type(e).__name__}: {str(e)[:160]}")


@app.command()
def analyze(runs: Path, out: Path = Path("results")):
    """Result tables: AUROC/CI, sens@90%spec, calibration, transfer gap, rank agreement, reliability."""
    from .analysis.report import write_all

    for name, p in write_all(runs, out).items():
        typer.echo(f"{name}: {p}")


@app.command()
def reference(manifest: Path, cache: Path = Path("cache"), out: Path = Path("results"), dataset: str = "behavioral"):
    """Classical SVM / gradient-boosting reference on the numeric feature sheet (nested CV)."""
    from .analysis import metrics as M
    from .analysis.reference import feature_frame, nested_cv_scores

    df = feature_frame(load_manifest(manifest), cache)
    scores = nested_cv_scores(df, dataset=dataset)
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / f"reference_scores_{dataset}.csv", index=False)
    for col in [c for c in scores.columns if c.startswith("score_")]:
        a = M.auroc_ci(scores["label"], scores[col])
        typer.echo(f"{col}: AUROC {a['auroc']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}]")


if __name__ == "__main__":
    app()
