"""Run grid: model × dataset × condition × prompt_mode × repeat, over eval sessions.

Every call appends one JSON line to <out_dir>/runs.jsonl. Re-running skips rows
already present (idempotent), so an interrupted grid resumes where it stopped.
Few-shot examples come only from split == "calibration" sessions of the same
dataset, so eval labels never enter a prompt.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from tqdm import tqdm

from .config import BenchConfig
from .manifest import Session, manifest_hash
from .models import Request, UnsupportedCondition, make_client
from .preprocess.feature_sheet import SessionInputs, build_inputs

N_FEWSHOT = 4


def _payload(inp: SessionInputs, condition: str) -> str:
    if condition == "transcript":
        return inp.transcript
    if condition == "feature_sheet":
        return inp.feature_sheet
    if condition == "raw_media":
        # The media itself travels in Request.frames / video_path; the text part is the transcript.
        return "Media attached. " + inp.transcript
    raise ValueError(condition)


def _fewshot_block(examples: list[tuple[SessionInputs, int]], condition: str) -> str:
    chunks = []
    for i, (inp, label) in enumerate(examples, 1):
        chunks.append(
            f"Example {i}:\n{_payload(inp, condition)}\n"
            f"Clinician label: {'elevated' if label else 'control'}"
        )
    return "\n\n".join(chunks)


def _media_for(spec, inp: SessionInputs, condition: str, cfg: BenchConfig) -> dict:
    if condition != "raw_media":
        return {}
    if spec.raw_media_kind == "video":
        return {"video_path": inp.video_path}
    if spec.raw_media_kind == "audio":
        return {"audio_path": inp.audio_path}
    frames = [b for task in ("poetry", "image_description") for b in inp.frames.get(task, [])]
    return {"frames": frames}


def _done_keys(log_path: Path) -> set[tuple]:
    if not log_path.exists():
        return set()
    keys = set()
    with log_path.open() as f:
        for line in f:
            r = json.loads(line)
            keys.add((r["model"], r["session_id"], r["condition"], r["prompt_mode"], r["repeat"]))
    return keys


def run_grid(
    cfg: BenchConfig,
    sessions: list[Session],
    manifest_path: str | Path,
    cache_dir: str | Path,
    out_dir: str | Path,
    models: Optional[Iterable[str]] = None,
    conditions: Optional[Iterable[str]] = None,
    prompt_modes: Optional[Iterable[str]] = None,
    repeats: Optional[int] = None,
    skip_media: bool = False,
    seed: int = 0,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "runs.jsonl"
    done = _done_keys(log_path)
    mhash = manifest_hash(manifest_path)

    models = list(models or cfg.models)
    conditions = list(conditions or cfg.conditions)
    prompt_modes = list(prompt_modes or cfg.prompt_modes)
    repeats = repeats or cfg.repeats

    # Build inputs once per session (cached on disk)
    inputs: dict[str, SessionInputs] = {}
    for s in tqdm(sessions, desc="inputs"):
        inputs[s.session_id] = build_inputs(
            s, cache_dir, cfg.frames_per_task, cfg.frame_max_side, skip_media=skip_media
        )

    rng = random.Random(seed)
    calib = {ds: [s for s in sessions if s.split == "calibration" and s.dataset == ds] for ds in ("behavioral", "synthetic")}
    evals = [s for s in sessions if s.split == "eval"]

    system = cfg.prompts["system"]
    with log_path.open("a") as log:
        for mname in models:
            spec = cfg.models[mname]
            client = make_client(spec)
            for cond in conditions:
                if cond not in spec.supports:
                    continue
                for pmode in prompt_modes:
                    for s in tqdm(evals, desc=f"{mname}/{cond}/{pmode}", leave=False):
                        inp = inputs[s.session_id]
                        # Few-shot: balanced sample from calibration split of the same dataset, fixed per session
                        if pmode == "few_shot":
                            pool = calib[s.dataset]
                            pos = [c for c in pool if c.label == 1]
                            neg = [c for c in pool if c.label == 0]
                            k = min(N_FEWSHOT // 2, len(pos), len(neg))
                            srng = random.Random(f"{seed}:{s.session_id}")
                            ex = srng.sample(pos, k) + srng.sample(neg, k)
                            srng.shuffle(ex)
                            examples = [(inputs[c.session_id], c.label) for c in ex]
                            user_text = cfg.prompts["few_shot"].format(
                                n_examples=len(examples), examples=_fewshot_block(examples, cond),
                                session_id=s.session_id, condition=cond, payload=_payload(inp, cond),
                            )
                        else:
                            user_text = cfg.prompts["zero_shot"].format(
                                session_id=s.session_id, condition=cond, payload=_payload(inp, cond)
                            )
                        for rep in range(repeats):
                            key = (mname, s.session_id, cond, pmode, rep)
                            if key in done:
                                continue
                            req = Request(
                                system=system, user_text=user_text,
                                temperature=cfg.temperature, max_tokens=cfg.max_output_tokens,
                                **_media_for(spec, inp, cond, cfg),
                            )
                            row = {
                                "ts": datetime.now(timezone.utc).isoformat(),
                                "model": mname, "provider": spec.provider, "model_id": spec.model_id,
                                "session_id": s.session_id, "dataset": s.dataset, "label": s.label,
                                "condition": cond, "prompt_mode": pmode, "repeat": rep,
                                "manifest_hash": mhash, "prompt_hash": cfg.prompt_hash,
                                "temperature": cfg.temperature,
                            }
                            try:
                                resp = client.complete(req)
                                row.update({
                                    "status": "error" if resp.error else ("refused" if resp.refused else ("unparsed" if resp.parsed is None else "ok")),
                                    "risk_score": resp.risk_score,
                                    "risk_label": (resp.parsed or {}).get("risk_label"),
                                    "task_most_informative": (resp.parsed or {}).get("task_most_informative"),
                                    "evidence": (resp.parsed or {}).get("evidence"),
                                    "raw_text": resp.raw_text, "error": resp.error,
                                    "latency_s": resp.latency_s,
                                    "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
                                })
                            except UnsupportedCondition as e:
                                row.update({"status": "unsupported", "risk_score": None, "error": str(e)})
                            log.write(json.dumps(row, ensure_ascii=False) + "\n")
                            log.flush()
                            done.add(key)
    return log_path
