"""Session manifest: one row per recorded session, for both datasets.

Columns (CSV):
    session_id      unique study id (never a name or MRN)
    dataset         "behavioral" | "synthetic"
    label           1 = elevated-risk cohort, 0 = clinical-control cohort
    video_path      path to the full session video (may be empty for transcript-only sets)
    audio_path      path to session audio (wav); extracted from video if empty
    poetry_start    seconds, start of the poetry task inside the recording
    poetry_end      seconds
    image_start     seconds, start of the image-description task
    image_end       seconds
    stimulus_onset  seconds, when the image was shown (for response latency)
    transcript_path path to a plain-text or JSON transcript (optional; Whisper fills it)
    openface_csv    path to OpenFace FeatureExtraction output (optional)
    split           "calibration" (few-shot examples only) | "eval"
    depression_score, age, sex, language   covariates (optional)
    synth_method    for synthetic rows: "actor" | "tts_avatar" | "llm_text" (optional)
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, field_validator


class Session(BaseModel):
    session_id: str
    dataset: Literal["behavioral", "synthetic"]
    label: int
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    poetry_start: float = 0.0
    poetry_end: float = 0.0
    image_start: float = 0.0
    image_end: float = 0.0
    stimulus_onset: Optional[float] = None
    transcript_path: Optional[str] = None
    openface_csv: Optional[str] = None
    split: Literal["calibration", "eval"] = "eval"
    depression_score: Optional[float] = None
    age: Optional[float] = None
    sex: Optional[str] = None
    language: Optional[str] = None
    synth_method: Optional[str] = None

    @field_validator("label")
    @classmethod
    def _binary(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("label must be 0 or 1")
        return v

    @property
    def tasks(self) -> dict[str, tuple[float, float]]:
        return {
            "poetry": (self.poetry_start, self.poetry_end),
            "image_description": (self.image_start, self.image_end),
        }


def _clean(v):
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return None
    return v


def load_manifest(path: str | Path) -> list[Session]:
    df = pd.read_csv(path, dtype={"session_id": str})
    sessions = []
    for rec in df.to_dict(orient="records"):
        rec = {k: _clean(v) for k, v in rec.items()}
        sessions.append(Session(**{k: v for k, v in rec.items() if v is not None}))
    ids = [s.session_id for s in sessions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate session_id in manifest")
    return sessions


def manifest_hash(path: str | Path) -> str:
    """Content hash recorded in every run log, so a frozen dataset is provably unchanged."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
