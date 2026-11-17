"""Build, per session, the three input conditions:

    transcript    -> text
    feature_sheet -> text (face + voice + timing + language per task, numbers only)
    raw_media     -> video path, audio path, and sampled frames per task

Everything is cached to <cache_dir>/<session_id>/ so model runs never recompute.
Nothing here sees the label.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..manifest import Session
from . import face, frames as fr, text as tx, voice


@dataclass
class SessionInputs:
    session_id: str
    transcript: str                                   # full text, both tasks
    feature_sheet: str                                # text sheet
    features: dict[str, dict[str, float]]             # numeric features per task (for the classical reference)
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    frames: dict[str, list[bytes]] = field(default_factory=dict)   # task -> jpeg bytes


def _task_transcript_block(name: str, text: str) -> str:
    title = "Poetry reading task" if name == "poetry" else "Image-description task"
    return f"[{title}] Transcript:\n{text.strip() or '(no speech detected)'}"


def build_inputs(
    s: Session,
    cache_dir: str | Path,
    frames_per_task: int = 8,
    frame_max_side: int = 768,
    whisper_model: str = "medium",
    skip_media: bool = False,
) -> SessionInputs:
    cache = Path(cache_dir) / s.session_id
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / "inputs.json"
    if cached.exists():
        d = json.loads(cached.read_text())
        frames = {}
        for task in s.tasks:
            fdir = cache / f"frames_{task}"
            if fdir.exists():
                frames[task] = [p.read_bytes() for p in sorted(fdir.glob("*.jpg"))]
        return SessionInputs(**d, frames=frames)

    # --- audio ---------------------------------------------------------------
    audio_path = s.audio_path
    if audio_path is None and s.video_path and not skip_media:
        audio_path = str(fr.extract_audio(s.video_path, cache / "audio.wav"))

    # --- transcript ----------------------------------------------------------
    if s.transcript_path:
        tr = tx.load_transcript(s.transcript_path)
    elif audio_path and not skip_media:
        tr = tx.transcribe(audio_path, model_name=whisper_model, language=s.language)
        (cache / "transcript.json").write_text(json.dumps(tr, ensure_ascii=False, indent=1))
    else:
        tr = {"text": "", "words": []}

    # --- OpenFace ------------------------------------------------------------
    of = face.load_openface(s.openface_csv) if s.openface_csv else None

    sheet_blocks, transcript_blocks, features = [], [], {}
    for task, (t0, t1) in s.tasks.items():
        seg_text = tx.segment_text(tr, t0, t1)
        transcript_blocks.append(_task_transcript_block(task, seg_text))

        feats: dict[str, float] = {}
        lines = [f"[{'Poetry reading task' if task == 'poetry' else 'Image-description task'}] duration {t1 - t0:.1f} s"]

        if of is not None:
            fstats = face.summarise_face(of, t0, t1)
            feats.update({f"face_{k}": v for k, v in fstats.items()})
            lines.append(face.describe_face(fstats))
        else:
            lines.append("Face: not available.")

        if audio_path and not skip_media and t1 > t0:
            try:
                vf = voice.extract_egemaps(audio_path, t0, t1)
                vf["pause_ratio"] = voice.pause_ratio(audio_path, t0, t1)
            except Exception as e:  # missing opensmile/librosa or bad audio: record, don't crash
                vf = {"_error": str(e)}
            feats.update({f"voice_{k}": v for k, v in vf.items() if isinstance(v, float)})
            lines.append(voice.describe_voice(vf))
        else:
            lines.append("Voice: not available.")

        latency = tx.response_latency(tr, s.stimulus_onset, t1) if task == "image_description" else None
        tfeat = tx.text_features(seg_text, duration=(t1 - t0) if t1 > t0 else None)
        if latency is not None:
            tfeat["response_latency_s"] = latency
        feats.update({f"text_{k}": v for k, v in tfeat.items()})
        lines.append(tx.describe_text(tfeat, latency))
        lines.append(f"Transcript: {seg_text.strip() or '(no speech detected)'}")

        sheet_blocks.append("\n".join(lines))
        features[task] = feats

    # --- frames --------------------------------------------------------------
    frames: dict[str, list[bytes]] = {}
    if s.video_path and not skip_media:
        for task, (t0, t1) in s.tasks.items():
            fs = fr.sample_frames(s.video_path, t0, t1, n=frames_per_task, max_side=frame_max_side)
            frames[task] = fs
            fdir = cache / f"frames_{task}"
            fdir.mkdir(exist_ok=True)
            for i, b in enumerate(fs):
                (fdir / f"{i:02d}.jpg").write_bytes(b)

    inputs = SessionInputs(
        session_id=s.session_id,
        transcript="\n\n".join(transcript_blocks),
        feature_sheet="\n\n".join(sheet_blocks),
        features=features,
        video_path=s.video_path,
        audio_path=audio_path,
        frames=frames,
    )
    cached.write_text(json.dumps({k: v for k, v in inputs.__dict__.items() if k != "frames"}, indent=1))
    return inputs
