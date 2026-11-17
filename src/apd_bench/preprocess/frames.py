"""Evenly spaced frame sampling from a video segment (for models that take images, not video)."""
from __future__ import annotations

import base64
import subprocess
import tempfile
from pathlib import Path


def sample_frames(video_path: str | Path, start: float, end: float, n: int = 8, max_side: int = 768) -> list[bytes]:
    """Return n JPEG frames as bytes, evenly spaced across [start, end]. Uses ffmpeg."""
    if end <= start:
        return []
    frames = []
    with tempfile.TemporaryDirectory() as td:
        for i in range(n):
            t = start + (end - start) * (i + 0.5) / n
            out = Path(td) / f"f{i:02d}.jpg"
            cmd = [
                "ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video_path),
                "-frames:v", "1", "-vf", f"scale='min({max_side},iw)':-2", "-q:v", "4", str(out),
            ]
            subprocess.run(cmd, check=True)
            frames.append(out.read_bytes())
    return frames


def extract_audio(video_path: str | Path, out_wav: str | Path) -> Path:
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(video_path), "-ac", "1", "-ar", "16000", str(out_wav)]
    subprocess.run(cmd, check=True)
    return Path(out_wav)


def b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("utf-8")
