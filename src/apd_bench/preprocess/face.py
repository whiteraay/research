"""Facial features from OpenFace FeatureExtraction CSV output.

OpenFace is run separately (it is a C++ binary):
    FeatureExtraction -f session.mp4 -aus -pose -gaze -out_dir openface/
This module summarises its per-frame CSV per task segment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# AUs most cited for low affect / distress in the literature (see research plan, RQ3)
KEY_AUS = ["AU01", "AU04", "AU06", "AU12", "AU15", "AU17"]


def load_openface(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def summarise_face(df: pd.DataFrame, start: float, end: float) -> dict[str, float]:
    """Summary statistics of AU intensity, head motion and gaze for one task segment."""
    seg = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    if "success" in seg.columns:
        detect_rate = float(seg["success"].mean()) if len(seg) else 0.0
        seg = seg[seg["success"] == 1]
    else:
        detect_rate = 1.0
    out: dict[str, float] = {"face_detect_rate": detect_rate, "n_frames": float(len(seg))}
    if seg.empty:
        return out
    for au in KEY_AUS:
        r = f"{au}_r"
        if r in seg.columns:
            v = seg[r].to_numpy()
            out[f"{au}_mean"] = float(v.mean())
            out[f"{au}_std"] = float(v.std())
            out[f"{au}_active_frac"] = float((v > 1.0).mean())
    # head motion: frame-to-frame rotation magnitude
    rot = [c for c in ("pose_Rx", "pose_Ry", "pose_Rz") if c in seg.columns]
    if rot:
        d = np.diff(seg[rot].to_numpy(), axis=0)
        out["head_motion_mean"] = float(np.linalg.norm(d, axis=1).mean()) if len(d) else 0.0
    gaze = [c for c in ("gaze_angle_x", "gaze_angle_y") if c in seg.columns]
    if gaze:
        out["gaze_var"] = float(seg[gaze].var().sum())
    return out


def describe_face(stats: dict[str, float]) -> str:
    """Plain-language line for the feature sheet. Numbers only; no interpretation."""
    if stats.get("n_frames", 0) == 0:
        return "Face: no frames with a detected face."
    parts = [f"face detected in {stats['face_detect_rate']:.0%} of frames"]
    for au in KEY_AUS:
        if f"{au}_mean" in stats:
            parts.append(f"{au} mean {stats[f'{au}_mean']:.2f} (active {stats[f'{au}_active_frac']:.0%})")
    if "head_motion_mean" in stats:
        parts.append(f"head motion {stats['head_motion_mean']:.3f} rad/frame")
    if "gaze_var" in stats:
        parts.append(f"gaze variance {stats['gaze_var']:.3f}")
    return "Face: " + "; ".join(parts) + "."
