"""Acoustic features: eGeMAPS functionals via openSMILE, plus pause ratio."""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Subset of eGeMAPSv02 functionals used on the feature sheet. Full set is stored too.
SHEET_FEATURES = {
    "F0semitoneFrom27.5Hz_sma3nz_amean": "F0 mean (semitones)",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": "F0 variability (norm. SD)",
    "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2": "F0 range (2-98 pct)",
    "loudness_sma3_amean": "loudness mean",
    "loudness_sma3_stddevNorm": "loudness variability",
    "jitterLocal_sma3nz_amean": "jitter",
    "shimmerLocaldB_sma3nz_amean": "shimmer (dB)",
    "HNRdBACF_sma3nz_amean": "harmonics-to-noise (dB)",
    "VoicedSegmentsPerSec": "voiced segments/s",
    "MeanUnvoicedSegmentLength": "mean pause length (s)",
    "mfcc1_sma3_amean": "MFCC1 mean",
}


def extract_egemaps(audio_path: str | Path, start: float, end: float) -> dict[str, float]:
    """Run openSMILE eGeMAPSv02 on one segment. Requires `pip install opensmile`."""
    import opensmile  # imported lazily so the package works without it

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    df = smile.process_file(str(audio_path), start=start, end=end)
    return {k: float(v) for k, v in df.iloc[0].items()}


def pause_ratio(audio_path: str | Path, start: float, end: float, top_db: float = 35.0) -> float:
    """Fraction of the segment that is silence (librosa energy-based split)."""
    import librosa

    y, sr = librosa.load(str(audio_path), sr=16000, offset=start, duration=max(end - start, 0.1))
    if len(y) == 0:
        return 1.0
    intervals = librosa.effects.split(y, top_db=top_db)
    voiced = sum(e - s for s, e in intervals)
    return float(1.0 - voiced / len(y))


def describe_voice(feats: dict[str, float]) -> str:
    parts = []
    for key, label in SHEET_FEATURES.items():
        if key in feats and np.isfinite(feats[key]):
            parts.append(f"{label} {feats[key]:.3g}")
    if "pause_ratio" in feats:
        parts.append(f"pause ratio {feats['pause_ratio']:.2f}")
    if "speech_rate_wps" in feats:
        parts.append(f"speech rate {feats['speech_rate_wps']:.2f} words/s")
    return "Voice: " + ("; ".join(parts) if parts else "not available") + "."
