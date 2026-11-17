#!/usr/bin/env python
"""Shared recording-condition audit: one row per video, label-blind.

    python scripts/audit_videos.py --data data --out data/audit.csv

Columns: session_id, path, duration_s, width, height, fps, mean_luminance, luminance_std,
         face_detect_rate, audio_snr_db, speech_fraction, audio_rms_db,
         duration_tertile, luminance_tertile, snr_tertile

Lighting   = mean grey level (0-255) of N frames sampled evenly by seeking (no full decode).
Face rate  = fraction of those frames with a face (OpenCV Haar cascade; coarse but dependency-free).
Resumable: progress is kept in <out>.partial.csv; re-run the same command after a Colab disconnect.
Audio SNR  = 10*log10(P_speech / P_noise) with speech/noise frames split by an energy threshold
             (WebRTC-VAD-free approximation); speech_fraction = share of frames above threshold.
Tertiles   = computed over ALL rows (train + test together) so every agent gets identical strata.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    v = next((s for s in j["streams"] if s["codec_type"] == "video"), {})
    num, den = (v.get("avg_frame_rate", "0/1").split("/") + ["1"])[:2]
    fps = float(num) / float(den) if float(den) else 0.0
    return {"duration_s": float(j["format"].get("duration", 0)), "width": v.get("width"), "height": v.get("height"), "fps": round(fps, 2)}


def _cascade():
    import cv2
    casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if casc.empty():
        # some headless wheels ship without cascades; fetch once next to this script
        local = Path(__file__).with_name("haarcascade_frontalface_default.xml")
        if not local.exists():
            import urllib.request
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml", local)
        casc = cv2.CascadeClassifier(str(local))
    return casc


def frame_stats(path: Path, duration: float, n: int = 30) -> dict:
    """Seek to n evenly spaced times (fast: no full decode), then luminance + face detection on those frames."""
    import cv2
    nan = {"mean_luminance": float("nan"), "luminance_std": float("nan"), "face_detect_rate": float("nan"), "frames_sampled": 0}
    if not duration or duration <= 0:
        return nan
    lum, faces = [], 0
    casc = _cascade()
    with tempfile.TemporaryDirectory() as td:
        for i in range(n):
            t = duration * (i + 0.5) / n
            f = Path(td) / f"f{i:02d}.jpg"
            subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(path), "-frames:v", "1",
                            "-vf", "scale=480:-2", "-y", str(f)], check=False)
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) if f.exists() else None
            if img is None:
                continue
            lum.append(float(img.mean()))
            if not casc.empty() and len(casc.detectMultiScale(img, 1.2, 4)) > 0:
                faces += 1
    if not lum:
        return nan
    return {"mean_luminance": float(np.mean(lum)), "luminance_std": float(np.std(lum)),
            "face_detect_rate": faces / len(lum) if not casc.empty() else float("nan"), "frames_sampled": len(lum)}


def audio_stats(path: Path) -> dict:
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.wav"
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "wav", str(wav)],
                           capture_output=True)
        if r.returncode != 0 or not wav.exists():
            return {"audio_snr_db": float("nan"), "speech_fraction": float("nan"), "audio_rms_db": float("nan")}
        import soundfile as sf
        y, sr = sf.read(str(wav))
    if len(y) < sr:
        return {"audio_snr_db": float("nan"), "speech_fraction": float("nan"), "audio_rms_db": float("nan")}
    frame = int(0.03 * sr)
    n = len(y) // frame
    e = (y[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12
    edb = 10 * np.log10(e)
    thr = np.percentile(edb, 30) + 6.0          # speech = frames well above the quiet floor
    sp, nz = edb[edb > thr], edb[edb <= thr]
    if len(sp) < 5 or len(nz) < 5:              # near-constant signal: fall back to top vs bottom deciles
        lo, hi = np.percentile(edb, [10, 90])
        sp, nz = edb[edb >= hi], edb[edb <= lo]
    pw = lambda x: 10 * np.log10(np.mean(10 ** (x / 10)))
    snr = pw(sp) - pw(nz) if len(sp) and len(nz) else float("nan")
    return {"audio_snr_db": float(snr), "speech_fraction": float((edb > thr).mean()), "audio_rms_db": float(10 * np.log10(np.mean(e)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data/audit.csv")
    ap.add_argument("--frames", type=int, default=30, help="frames sampled per video for lighting/face stats")
    a = ap.parse_args()
    data = Path(a.data)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vids = sorted(p for p in list((data / "train").rglob("*")) + list((data / "test").rglob("*"))
                  if p.is_file() and p.suffix.lower() in VIDEO_EXT)

    # resumable: rows already measured are kept in <out>.partial.csv
    partial = out.with_name(out.stem + ".partial.csv")
    done = pd.read_csv(partial, dtype={"session_id": str}) if partial.exists() else pd.DataFrame()
    done_ids = set(done["session_id"]) if len(done) else set()
    todo = [p for p in vids if p.stem not in done_ids]
    print(f"{len(vids)} videos, {len(done_ids)} already audited, {len(todo)} to go")

    for i, p in enumerate(todo, 1):
        row = {"session_id": p.stem, "path": str(p.relative_to(data))}
        errs = []
        for name, step in (("probe", lambda: row.update(ffprobe(p))),
                           ("frames", lambda: row.update(frame_stats(p, row.get("duration_s", 0), a.frames))),
                           ("audio", lambda: row.update(audio_stats(p)))):
            try:
                step()
            except Exception as e:  # one failing stat must not blank the others
                errs.append(f"{name}: {type(e).__name__}: {str(e).splitlines()[0][:120] if str(e) else ''}")
        row["error"] = " | ".join(errs)
        pd.DataFrame([row]).to_csv(partial, mode="a", header=not partial.exists(), index=False)
        print(f"[{i}/{len(todo)}] {p.stem}: {row.get('duration_s', float('nan')):.0f}s "
              f"lum={row.get('mean_luminance', float('nan')):.0f} face={row.get('face_detect_rate', float('nan')):.2f} "
              f"snr={row.get('audio_snr_db', float('nan')):.1f}dB{'  ERR ' + row['error'] if errs else ''}", flush=True)

    df = pd.read_csv(partial, dtype={"session_id": str}).drop_duplicates("session_id", keep="last")
    df = df[df.session_id.isin({p.stem for p in vids})]
    for col, name in (("duration_s", "duration_tertile"), ("mean_luminance", "luminance_tertile"), ("audio_snr_db", "snr_tertile")):
        if col in df and df[col].notna().sum() >= 3:
            r = df[col].rank(method="first")
            df[name] = pd.qcut(r, 3, labels=["low", "mid", "high"]).astype(str).where(df[col].notna(), "")
    df.to_csv(out, index=False)
    n_err = int((df["error"].fillna("") != "").sum())
    print(f"\nwrote {out} ({len(df)} videos, {n_err} with errors)")


if __name__ == "__main__":
    main()
