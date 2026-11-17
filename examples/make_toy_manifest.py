"""Generate a tiny transcript-only toy dataset to exercise the pipeline.

Not research data: transcripts are templated, labels are assigned by template.
Usage: python examples/make_toy_manifest.py examples/toy
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

CONTROL = [
    "The light comes slowly through the window and the morning starts again. I read it fine, it's a calm poem.",
    "I see two people at a table, maybe having coffee. They look like they're talking about their day. Probably friends.",
]
ELEVATED = [
    "The light comes slowly through the window. Everything feels heavy and tired. I don't know, it's empty.",
    "I see someone alone at a table. Nothing much is happening. They look lost, like there's no point to it.",
]


def make(out: Path, n_behav: int = 40, n_synth: int = 40, seed: int = 0):
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "transcripts").mkdir(exist_ok=True)
    rows = []
    for dataset, n in (("behavioral", n_behav), ("synthetic", n_synth)):
        for i in range(n):
            label = i % 2
            sid = f"{dataset[:3]}{i:03d}"
            src = ELEVATED if label else CONTROL
            # behavioral: noisier (mix lines); synthetic: clean templates
            if dataset == "behavioral" and rng.random() < 0.3:
                src = [src[0], (CONTROL if label else ELEVATED)[1]]
            words, t = [], 0.0
            for seg_i, line in enumerate(src):
                t = 5.0 if seg_i == 0 else 40.0 + rng.uniform(0.5, 4.0 if label else 1.5)
                for w in line.split():
                    words.append({"word": w, "start": round(t, 2), "end": round(t + 0.3, 2)})
                    t += 0.35
            tp = out / "transcripts" / f"{sid}.json"
            tp.write_text(json.dumps({"text": " ".join(w["word"] for w in words), "words": words}))
            rows.append({
                "session_id": sid, "dataset": dataset, "label": label,
                "poetry_start": 0, "poetry_end": 30, "image_start": 35, "image_end": 70, "stimulus_onset": 40,
                "transcript_path": str(tp), "split": "calibration" if i < 6 else "eval",
            })
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    print(f"wrote {out / 'manifest.csv'} ({len(rows)} sessions)")


if __name__ == "__main__":
    make(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("examples/toy"))
