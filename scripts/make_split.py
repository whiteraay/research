#!/usr/bin/env python
"""Make the one shared train/test split from the lab's Experimental/ and Control/ folders.

Run once by the coordinator, in Colab, on the MOUNTED Drive folder:

    python scripts/make_split.py --src /content/drive/MyDrive/video --dry-run          # look first
    python scripts/make_split.py --src /content/drive/MyDrive/video \
        --out /content/drive/MyDrive/apd_benchmark/data \
        --private-out /content/drive/MyDrive/apd_private --test-frac 0.3 --seed 0

Input layout (either works):
    <src>/Experimental/<file>.mp4                 participant = filename prefix (see --participant-regex)
    <src>/Experimental/<participant>/<file>.mp4   participant = sub-folder name

Output (shared read-only with operators):
    <out>/train/Experimental/<sid>.<ext>, <out>/train/Control/<sid>.<ext>
    <out>/train/train_labels.csv        session_id, cohort, participant_id
    <out>/test/<sid>.<ext>              anonymised, flat, no labels
Private (coordinator only; never inside <out>):
    <private-out>/test_labels.csv, <private-out>/split_manifest.csv

Copies are atomic (.partial then rename) and skipped if already present, so an interrupted
run (Colab disconnect) can simply be started again with the same arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
COHORTS = (("elevated", "Experimental"), ("control", "Control"))


def scan(src: Path, regex: str) -> tuple[pd.DataFrame, Counter]:
    rows, other = [], Counter()
    for cohort, folder in COHORTS:
        d = src / folder
        if not d.is_dir():
            raise SystemExit(f"missing folder {d}")
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.name.startswith("."):
                continue
            if p.suffix.lower() not in VIDEO_EXT:
                other[p.suffix.lower() or "(none)"] += 1
                continue
            rel = p.relative_to(d)
            if len(rel.parts) > 1:
                pid = rel.parts[0]
            else:
                m = re.match(regex, p.stem)
                pid = m.group(1) if m else p.stem
            rows.append({"orig_path": str(p), "orig_rel": f"{folder}/{rel}", "cohort": cohort,
                         "participant_id": pid, "bytes": p.stat().st_size})
    return pd.DataFrame(rows), other


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--private-out", default=None, help="where test_labels.csv and split_manifest.csv go (default: <out>/../private)")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--participant-regex", default=r"^([^_]+)_", help="group 1 = participant id; default = prefix before first _")
    ap.add_argument("--dry-run", action="store_true", help="show counts and grouping, copy nothing")
    ap.add_argument("--show", type=int, default=8, help="examples per cohort in dry-run")
    a = ap.parse_args()

    src = Path(a.src)
    df, other = scan(src, a.participant_regex)
    if df.empty:
        raise SystemExit("no videos found")

    per = df.groupby("cohort").agg(videos=("orig_path", "count"), participants=("participant_id", "nunique"),
                                   gb=("bytes", lambda b: round(b.sum() / 1e9, 2)))
    print(per.to_string(), "\n")
    if other:
        print("non-video files ignored:", dict(other))
    shared = set(df[df.cohort == "elevated"].participant_id) & set(df[df.cohort == "control"].participant_id)
    if shared:
        print(f"WARNING: {len(shared)} participant id(s) appear in BOTH cohorts, e.g. {sorted(shared)[:5]}. "
              "Your --participant-regex probably does not capture the real participant id.")
    multi = (df.groupby("participant_id").size() > 1).sum()
    print(f"participants with >1 video: {multi}")

    if a.dry_run:
        for cohort, g in df.groupby("cohort"):
            print(f"\n{cohort}: first {a.show} files -> participant id")
            for r in g.head(a.show).itertuples():
                print(f"  {r.orig_rel:60s} -> {r.participant_id}")
        print("\nDry run only. If the participant ids look wrong, pass --participant-regex.")
        return

    out = Path(a.out)
    private = Path(a.private_out) if a.private_out else out.parent / "private"
    if private.resolve() == out.resolve() or out.resolve() in private.resolve().parents:
        raise SystemExit("--private-out must not be inside --out")

    df["session_id"] = ["s" + hashlib.sha1(f"{a.seed}:{p}".encode()).hexdigest()[:8] for p in df.orig_rel]
    df["participant_hash"] = ["p" + hashlib.sha1(f"{a.seed}:{c}:{p}".encode()).hexdigest()[:6]
                              for c, p in zip(df.cohort, df.participant_id)]

    rng = random.Random(a.seed)
    df["split"] = "train"
    for cohort, g in df.groupby("cohort"):
        pids = sorted(g.participant_id.unique())
        rng.shuffle(pids)
        n_test = max(1, round(len(pids) * a.test_frac))
        df.loc[(df.cohort == cohort) & df.participant_id.isin(set(pids[:n_test])), "split"] = "test"

    for f in ("Experimental", "Control"):
        (out / "train" / f).mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    dests, t0, done_bytes = [], time.time(), 0
    total_bytes = int(df.bytes.sum())
    for i, r in enumerate(df.itertuples(), 1):
        ext = Path(r.orig_path).suffix.lower()
        dest = (out / "train" / ("Experimental" if r.cohort == "elevated" else "Control") / f"{r.session_id}{ext}"
                if r.split == "train" else out / "test" / f"{r.session_id}{ext}")
        if not (dest.exists() and dest.stat().st_size == r.bytes):
            tmp = dest.with_name(dest.name + ".partial")
            shutil.copyfile(r.orig_path, tmp)
            tmp.rename(dest)
        dests.append(str(dest.relative_to(out)))
        done_bytes += r.bytes
        if i % 10 == 0 or i == len(df):
            el = time.time() - t0
            print(f"  {i}/{len(df)} files, {done_bytes / 1e9:.2f}/{total_bytes / 1e9:.2f} GB, {el / 60:.1f} min", flush=True)
    df["dest_rel"] = dests

    tr, te = df[df.split == "train"], df[df.split == "test"]
    tr[["session_id", "cohort", "participant_hash"]].rename(columns={"participant_hash": "participant_id"}) \
        .to_csv(out / "train" / "train_labels.csv", index=False)
    te[["session_id", "cohort", "participant_hash"]].rename(columns={"participant_hash": "participant_id"}) \
        .to_csv(private / "test_labels.csv", index=False)
    df.to_csv(private / "split_manifest.csv", index=False)

    print("\n", df.groupby(["split", "cohort"]).agg(sessions=("session_id", "count"), participants=("participant_id", "nunique")))
    print(f"\nshared:  {out}/train, {out}/test")
    print(f"PRIVATE: {private}/test_labels.csv, {private}/split_manifest.csv  (never share these)")


if __name__ == "__main__":
    sys.exit(main())
