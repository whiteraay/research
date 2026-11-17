"""Deterministic mock so the whole pipeline runs without keys or data.

It scores from crude transcript cues plus a hash-seeded jitter, so AUROC on the
toy manifest is above chance but not perfect. Never use its outputs as results.
"""
from __future__ import annotations

import hashlib
import json
import re

from .base import BaseClient, Request

_NEG = ("tired", "empty", "alone", "hopeless", "nothing", "dark", "heavy", "pain")


class MockClient(BaseClient):
    provider = "mock"

    def _call(self, req: Request) -> tuple[str, dict]:
        text = req.user_text.lower()
        neg = sum(text.count(w) for w in _NEG)
        words = max(len(re.findall(r"[a-z']+", text)), 1)
        base = 35.0 + 400.0 * neg / words
        m = re.search(r"response latency ([\d.]+) s", text)
        if m:
            base += 4.0 * float(m.group(1))
        seed = int(hashlib.md5(req.user_text.encode()).hexdigest(), 16) % 21 - 10
        score = min(max(base + seed, 0.0), 100.0)
        out = {
            "risk_score": round(score, 1),
            "risk_label": "elevated" if score >= 50 else "control",
            "confidence": "low",
            "evidence": {
                "facial": "not available" if not req.frames else "frames received",
                "vocal": "not available",
                "timing": m.group(0) if m else "not available",
                "language": f"{neg} negative-lexicon hits in {words} words",
            },
            "task_most_informative": "equal",
        }
        return json.dumps(out), {"input": len(text) // 4, "output": 80}
