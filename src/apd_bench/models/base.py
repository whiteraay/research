"""Unified client interface. Each backend turns a Request into a Response.

A Request carries the system prompt, the user text, and optional media
(frames as JPEG bytes, or a video/audio file path). Backends that cannot
take a given media kind raise UnsupportedCondition, which the runner logs.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional


class UnsupportedCondition(Exception):
    pass


@dataclass
class Request:
    system: str
    user_text: str
    frames: list[bytes] = field(default_factory=list)   # JPEG bytes
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 600


@dataclass
class Response:
    raw_text: str
    parsed: Optional[dict]          # None if the JSON could not be parsed
    refused: bool                   # provider refusal / content filter / empty output
    latency_s: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    error: Optional[str] = None

    @property
    def risk_score(self) -> Optional[float]:
        if not self.parsed:
            return None
        try:
            v = float(self.parsed.get("risk_score"))
            return min(max(v, 0.0), 100.0)
        except (TypeError, ValueError):
            return None


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(text: str) -> Optional[dict]:
    """Tolerant parse: strips code fences and leading prose."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = _JSON_RE.search(t)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


class BaseClient:
    provider = "base"

    def __init__(self, model_id: str):
        self.model_id = model_id

    def complete(self, req: Request) -> Response:
        t0 = time.perf_counter()
        try:
            text, usage = self._call(req)
        except UnsupportedCondition:
            raise
        except Exception as e:  # network, auth, content filter → recorded as an error row
            return Response("", None, refused=True, latency_s=time.perf_counter() - t0, error=f"{type(e).__name__}: {e}")
        parsed = parse_json(text)
        refused = parsed is None and self._looks_like_refusal(text)
        return Response(
            raw_text=text, parsed=parsed, refused=refused,
            latency_s=time.perf_counter() - t0,
            input_tokens=usage.get("input"), output_tokens=usage.get("output"),
        )

    def _call(self, req: Request) -> tuple[str, dict]:
        raise NotImplementedError

    @staticmethod
    def _looks_like_refusal(text: str) -> bool:
        t = text.lower()
        return (not t.strip()) or any(k in t for k in ("i can't", "i cannot", "unable to", "not able to", "i won't"))
