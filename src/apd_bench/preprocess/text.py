"""Transcription (Whisper, word timestamps) and transcript-level features.

Transcript JSON format used throughout:
    {"words": [{"word": "I", "start": 12.31, "end": 12.40}, ...]}
A plain .txt transcript is accepted but then latency and speech rate are unavailable.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

FIRST_PERSON_SG = {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'd", "i'll"}
NEGATIVE_LEX = {  # tiny seed lexicon; swap for a validated one (e.g. LIWC/NRC) in production
    "sad", "tired", "empty", "alone", "hopeless", "dark", "pain", "lost", "nothing",
    "worthless", "heavy", "cold", "crying", "end", "gone", "broken",
}


def transcribe(audio_path: str | Path, model_name: str = "medium", language: Optional[str] = None) -> dict:
    """Whisper with word timestamps. Requires `pip install openai-whisper` and ffmpeg."""
    import whisper

    model = whisper.load_model(model_name)
    res = model.transcribe(str(audio_path), word_timestamps=True, language=language)
    words = []
    for seg in res["segments"]:
        for w in seg.get("words", []):
            words.append({"word": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"])})
    return {"text": res["text"].strip(), "words": words, "language": res.get("language")}


def load_transcript(path: str | Path) -> dict:
    p = Path(path)
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text())
    return {"text": p.read_text().strip(), "words": []}


def segment_words(tr: dict, start: float, end: float) -> list[dict]:
    return [w for w in tr.get("words", []) if start <= w["start"] <= end]


def segment_text(tr: dict, start: float, end: float) -> str:
    ws = segment_words(tr, start, end)
    if ws:
        return " ".join(w["word"] for w in ws)
    return tr.get("text", "")  # no timestamps: fall back to the whole transcript


def response_latency(tr: dict, stimulus_onset: Optional[float], end: float) -> Optional[float]:
    """Seconds from image onset to the first spoken word."""
    if stimulus_onset is None:
        return None
    ws = [w for w in tr.get("words", []) if stimulus_onset <= w["start"] <= end]
    return (ws[0]["start"] - stimulus_onset) if ws else None


def text_features(text: str, duration: Optional[float] = None) -> dict[str, float]:
    toks = re.findall(r"[a-zA-Z']+", text.lower())
    n = max(len(toks), 1)
    out = {
        "n_words": float(len(toks)),
        "type_token_ratio": len(set(toks)) / n,
        "first_person_sg_rate": sum(t in FIRST_PERSON_SG for t in toks) / n,
        "negative_lex_rate": sum(t in NEGATIVE_LEX for t in toks) / n,
    }
    if duration and duration > 0:
        out["speech_rate_wps"] = len(toks) / duration
    return out


def describe_text(feats: dict[str, float], latency: Optional[float]) -> str:
    parts = [
        f"{int(feats['n_words'])} words",
        f"type-token ratio {feats['type_token_ratio']:.2f}",
        f"first-person singular rate {feats['first_person_sg_rate']:.3f}",
        f"negative-lexicon rate {feats['negative_lex_rate']:.3f}",
    ]
    if "speech_rate_wps" in feats:
        parts.append(f"speech rate {feats['speech_rate_wps']:.2f} words/s")
    s = "Language: " + "; ".join(parts) + "."
    if latency is not None:
        s += f" Timing: response latency {latency:.2f} s from stimulus onset to first word."
    return s
