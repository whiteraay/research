"""Gemini via the google-genai SDK. Gemini accepts video files natively, so
raw_media for Gemini sends the session video (audio included) rather than frames.

VERIFY before locking the study: model id, video size limits (large files go
through client.files.upload), and the safety-settings API at
https://ai.google.dev/gemini-api/docs. Needs GEMINI_API_KEY (or GOOGLE_API_KEY).
"""
from __future__ import annotations

from pathlib import Path

from .base import BaseClient, Request


class GeminiClient(BaseClient):
    provider = "gemini"

    def __init__(self, model_id: str):
        super().__init__(model_id)
        from google import genai

        self._genai = genai
        self._client = genai.Client()

    def _call(self, req: Request) -> tuple[str, dict]:
        from google.genai import types

        parts: list = []
        if req.video_path:
            p = Path(req.video_path)
            if p.stat().st_size > 15 * 1024 * 1024:   # inline limit is ~20 MB; use Files API above that
                up = self._client.files.upload(file=str(p))
                parts.append(up)
            else:
                parts.append(types.Part.from_bytes(data=p.read_bytes(), mime_type="video/mp4"))
        elif req.audio_path:
            parts.append(types.Part.from_bytes(data=Path(req.audio_path).read_bytes(), mime_type="audio/wav"))
        for f in req.frames:
            parts.append(types.Part.from_bytes(data=f, mime_type="image/jpeg"))
        parts.append(req.user_text)

        resp = self._client.models.generate_content(
            model=self.model_id,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=req.system,
                temperature=req.temperature,
                max_output_tokens=req.max_tokens,
            ),
        )
        # A blocked response has no candidates / empty text; record it as empty → refusal
        text = resp.text or "" if getattr(resp, "candidates", None) else ""
        um = getattr(resp, "usage_metadata", None)
        usage = {
            "input": getattr(um, "prompt_token_count", None),
            "output": getattr(um, "candidates_token_count", None),
        }
        return text, usage
