"""GPT via the OpenAI Python SDK (Chat Completions, image inputs as data URLs).

VERIFY before locking the study: model id, whether the chosen model accepts
audio in the same request as images, and the current responses/chat API shape
at https://platform.openai.com/docs. This client sends frames + text, which is
the parity condition used for Claude and GPT (see configs/models.yaml).
Needs OPENAI_API_KEY in the environment.
"""
from __future__ import annotations

from ..preprocess.frames import b64
from .base import BaseClient, Request, UnsupportedCondition


class OpenAIClient(BaseClient):
    provider = "openai"

    def __init__(self, model_id: str):
        super().__init__(model_id)
        from openai import OpenAI

        self._client = OpenAI()

    def _call(self, req: Request) -> tuple[str, dict]:
        if req.video_path or req.audio_path:
            raise UnsupportedCondition("video/audio file input not enabled in this client; use frames")
        content: list[dict] = []
        for f in req.frames:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64(f)}", "detail": "low"}})
        content.append({"type": "text", "text": req.user_text})
        resp = self._client.chat.completions.create(
            model=self.model_id,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            messages=[
                {"role": "system", "content": req.system},
                {"role": "user", "content": content},
            ],
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        if getattr(choice, "finish_reason", "") == "content_filter":
            text = ""
        u = resp.usage
        usage = {"input": getattr(u, "prompt_tokens", None), "output": getattr(u, "completion_tokens", None)}
        return text, usage
