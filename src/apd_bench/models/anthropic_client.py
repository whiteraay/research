"""Claude via the Anthropic Python SDK (Messages API).

Reference: https://platform.claude.com/docs/en/claude_api_primer
Images are sent as base64 content blocks; the API does not take video or audio
files, so raw_media for Claude means sampled frames + the user text.
Needs ANTHROPIC_API_KEY in the environment.
"""
from __future__ import annotations

from ..preprocess.frames import b64
from .base import BaseClient, Request, UnsupportedCondition


class AnthropicClient(BaseClient):
    provider = "anthropic"

    def __init__(self, model_id: str):
        super().__init__(model_id)
        import anthropic

        self._client = anthropic.Anthropic()

    def _call(self, req: Request) -> tuple[str, dict]:
        if req.video_path or req.audio_path:
            raise UnsupportedCondition("Claude Messages API takes images, not video/audio files")
        content: list[dict] = []
        for f in req.frames:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64(f)}})
        content.append({"type": "text", "text": req.user_text})
        msg = self._client.messages.create(
            model=self.model_id,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            system=req.system,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        usage = {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens}
        return text, usage
