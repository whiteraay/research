from __future__ import annotations

from ..config import ModelSpec
from .base import BaseClient, Request, Response, UnsupportedCondition


def make_client(spec: ModelSpec) -> BaseClient:
    if spec.provider == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(spec.model_id)
    if spec.provider == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient(spec.model_id)
    if spec.provider == "gemini":
        from .gemini_client import GeminiClient
        return GeminiClient(spec.model_id)
    if spec.provider == "mock":
        from .mock_client import MockClient
        return MockClient(spec.model_id)
    raise ValueError(f"unknown provider {spec.provider}")


__all__ = ["make_client", "BaseClient", "Request", "Response", "UnsupportedCondition"]
