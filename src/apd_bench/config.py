from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


@dataclass
class ModelSpec:
    name: str
    provider: str
    model_id: str
    supports: list[str]
    raw_media_kind: str = "frames"


@dataclass
class BenchConfig:
    models: dict[str, ModelSpec]
    conditions: list[str]
    prompt_modes: list[str]
    temperature: float = 0.0
    max_output_tokens: int = 600
    repeats: int = 5
    frames_per_task: int = 8
    frame_max_side: int = 768
    prompts: dict[str, str] = field(default_factory=dict)

    @property
    def prompt_hash(self) -> str:
        blob = "\n---\n".join(f"{k}\n{v}" for k, v in sorted(self.prompts.items()))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_config(path: str | Path, prompts_dir: Path = PROMPTS_DIR) -> BenchConfig:
    raw = yaml.safe_load(Path(path).read_text())
    models = {
        name: ModelSpec(name=name, **spec) for name, spec in raw["models"].items()
    }
    prompts = {p.stem: p.read_text() for p in sorted(prompts_dir.glob("*.txt"))}
    return BenchConfig(
        models=models,
        conditions=raw["conditions"],
        prompt_modes=raw["prompt_modes"],
        temperature=raw.get("temperature", 0.0),
        max_output_tokens=raw.get("max_output_tokens", 600),
        repeats=raw.get("repeats", 5),
        frames_per_task=raw.get("frames", {}).get("per_task", 8),
        frame_max_side=raw.get("frames", {}).get("max_side_px", 768),
        prompts=prompts,
    )
