"""
Pydantic schema for LLM-generated synthetic diary samples
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, field_validator

from ...utils.emotion_constants import DEFAULT_EMOTION_LABELS


class SyntheticSample(BaseModel):
    """One LLM-generated diary sample, with full generation provenance."""

    sample_id: str
    text: str
    label: int
    label_name: str
    model: str  # "Llama-3-8B-Instruct" | "Qwen3-8B"
    channel: str  # "openrouter" | "hf_colab" | "scripted"
    axis_style: str  # giá trị rút từ trục "văn phong"
    axis_length: str  # giá trị rút từ trục "độ dài"
    axis_context: str  # giá trị rút từ trục "ngữ cảnh"
    prompt_template_id: str
    generation_round: int
    generated_at: str  # ISO 8601 UTC

    @field_validator("label")
    @classmethod
    def _label_in_range(cls, v: int) -> int:
        if v not in DEFAULT_EMOTION_LABELS:
            raise ValueError(f"label must be one of {sorted(DEFAULT_EMOTION_LABELS)}, got {v}")
        return v


def new_sample_id(model: str, label: int) -> str:
    """
    Build a stable, unique sample id.

    Args:
        model: Display name of the generating model (e.g. "Llama-3-8B-Instruct")
        label: Emotion label index (0-6)

    Returns:
        A string id like "llama-3-8b-instruct-2-a1b2c3d4e5"
    """
    slug = model.lower().replace(" ", "-")
    return f"{slug}-{label}-{uuid4().hex[:10]}"


def now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def read_samples_jsonl(path: str | Path) -> list[SyntheticSample]:
    """Read a JSONL file of SyntheticSample rows (one per line)."""
    with open(path, encoding="utf-8") as f:
        return [SyntheticSample.model_validate_json(line) for line in f if line.strip()]


def write_samples_jsonl(samples: list[SyntheticSample], path: str | Path, mode: str = "w") -> None:
    """Write SyntheticSample rows to a JSONL file, one per line."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, mode, encoding="utf-8") as f:
        for sample in samples:
            f.write(sample.model_dump_json() + "\n")
