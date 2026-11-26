from __future__ import annotations

from dataclasses import dataclass


class CoordinationError(RuntimeError):
    """The requested model handoff is unsafe or cannot complete."""


@dataclass(frozen=True)
class ChatOutcome:
    answer: str
    image_prompt: str
    aspect_ratio: str
    context_saved: bool
    warning: str | None = None
    reasoning_content: str = ""


@dataclass(frozen=True)
class GenerationOutcome:
    generation_id: str
    prompt_id: str
    image: dict[str, str]
    cleanup_warning: str | None = None
