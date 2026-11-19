from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .settings import ConfigError, Settings
from .resolution import calculate_resolution


class WorkflowError(ValueError):
    """The configured MiniMax API workflow is unsafe or invalid."""


def build_workflow(settings: Settings, prompt: str, seed: int,
                   aspect_ratio: str | None = None, megapixels: float | None = None,
                   sampler_name: str | None = None, scheduler: str | None = None) -> dict[str, Any]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise WorkflowError("prompt must be a non-empty string")
    if len(prompt) > 20000:
        raise WorkflowError("prompt exceeds 20000 characters")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= (2**64 - 1):
        raise WorkflowError("seed must be an unsigned 64-bit integer")

    config = settings.data
    generation = config["generation"]
    if generation["prompt_enhancement"] is not False:
        raise ConfigError("generation.prompt_enhancement must be false")
    selected_ratio = aspect_ratio if aspect_ratio is not None else generation["default_aspect_ratio"]
    selected_megapixels = megapixels if megapixels is not None else generation["default_megapixels"]
    selected_sampler = sampler_name or generation["default_sampler"]
    selected_scheduler = scheduler or generation["default_scheduler"]
    if selected_sampler not in generation["sampler_options"]:
        raise WorkflowError("sampler_name is not one of the configured ComfyUI KSampler options")
    if selected_scheduler not in generation["scheduler_options"]:
        raise WorkflowError("scheduler is not one of the configured ComfyUI KSampler options")
    try:
        selected_width, selected_height = calculate_resolution(
            selected_ratio, selected_megapixels, generation["resolution_multiple"], generation["aspect_ratios"]
        )
    except ValueError as error:
        raise WorkflowError(str(error)) from error

    workflow_path = settings.path("paths.workflow_api")
    try:
        with workflow_path.open("r", encoding="utf-8") as file:
            template = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise WorkflowError(f"cannot read workflow JSON {workflow_path}: {error}") from error
    if not isinstance(template, dict):
        raise WorkflowError("workflow root must be a JSON object")

    graph = copy.deepcopy(template)
    classes = {node.get("class_type") for node in graph.values() if isinstance(node, dict)}
    forbidden = {"TextGenerateV2", "ComfySwitchNode", "JsonExtractString"}
    if classes & forbidden:
        raise WorkflowError("workflow still contains prompt-enhancement nodes")
    if "MiniMaxH3ImageConditioning" not in classes:
        raise WorkflowError("workflow must include MiniMaxH3ImageConditioning")
    if any("qwen3.5" in json.dumps(node).lower() for node in graph.values()):
        raise WorkflowError("workflow must not reference the Qwen3.5 prompt-enhancement model")

    try:
        models = config["comfyui"]["models"]
        graph["1"]["inputs"]["unet_name"] = models["diffusion_model"]
        graph["2"]["inputs"]["clip_name"] = models["text_encoder"]
        graph["3"]["inputs"]["vae_name"] = models["vae"]
        graph["4"]["inputs"].update({"prompt": prompt.strip(), "width": selected_width,
                                     "height": selected_height})
        graph["6"]["inputs"].update({
            "seed": seed,
            "steps": generation["steps"],
            "cfg": generation["cfg"],
            "sampler_name": selected_sampler,
            "scheduler": selected_scheduler,
        })
    except (KeyError, TypeError) as error:
        raise WorkflowError(f"workflow node inputs do not match the MiniMax API graph: {error}") from error
    return graph