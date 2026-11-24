from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .modes import (
    DEFAULT_MODE_ID,
    ModeDefinition,
    calculate_mode_dimensions,
    get_mode,
    validate_source_image_count,
)
from .settings import ConfigError, Settings


class WorkflowError(ValueError):
    """The configured image workflow is unsafe or invalid."""


@dataclass(frozen=True)
class WorkflowContext:
    mode: ModeDefinition
    generation: dict[str, Any]
    models: dict[str, Any]
    prompt: str
    seed: int
    width: int
    height: int
    sampler_name: str
    scheduler: str
    source_images: list[dict[str, Any]]


def _configure_minimax_h3_common(graph: dict[str, Any], context: WorkflowContext) -> None:
    graph["1"]["inputs"]["unet_name"] = context.models["diffusion_model"]
    graph["2"]["inputs"]["clip_name"] = context.models["text_encoder"]
    graph["3"]["inputs"]["vae_name"] = context.models["vae"]
    graph["4"]["inputs"].update({
        "prompt": context.prompt,
        "width": context.width,
        "height": context.height,
    })
    graph["6"]["inputs"].update({
        "seed": context.seed,
        "steps": context.generation["steps"],
        "cfg": context.generation["cfg"],
        "sampler_name": context.sampler_name,
        "scheduler": context.scheduler,
    })


def _configure_minimax_h3_t2i(graph: dict[str, Any], context: WorkflowContext) -> None:
    _configure_minimax_h3_common(graph, context)


def _configure_minimax_h3_edit(graph: dict[str, Any], context: WorkflowContext) -> None:
    _configure_minimax_h3_common(graph, context)
    graph["4"]["inputs"].update({
        "width": ["12", 0],
        "height": ["12", 1],
        "images.image_1": ["11", 0],
    })
    graph["10"]["inputs"].update({
        "resize_type": "scale total pixels",
        "resize_type.megapixels": context.generation["selected_megapixels"],
    })
    graph["11"]["inputs"].update({
        "resize_type": "scale to multiple",
        "resize_type.multiple": context.generation["resolution_multiple"],
    })
    graph["9"]["inputs"]["image"] = _source_image_path(context.source_images[0])
    for index, image in enumerate(context.source_images[1:], start=2):
        node_id = str(11 + index)
        graph[node_id]["inputs"]["image"] = _source_image_path(image)
        graph["4"]["inputs"][f"images.image_{index}"] = [node_id, 0]
    for image_number in range(len(context.source_images) + 1, 6):
        graph.pop(str(11 + image_number), None)


_WORKFLOW_STRATEGIES: dict[str, Callable[[dict[str, Any], WorkflowContext], None]] = {
    "minimax-h3-text-to-image": _configure_minimax_h3_t2i,
    "minimax-h3-image-edit": _configure_minimax_h3_edit,
}


def build_workflow(settings: Settings, prompt: str, seed: int,
                   aspect_ratio: str | None = None, megapixels: float | None = None,
                   sampler_name: str | None = None, scheduler: str | None = None,
                   image_mode: str = DEFAULT_MODE_ID,
                   source_images: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise WorkflowError("prompt must be a non-empty string")
    if len(prompt) > 20000:
        raise WorkflowError("prompt exceeds 20000 characters")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= (2**64 - 1):
        raise WorkflowError("seed must be an unsigned 64-bit integer")

    try:
        mode = get_mode(image_mode)
        images = source_images or []
        validate_source_image_count(mode, len(images))
    except ValueError as error:
        raise WorkflowError(str(error)) from error
    if not all(isinstance(image, dict) for image in images):
        raise WorkflowError("source images must be objects")

    config = settings.data
    generation = config["generation"]
    if generation["prompt_enhancement"] is not False:
        raise ConfigError("generation.prompt_enhancement must be false")
    selected_sampler = sampler_name or generation["default_sampler"]
    selected_scheduler = scheduler or generation["default_scheduler"]
    if selected_sampler not in generation["sampler_options"]:
        raise WorkflowError("sampler_name is not one of the configured ComfyUI KSampler options")
    if selected_scheduler not in generation["scheduler_options"]:
        raise WorkflowError("scheduler is not one of the configured ComfyUI KSampler options")
    try:
        selected_width, selected_height = calculate_mode_dimensions(
            mode, generation, aspect_ratio, megapixels, images,
        )
    except (TypeError, ValueError) as error:
        raise WorkflowError(str(error)) from error

    mode_config = config["generation_modes"][mode.id]
    workflow_path = settings.path(f"generation_modes.{mode.id}.workflow")
    try:
        with workflow_path.open("r", encoding="utf-8") as file:
            template = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise WorkflowError(f"cannot read workflow JSON {workflow_path}: {error}") from error
    if not isinstance(template, dict):
        raise WorkflowError("workflow root must be a JSON object")

    graph = copy.deepcopy(template)
    classes = {
        node.get("class_type")
        for node in graph.values()
        if isinstance(node, dict)
    }
    if classes & set(mode.forbidden_node_classes):
        raise WorkflowError(f"{mode.label} workflow contains a forbidden prompt-enhancement node")
    missing_classes = set(mode.required_node_classes) - classes
    if missing_classes:
        required = ", ".join(sorted(missing_classes))
        raise WorkflowError(f"workflow must include {required}")
    serialized = json.dumps(graph).lower()
    if any(marker.lower() in serialized for marker in mode.forbidden_model_markers):
        raise WorkflowError(f"{mode.label} workflow references a forbidden model")

    strategy = _WORKFLOW_STRATEGIES.get(mode.workflow_strategy)
    if strategy is None:
        raise WorkflowError(f"no workflow strategy is registered for image mode {mode.id}")
    try:
        selected_megapixels = (
            min(generation["default_megapixels"], mode.max_megapixels)
            if megapixels is None else megapixels
        )
        models = {
            model_key: model_config["file"]
            for model_key, model_config in mode_config["models"].items()
        }
        context = WorkflowContext(
            mode=mode,
            generation={**generation, "selected_megapixels": selected_megapixels},
            models=models,
            prompt=prompt.strip(),
            seed=seed,
            width=selected_width,
            height=selected_height,
            sampler_name=selected_sampler,
            scheduler=selected_scheduler,
            source_images=images,
        )
        strategy(graph, context)
    except (KeyError, TypeError) as error:
        raise WorkflowError(
            f"workflow node inputs do not match the {mode.label} graph: {error}"
        ) from error
    return graph


def _source_image_path(source_image: dict[str, Any]) -> str:
    filename = source_image.get("filename")
    subfolder = source_image.get("subfolder")
    if (not isinstance(filename, str) or filename in {"", ".", ".."} or Path(filename).name != filename
            or "\\" in filename or not isinstance(subfolder, str)
            or Path(subfolder).is_absolute() or "\\" in subfolder
            or any(part in {"", ".", ".."} for part in subfolder.split("/"))):
        raise WorkflowError("source image has an unsafe ComfyUI input path")
    return f"{subfolder}/{filename}"
