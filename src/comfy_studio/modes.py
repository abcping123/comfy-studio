from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .resolution import calculate_edit_resolution, calculate_resolution


MIN_MEGAPIXELS = 0.5
DEFAULT_MAX_SOURCE_IMAGES = 5
DEFAULT_MODE_ID = "text-to-image"
CHARACTER_SWAP_LORA_MODEL_KEY = "character_swap_lora"
CHARACTER_SWAP_LORA_FILENAME = "h3_image_character_swap_v1.safetensors"


@dataclass(frozen=True)
class ModePresentation:
    prompt_label: str
    megapixels_label: str
    composer_hint: str
    generate_label: str
    missing_sources_hint: str
    source_images_label: str
    source_images_help: str


@dataclass(frozen=True)
class ModeDefinition:
    id: str
    label: str
    default_workflow_path: str
    default_system_prompt_file: str
    workflow_strategy: str
    resolution_strategy: Literal["aspect-ratio", "source-image"]
    max_megapixels: float
    source_image_min_count: int
    source_image_max_count: int
    source_image_roles: tuple[str, ...]
    source_image_error: str
    send_source_images_to_chat: bool
    stored_aspect_ratio_label: str | None
    presentation: ModePresentation
    required_node_classes: tuple[str, ...]
    forbidden_node_classes: tuple[str, ...]
    forbidden_model_markers: tuple[str, ...]
    required_model_keys: tuple[str, ...]
    required_model_prefixes: tuple[tuple[str, str], ...] = ()
    default_megapixels: float | None = None


IMAGE_MODES = (
    ModeDefinition(
        id="text-to-image",
        label="Text to image",
        default_workflow_path="data/comfyui/minimax_h3_image_t2i.json",
        default_system_prompt_file="data/prompts/minimax_h3_image_t2i.txt",
        workflow_strategy="minimax-h3-text-to-image",
        resolution_strategy="aspect-ratio",
        max_megapixels=20.0,
        source_image_min_count=0,
        source_image_max_count=0,
        source_image_roles=(),
        source_image_error="text-to-image mode does not accept source images",
        send_source_images_to_chat=False,
        stored_aspect_ratio_label=None,
        presentation=ModePresentation(
            prompt_label="Prompt",
            megapixels_label="Pixels (MP)",
            composer_hint="The chat model will help shape the image prompt",
            generate_label="Generate image",
            missing_sources_hint="",
            source_images_label="Text to image source images",
            source_images_help="",
        ),
        required_node_classes=("MiniMaxH3ImageConditioning",),
        forbidden_node_classes=("TextGenerateV2", "ComfySwitchNode", "JsonExtractString"),
        forbidden_model_markers=("qwen3.5",),
        required_model_keys=("diffusion_model", "text_encoder", "vae"),
        required_model_prefixes=(("text_encoder", "qwen3vl_"),),
    ),
    ModeDefinition(
        id="edit",
        label="Image edit",
        default_workflow_path="data/comfyui/minimax_h3_image_edit.json",
        default_system_prompt_file="data/prompts/minimax_h3_image_edit.txt",
        workflow_strategy="minimax-h3-image-edit",
        resolution_strategy="source-image",
        max_megapixels=16.0,
        source_image_min_count=1,
        source_image_max_count=DEFAULT_MAX_SOURCE_IMAGES,
        source_image_roles=("base image", "reference image"),
        source_image_error="edit mode requires one base image and at most four reference images",
        send_source_images_to_chat=True,
        stored_aspect_ratio_label="Source image",
        presentation=ModePresentation(
            prompt_label="Edit prompt",
            megapixels_label="Target MP",
            composer_hint="The chat model can inspect uploaded image pixels to help shape the edit prompt",
            generate_label="Edit image",
            missing_sources_hint="Upload a base image",
            source_images_label="Edit source images",
            source_images_help=(
                "Images are ordered. Picture 1 is the edit target; later pictures are references. "
                "Drag pictures to change their order. Keyboard users can focus a picture and press "
                "Alt+Shift+Left or Alt+Shift+Right."
            ),
        ),
        required_node_classes=("MiniMaxH3ImageConditioning",),
        forbidden_node_classes=("TextGenerateV2", "ComfySwitchNode", "JsonExtractString"),
        forbidden_model_markers=("qwen3.5",),
        required_model_keys=("diffusion_model", "text_encoder", "vae"),
        required_model_prefixes=(("text_encoder", "qwen3vl_"),),
    ),
    ModeDefinition(
        id="character-swap",
        label="Character swap",
        default_workflow_path="data/comfyui/minimax_h3_image_character_swap.json",
        default_system_prompt_file="data/prompts/minimax_h3_image_character_swap.txt",
        workflow_strategy="minimax-h3-character-swap",
        resolution_strategy="source-image",
        max_megapixels=16.0,
        source_image_min_count=2,
        source_image_max_count=DEFAULT_MAX_SOURCE_IMAGES,
        source_image_roles=("scene image", "character reference"),
        source_image_error=(
            "character swap requires one scene image and at least one character reference"
        ),
        send_source_images_to_chat=True,
        stored_aspect_ratio_label="Source image",
        presentation=ModePresentation(
            prompt_label="Swap instruction",
            megapixels_label="Target MP",
            composer_hint="The chat model can inspect the images to help shape the swap instruction",
            generate_label="Swap character",
            missing_sources_hint="Upload a scene image and a character reference",
            source_images_label="Character swap images",
            source_images_help=(
                "Picture 1 is the scene; Picture 2 and later are character references. "
                "Drag pictures to change their order. Keyboard users can focus a picture and press "
                "Alt+Shift+Left or Alt+Shift+Right."
            ),
        ),
        required_node_classes=("MiniMaxH3ImageConditioning", "LoadLoraModel"),
        forbidden_node_classes=("TextGenerateV2", "ComfySwitchNode", "JsonExtractString"),
        forbidden_model_markers=("qwen3.5",),
        required_model_keys=(
            "diffusion_model",
            "text_encoder",
            "vae",
            CHARACTER_SWAP_LORA_MODEL_KEY,
        ),
        required_model_prefixes=(
            ("text_encoder", "qwen3vl_"),
            (CHARACTER_SWAP_LORA_MODEL_KEY, "h3_image_character_swap_v1"),
        ),
        default_megapixels=1.0,
    ),
)

MODE_REGISTRY = {mode.id: mode for mode in IMAGE_MODES}
if len(MODE_REGISTRY) != len(IMAGE_MODES):
    raise RuntimeError("image mode ids must be unique")
if DEFAULT_MODE_ID not in MODE_REGISTRY:
    raise RuntimeError("the default image mode must be registered")

MAX_SOURCE_IMAGES = max(mode.source_image_max_count for mode in IMAGE_MODES)


def get_mode(mode_id: str) -> ModeDefinition:
    if not isinstance(mode_id, str) or not mode_id:
        raise ValueError("image mode must be a non-empty string")
    try:
        return MODE_REGISTRY[mode_id]
    except KeyError as error:
        raise ValueError(f"unsupported image mode: {mode_id}") from error


def validate_source_image_count(mode: ModeDefinition, count: int) -> None:
    if not mode.source_image_min_count <= count <= mode.source_image_max_count:
        raise ValueError(mode.source_image_error)


def source_image_role(mode: ModeDefinition, index: int) -> str:
    if not mode.source_image_roles:
        return "source image"
    return mode.source_image_roles[min(index, len(mode.source_image_roles) - 1)]


def mode_default_megapixels(mode: ModeDefinition, generation: dict[str, Any]) -> float:
    default = (
        generation["default_megapixels"]
        if mode.default_megapixels is None
        else mode.default_megapixels
    )
    return float(min(default, mode.max_megapixels))


def calculate_mode_dimensions(
    mode: ModeDefinition,
    generation: dict[str, Any],
    aspect_ratio: str | None,
    megapixels: float | None,
    source_images: list[dict[str, Any]],
) -> tuple[int, int]:
    selected_megapixels = mode_default_megapixels(mode, generation) if megapixels is None else megapixels
    if (isinstance(selected_megapixels, bool)
            or not isinstance(selected_megapixels, (int, float))
            or not MIN_MEGAPIXELS <= selected_megapixels <= mode.max_megapixels):
        prefix = "edit megapixels" if mode.resolution_strategy == "source-image" else "megapixels"
        raise ValueError(
            f"{prefix} must be between {MIN_MEGAPIXELS:.1f} and {mode.max_megapixels:.1f}"
        )

    if mode.resolution_strategy == "aspect-ratio":
        selected_ratio = aspect_ratio or generation["default_aspect_ratio"]
        return calculate_resolution(
            selected_ratio,
            selected_megapixels,
            generation["resolution_multiple"],
            generation["aspect_ratios"],
        )
    if mode.resolution_strategy == "source-image":
        validate_source_image_count(mode, len(source_images))
        base_image = source_images[0]
        return calculate_edit_resolution(
            base_image.get("width"),
            base_image.get("height"),
            selected_megapixels,
            generation["resolution_multiple"],
        )
    raise ValueError(f"unsupported resolution strategy: {mode.resolution_strategy}")


def public_mode_options(generation: dict[str, Any]) -> list[dict[str, Any]]:
    options = []
    for mode in IMAGE_MODES:
        options.append({
            "id": mode.id,
            "label": mode.label,
            "resolution": {
                "strategy": mode.resolution_strategy,
                "megapixels": {
                    "min": MIN_MEGAPIXELS,
                    "max": mode.max_megapixels,
                    "step": 0.01,
                    "default": mode_default_megapixels(mode, generation),
                },
            },
            "source_images": {
                "min_count": mode.source_image_min_count,
                "max_count": mode.source_image_max_count,
                "roles": list(mode.source_image_roles),
            },
            "ui": {
                "prompt_label": mode.presentation.prompt_label,
                "megapixels_label": mode.presentation.megapixels_label,
                "composer_hint": mode.presentation.composer_hint,
                "generate_label": mode.presentation.generate_label,
                "missing_sources_hint": mode.presentation.missing_sources_hint,
                "source_images_label": mode.presentation.source_images_label,
                "source_images_help": mode.presentation.source_images_help,
            },
        })
    return options
