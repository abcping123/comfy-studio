from __future__ import annotations

import math
import re
from typing import Any


def aspect_ratio_map(options: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for option in options:
        if not isinstance(option, dict):
            raise ValueError("aspect ratio options must be objects")
        label = option.get("label")
        width = option.get("width")
        height = option.get("height")
        if (not isinstance(label, str) or not label or isinstance(width, bool) or not isinstance(width, int)
                or isinstance(height, bool) or not isinstance(height, int) or width <= 0 or height <= 0):
            raise ValueError("aspect ratio options need a label and positive integer ratio")
        if label in result:
            raise ValueError(f"duplicate aspect ratio label: {label}")
        result[label] = (width, height)
    if not result:
        raise ValueError("at least one aspect ratio option is required")
    return result


def normalize_aspect_ratio(value: str, options: list[dict[str, Any]]) -> tuple[str, bool]:
    ratios = aspect_ratio_map(options)
    candidate = value.strip()
    if candidate in ratios:
        return candidate, False
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*[:/]\s*(\d+(?:\.\d+)?)(?!\d)", candidate)
    if match is None:
        raise ValueError(f"aspect ratio is not recognized: {candidate}")
    requested_width, requested_height = float(match.group(1)), float(match.group(2))
    if requested_width <= 0 or requested_height <= 0:
        raise ValueError(f"aspect ratio is invalid: {candidate}")
    requested_ratio = requested_width / requested_height
    nearest = min(ratios, key=lambda label: abs(
        math.log(requested_ratio / (ratios[label][0] / ratios[label][1]))
    ))
    nearest_width, nearest_height = ratios[nearest]
    changed = not math.isclose(requested_ratio, nearest_width / nearest_height, rel_tol=1e-12)
    return nearest, changed


def calculate_resolution(aspect_ratio: str, megapixels: float, multiple: int = 32,
                         aspect_ratios: list[dict[str, Any]] | None = None) -> tuple[int, int]:
    """Mirror ComfyUI ResolutionSelector.execute, including Python's ties-to-even round."""
    if aspect_ratios is None:
        raise ValueError("aspect ratio options must come from the project JSON config")
    ratios = aspect_ratio_map(aspect_ratios)
    if aspect_ratio not in ratios:
        raise ValueError(f"unsupported aspect ratio: {aspect_ratio}")
    if isinstance(megapixels, bool) or not isinstance(megapixels, (int, float)) or not 0.5 <= megapixels <= 20.0:
        raise ValueError("megapixels must be between 0.5 and 20.0")
    if isinstance(multiple, bool) or not isinstance(multiple, int) or not 8 <= multiple <= 128 or (multiple - 8) % 4:
        raise ValueError("multiple must be 8..128 in steps of 4")

    width_ratio, height_ratio = ratios[aspect_ratio]
    scale = math.sqrt(megapixels * 1024 * 1024 / (width_ratio * height_ratio))
    width = round(width_ratio * scale / multiple) * multiple
    height = round(height_ratio * scale / multiple) * multiple
    return width, height