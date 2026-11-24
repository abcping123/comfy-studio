from __future__ import annotations

import ipaddress
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .modes import IMAGE_MODES
from .resolution import aspect_ratio_map


_LEGACY_MODEL_DIRECTORY_IDS = {
    "diffusion_model": "diffusion_models",
    "text_encoder": "text_encoders",
    "vae": "vae",
}


class ConfigError(ValueError):
    """The project JSON configuration is invalid or unsafe."""


@dataclass(frozen=True)
class Settings:
    root: Path
    data: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.data[name]
        return value

    def path(self, dotted_key: str) -> Path:
        value: Any = self.data
        for part in dotted_key.split("."):
            value = value[part]
        path = Path(value)
        return path if path.is_absolute() else self.root / path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return project_root() / "config.json"


def _mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be a JSON object")
    return value


def _text(data: dict[str, Any], key: str, prefix: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{prefix}.{key} must be a non-empty string")
    return value.strip()


def _port(data: dict[str, Any], key: str, prefix: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ConfigError(f"{prefix}.{key} must be an integer from 1 to 65535")
    return value


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _endpoint(data: dict[str, Any], prefix: str) -> None:
    url = _text(data, "endpoint", prefix)
    try:
        parsed = urlsplit(url)
        url_port = parsed.port
    except ValueError as error:
        raise ConfigError(f"{prefix}.base_url is invalid: {error}") from error
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigError(f"{prefix}.endpoint must be an HTTP(S) URL with a host")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ConfigError(f"{prefix}.endpoint must not contain credentials, a query, or a fragment")
    if url_port is None or not 1 <= url_port <= 65535:
        raise ConfigError(f"{prefix}.endpoint must include a port from 1 to 65535")
    if not _is_loopback(parsed.hostname):
        raise ConfigError(f"{prefix}.endpoint must be loopback")


def _migrate_v1_config(data: dict[str, Any]) -> None:
    paths = _mapping(data, "paths")
    generation = _mapping(data, "generation")
    comfy = _mapping(data, "comfyui")
    legacy_models = _mapping(comfy, "models")

    model_directories: dict[str, Any] = {}
    for directory_id in dict.fromkeys(_LEGACY_MODEL_DIRECTORY_IDS.values()):
        legacy_key = f"{directory_id}_dir"
        if legacy_key in comfy:
            model_directories[directory_id] = comfy.pop(legacy_key)

    generation_modes: dict[str, Any] = {}
    for mode in IMAGE_MODES:
        mode_models: dict[str, dict[str, Any]] = {}
        for model_key in mode.required_model_keys:
            if model_key not in legacy_models:
                continue
            directory_id = _LEGACY_MODEL_DIRECTORY_IDS.get(model_key)
            if directory_id is not None:
                mode_models[model_key] = {
                    "directory": directory_id,
                    "file": legacy_models[model_key],
                }

        workflow_key = mode.legacy_workflow_path_key
        prompt_key = mode.legacy_system_prompt_key
        generation_modes[mode.id] = {
            "workflow": (
                paths.pop(workflow_key, mode.default_workflow_path)
                if workflow_key is not None else mode.default_workflow_path
            ),
            "system_prompt_file": (
                generation.pop(prompt_key, mode.default_system_prompt_file)
                if prompt_key is not None else mode.default_system_prompt_file
            ),
            "models": mode_models,
        }

    comfy.pop("models", None)
    comfy["model_directories"] = model_directories
    data["generation_modes"] = generation_modes
    data["schema_version"] = 3


def _migrate_v2_config(data: dict[str, Any]) -> None:
    if "generation_modes" in data:
        raise ConfigError("schema_version 2 must use image_modes")
    if "image_modes" not in data:
        raise ConfigError("schema_version 2 requires image_modes")
    data["generation_modes"] = data.pop("image_modes")
    data["schema_version"] = 3


def validate_config(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError("the root of config.json must be a JSON object")
    version = data.get("schema_version")
    if isinstance(version, bool):
        raise ConfigError("schema_version must be 1, 2, or 3")
    if version == 1:
        _migrate_v1_config(data)
    elif version == 2:
        _migrate_v2_config(data)
    elif version != 3:
        raise ConfigError("schema_version must be 1, 2, or 3")

    app = _mapping(data, "app")
    app_host = _text(app, "listen_host", "app")
    _port(app, "port", "app")
    token = app.get("access_token")
    if not isinstance(token, str):
        raise ConfigError("app.access_token must be a string")
    allow_insecure_lan_http = app.get("allow_insecure_lan_http", False)
    if not isinstance(allow_insecure_lan_http, bool):
        raise ConfigError("app.allow_insecure_lan_http must be a boolean")
    if not _is_loopback(app_host) and not token.strip():
        raise ConfigError("LAN access requires app.access_token")
    if not _is_loopback(app_host) and not allow_insecure_lan_http:
        raise ConfigError("LAN HTTP requires app.allow_insecure_lan_http=true")

    strata = _mapping(data, "strata")
    _endpoint(strata, "strata")
    _text(strata, "python", "strata")
    _text(strata, "server_script", "strata")
    _text(strata, "source_config_json", "strata")
    _text(strata, "api_key", "strata") if strata.get("api_key") else None
    _text(strata, "model", "strata")
    idle_unload = strata.get("idle_unload_s")
    if isinstance(idle_unload, bool) or not isinstance(idle_unload, (int, float)) or idle_unload < 0:
        raise ConfigError("strata.idle_unload_s must be a non-negative number")
    strata_timeout = strata.get("request_timeout_s")
    if isinstance(strata_timeout, bool) or not isinstance(strata_timeout, (int, float)) or strata_timeout <= 0:
        raise ConfigError("strata.request_timeout_s must be a positive number")
    comfy = _mapping(data, "comfyui")
    _endpoint(comfy, "comfyui")
    for key_name in ("python", "main_script", "working_directory", "output_directory"):
        _text(comfy, key_name, "comfyui")
    model_directories = _mapping(comfy, "model_directories")
    for directory_id in model_directories:
        if (not isinstance(directory_id, str) or not directory_id.strip()
                or "." in directory_id):
            raise ConfigError("comfyui.model_directories keys must be non-empty and contain no dots")
        _text(model_directories, directory_id, "comfyui.model_directories")
    comfy.setdefault("input_directory", "temp/comfyui/input")
    _text(comfy, "input_directory", "comfyui")
    gpu_id = comfy.get("gpu_id")
    if isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0:
        raise ConfigError("comfyui.gpu_id must be a non-negative integer")
    for key_name in ("request_timeout_s", "poll_interval_s"):
        value = comfy.get(key_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(f"comfyui.{key_name} must be a positive number")

    paths = _mapping(data, "paths")
    for key_name in ("strata_runtime_config", "strata_sessions", "database", "temp_root"):
        _text(paths, key_name, "paths")

    generation_modes = _mapping(data, "generation_modes")
    mode_ids = {mode.id for mode in IMAGE_MODES}
    unknown_modes = [
        mode_id for mode_id in generation_modes
        if not isinstance(mode_id, str) or mode_id not in mode_ids
    ]
    if unknown_modes:
        unknown = ", ".join(sorted(str(mode_id) for mode_id in unknown_modes))
        raise ConfigError(f"generation_modes contains unregistered mode(s): {unknown}")
    for mode in IMAGE_MODES:
        mode_prefix = f"generation_modes.{mode.id}"
        mode_config = _mapping(generation_modes, mode.id)
        mode_config.setdefault("workflow", mode.default_workflow_path)
        mode_config.setdefault("system_prompt_file", mode.default_system_prompt_file)
        _text(mode_config, "workflow", mode_prefix)
        _text(mode_config, "system_prompt_file", mode_prefix)

        models = _mapping(mode_config, "models")
        for model_key in models:
            model_prefix = f"{mode_prefix}.models.{model_key}"
            if not isinstance(model_key, str) or not model_key.strip():
                raise ConfigError(f"{mode_prefix}.models keys must be non-empty strings")
            model_config = _mapping(models, model_key)
            directory_id = _text(model_config, "directory", model_prefix)
            if directory_id not in model_directories:
                raise ConfigError(
                    f"{model_prefix}.directory must match a comfyui.model_directories key"
                )
            filename = _text(model_config, "file", model_prefix)
            if (filename in {".", ".."} or Path(filename).name != filename
                    or "\\" in filename or "\x00" in filename):
                raise ConfigError(f"{model_prefix}.file must be a model filename, not a path")

        for model_key in mode.required_model_keys:
            if model_key not in models:
                raise ConfigError(f"{mode_prefix}.models.{model_key} is required")
        for model_key, required_prefix in mode.required_model_prefixes:
            filename = models[model_key]["file"]
            if not isinstance(filename, str) or not filename.startswith(required_prefix):
                raise ConfigError(
                    f"{mode_prefix}.models.{model_key}.file must start with {required_prefix}"
                )

    generation = _mapping(data, "generation")
    if generation.get("prompt_enhancement") is not False:
        raise ConfigError("generation.prompt_enhancement must be false")
    prompt_skill_version = generation.get("prompt_skill_version")
    if isinstance(prompt_skill_version, bool) or not isinstance(prompt_skill_version, int) or prompt_skill_version < 1:
        raise ConfigError("generation.prompt_skill_version must be a positive integer")
    aspect_options = generation.get("aspect_ratios")
    try:
        aspect_options_map = aspect_ratio_map(aspect_options)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"generation.aspect_ratios is invalid: {error}") from error
    if generation.get("default_aspect_ratio") not in aspect_options_map:
        raise ConfigError("generation.default_aspect_ratio must match a ResolutionSelector option")
    megapixels = generation.get("default_megapixels")
    if isinstance(megapixels, bool) or not isinstance(megapixels, (int, float)) or not 0.5 <= megapixels <= 20.0:
        raise ConfigError("generation.default_megapixels must be between 0.5 and 20.0")
    try:
        if Decimal(str(megapixels)) * 100 % 1 != 0:
            raise ConfigError("generation.default_megapixels must use 0.01 MP increments")
    except InvalidOperation as error:
        raise ConfigError("generation.default_megapixels must be a number") from error
    multiple = generation.get("resolution_multiple")
    if isinstance(multiple, bool) or not isinstance(multiple, int) or not 8 <= multiple <= 128 or (multiple - 8) % 4:
        raise ConfigError("generation.resolution_multiple must be 8..128 in steps of 4")
    for key_name in ("steps",):
        value = generation.get(key_name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(f"generation.{key_name} must be a positive integer")
    for key_name in ("cfg",):
        if key_name not in generation:
            raise ConfigError(f"generation.{key_name} is required")
    for options_key, selected_key in (("sampler_options", "default_sampler"),
                                      ("scheduler_options", "default_scheduler")):
        options = generation.get(options_key)
        if not isinstance(options, list) or not options or not all(isinstance(item, str) and item for item in options):
            raise ConfigError(f"generation.{options_key} must be a non-empty list of strings")
        if generation.get(selected_key) not in options:
            raise ConfigError(f"generation.{selected_key} must be included in {options_key}")
    coordination = _mapping(data, "coordination")
    for key_name in ("generation_timeout_s", "poll_interval_s"):
        value = coordination.get(key_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(f"coordination.{key_name} must be a positive number")
    return data


def load_config(path: Path | None = None) -> Settings:
    config_path = (path or default_config_path()).resolve()
    try:
        with config_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except OSError as error:
        raise ConfigError(f"cannot read {config_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"invalid JSON in {config_path}: {error}") from error
    validate_config(data)
    return Settings(root=config_path.parent, data=data)