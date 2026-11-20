from __future__ import annotations

import ipaddress
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .resolution import aspect_ratio_map


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


def validate_config(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError("the root of config.json must be a JSON object")
    version = data.get("schema_version")
    if isinstance(version, bool) or version != 1:
        raise ConfigError("schema_version must be 1")

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
    free_vram = strata.get("min_free_vram_mib")
    if free_vram is not None and (isinstance(free_vram, bool) or not isinstance(free_vram, int) or free_vram < 0):
        raise ConfigError("strata.min_free_vram_mib must be a non-negative integer or null")

    comfy = _mapping(data, "comfyui")
    _endpoint(comfy, "comfyui")
    for key_name in ("python", "main_script", "working_directory", "diffusion_models_dir",
                     "text_encoders_dir", "vae_dir", "output_directory"):
        _text(comfy, key_name, "comfyui")
    gpu_id = comfy.get("gpu_id")
    if isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0:
        raise ConfigError("comfyui.gpu_id must be a non-negative integer")
    for key_name in ("request_timeout_s", "poll_interval_s"):
        value = comfy.get(key_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(f"comfyui.{key_name} must be a positive number")
    models = _mapping(comfy, "models")
    for key_name in ("diffusion_model", "text_encoder", "vae"):
        _text(models, key_name, "comfyui.models")

    paths = _mapping(data, "paths")
    for key_name in ("strata_runtime_config", "strata_sessions", "database", "temp_root", "workflow_api"):
        _text(paths, key_name, "paths")

    generation = _mapping(data, "generation")
    if generation.get("prompt_enhancement") is not False:
        raise ConfigError("generation.prompt_enhancement must be false")
    _text(generation, "system_prompt_file", "generation")
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
    for key_name in ("handoff_timeout_s", "generation_timeout_s", "poll_interval_s"):
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