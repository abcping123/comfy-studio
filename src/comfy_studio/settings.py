from __future__ import annotations

import ipaddress
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .chat_backends import CHAT_BACKENDS, get_chat_backend
from .modes import IMAGE_MODES
from .resolution import aspect_ratio_map


class ConfigError(ValueError):
    """The project JSON configuration is invalid or unsafe."""


@dataclass(frozen=True)
class Settings:
    root: Path
    data: dict[str, Any]
    chat_backend: str = "strata"

    def section(self, name: str) -> dict[str, Any]:
        if name in CHAT_BACKENDS:
            return self.backend_config(name)
        value = self.data[name]
        return value

    def backend_config(self, backend_id: str | None = None) -> dict[str, Any]:
        selected_id = backend_id or self.chat_backend
        backends = self.data["chat_backends"]
        return backends[selected_id]

    def path(self, dotted_key: str) -> Path:
        value: Any = self.data
        parts = dotted_key.split(".")
        if parts[0] in CHAT_BACKENDS:
            parts = ["chat_backends", *parts]
        for part in parts:
            value = value[part]
        path = Path(value).expanduser()
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


def _positive_integer(data: dict[str, Any], key: str, prefix: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{prefix}.{key} must be a positive integer")
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


def _validate_strata_config(strata: dict[str, Any]) -> None:
    _endpoint(strata, "chat_backends.strata")
    _positive_integer(strata, "max_tokens", "chat_backends.strata")
    for key_name in ("python", "server_script", "runtime_config", "model"):
        _text(strata, key_name, "chat_backends.strata")
    source_config = strata.get("source_config_json")
    if source_config is not None and (
        not isinstance(source_config, str) or (source_config and not source_config.strip())
    ):
        raise ConfigError("chat_backends.strata.source_config_json must be a path when set")
    if strata.get("api_key"):
        _text(strata, "api_key", "chat_backends.strata")
    idle_unload = strata.get("idle_unload_s")
    if isinstance(idle_unload, bool) or not isinstance(idle_unload, (int, float)) or idle_unload < 0:
        raise ConfigError("chat_backends.strata.idle_unload_s must be a non-negative number")
    timeout = strata.get("request_timeout_s")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ConfigError("chat_backends.strata.request_timeout_s must be a positive number")


def _validate_bonsai_config(bonsai: dict[str, Any]) -> None:
    _endpoint(bonsai, "chat_backends.bonsai")
    for key_name in ("server_binary", "model_file", "mmproj_file", "model"):
        _text(bonsai, key_name, "chat_backends.bonsai")
    for key_name in ("ctx_size", "parallel", "n_predict", "top_k", "max_tokens"):
        _positive_integer(bonsai, key_name, "chat_backends.bonsai")
    if not isinstance(bonsai.get("mmproj_gpu_offload"), bool):
        raise ConfigError("chat_backends.bonsai.mmproj_gpu_offload must be a boolean")
    gpu_layers = _text(bonsai, "n_gpu_layers", "chat_backends.bonsai")
    if gpu_layers not in {"all", "auto"} and not gpu_layers.isdecimal():
        raise ConfigError(
            "chat_backends.bonsai.n_gpu_layers must be 'all', 'auto', or a non-negative integer"
        )
    if _text(bonsai, "flash_attn", "chat_backends.bonsai") not in {"on", "off", "auto"}:
        raise ConfigError("chat_backends.bonsai.flash_attn must be 'on', 'off', or 'auto'")
    for key_name in ("temperature", "top_p", "min_p", "presence_penalty", "repeat_penalty",
                     "request_timeout_s", "startup_timeout_s"):
        value = bonsai.get(key_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"chat_backends.bonsai.{key_name} must be a number")
        if key_name in {"request_timeout_s", "startup_timeout_s"} and value <= 0:
            raise ConfigError(f"chat_backends.bonsai.{key_name} must be positive")
        if key_name == "temperature" and value < 0:
            raise ConfigError("chat_backends.bonsai.temperature must be non-negative")
        if key_name in {"top_p", "min_p"} and not 0 <= value <= 1:
            raise ConfigError(f"chat_backends.bonsai.{key_name} must be between 0 and 1")
        if key_name == "repeat_penalty" and value <= 0:
            raise ConfigError("chat_backends.bonsai.repeat_penalty must be positive")


_BACKEND_CONFIG_VALIDATORS = {
    "strata": _validate_strata_config,
    "bonsai": _validate_bonsai_config,
}


def validate_config(data: Any, required_backend: str | None = None) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError("the root of config.json must be a JSON object")
    version = data.get("schema_version")
    if isinstance(version, bool) or version != 8:
        raise ConfigError("schema_version must be 8")

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

    backend_configs = _mapping(data, "chat_backends")
    unknown_backends = set(backend_configs) - set(CHAT_BACKENDS)
    if unknown_backends:
        unknown = ", ".join(sorted(str(backend_id) for backend_id in unknown_backends))
        raise ConfigError(f"chat_backends contains unregistered backend(s): {unknown}")
    for backend_id in backend_configs:
        backend_config = _mapping(backend_configs, backend_id)
        backend_definition = get_chat_backend(backend_id)
        validator = _BACKEND_CONFIG_VALIDATORS.get(backend_definition.config_schema)
        if validator is None:
            raise ConfigError(f"no configuration validator is registered for {backend_id}")
        validator(backend_config)
    required_backends = CHAT_BACKENDS if required_backend is None else (required_backend,)
    for backend_id in required_backends:
        if backend_id not in CHAT_BACKENDS:
            raise ConfigError(f"chat backend must be one of: {', '.join(CHAT_BACKENDS)}")
        if backend_id not in backend_configs:
            raise ConfigError(f"chat_backends.{backend_id} is required")
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
    _text(comfy, "input_directory", "comfyui")
    gpu_id = comfy.get("gpu_id")
    if isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0:
        raise ConfigError("comfyui.gpu_id must be a non-negative integer")
    for key_name in ("request_timeout_s", "poll_interval_s"):
        value = comfy.get(key_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(f"comfyui.{key_name} must be a positive number")

    paths = _mapping(data, "paths")
    for key_name in ("runtime_root", "session_root", "output_archive"):
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


def load_config(path: Path | None = None, chat_backend: str = "strata") -> Settings:
    if chat_backend not in CHAT_BACKENDS:
        raise ConfigError(f"chat backend must be one of: {', '.join(CHAT_BACKENDS)}")
    config_path = (path or default_config_path()).resolve()
    try:
        with config_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except OSError as error:
        raise ConfigError(f"cannot read {config_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"invalid JSON in {config_path}: {error}") from error
    validate_config(data, required_backend=chat_backend)
    return Settings(root=config_path.parent, data=data, chat_backend=chat_backend)