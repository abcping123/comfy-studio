from __future__ import annotations

import json
import os
import secrets
import copy
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .settings import ConfigError, Settings, default_config_path, project_root, validate_config


class InitializationError(RuntimeError):
    """Local configuration cannot be initialized safely."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise InitializationError(f"cannot read JSON config {path}: {error}") from error
    if not isinstance(data, dict):
        raise InitializationError(f"JSON config root must be an object: {path}")
    return data


def _write_private_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise InitializationError(f"refusing to overwrite existing file: {path}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.chmod(path, 0o600)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _replace_private_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    _write_private_json(temporary, data)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def _merge_missing(defaults: Any, current: Any, prefix: str = "") -> tuple[Any, list[str]]:
    if not isinstance(defaults, dict) or not isinstance(current, dict):
        return current, []
    merged = dict(current)
    added: list[str] = []
    for key, default_value in defaults.items():
        location = f"{prefix}.{key}" if prefix else key
        if key not in merged:
            merged[key] = copy.deepcopy(default_value)
            added.append(location)
        elif isinstance(default_value, dict) and isinstance(merged[key], dict):
            merged[key], nested = _merge_missing(default_value, merged[key], location)
            added.extend(nested)
    return merged, added


def initialize_config(template_path: Path | None = None, output_path: Path | None = None) -> Path:
    template = (template_path or project_root() / "config.example.json").resolve()
    destination = (output_path or default_config_path()).resolve()
    data = _read_json(template)
    source_value = data.get("strata", {}).get("source_config_json")
    if not isinstance(source_value, str) or not source_value.strip():
        raise InitializationError("strata.source_config_json must point to the existing Strata JSON config")
    source_path = Path(source_value)
    if not source_path.is_absolute():
        source_path = template.parent / source_path
    source = _read_json(source_path)
    api_key = source.get("api_key")
    if not isinstance(api_key, str) or not api_key.strip():
        raise InitializationError(f"the Strata JSON config has no API key: {source_path}")

    data["strata"]["api_key"] = api_key
    if not data["app"].get("access_token"):
        data["app"]["access_token"] = secrets.token_urlsafe(32)
    try:
        validate_config(data)
    except ConfigError as error:
        raise InitializationError(str(error)) from error
    _write_private_json(destination, data)
    return destination


def set_vram_threshold(mib: int, config_path: Path | None = None) -> Path:
    if isinstance(mib, bool) or not isinstance(mib, int) or mib < 0:
        raise InitializationError("VRAM threshold must be a non-negative integer number of MiB")
    destination = (config_path or default_config_path()).resolve()
    data = _read_json(destination)
    data.setdefault("strata", {})["min_free_vram_mib"] = mib
    try:
        validate_config(data)
    except ConfigError as error:
        raise InitializationError(str(error)) from error
    _replace_private_json(destination, data)
    return destination


def configure_listeners(app_host: str, app_port: int, strata_port: int,
                        allow_insecure_lan_http: bool = False,
                        config_path: Path | None = None, comfy_port: int | None = None) -> Path:
    destination = (config_path or default_config_path()).resolve()
    data = _read_json(destination)
    data["app"].update({
        "listen_host": app_host,
        "port": app_port,
        "allow_insecure_lan_http": allow_insecure_lan_http,
    })
    data["strata"]["endpoint"] = f"http://127.0.0.1:{strata_port}"
    if comfy_port is not None:
        comfy_endpoint = urlsplit(data["comfyui"]["endpoint"])
        if not comfy_endpoint.hostname or comfy_endpoint.port is None:
            raise InitializationError("comfyui.endpoint must include a host and port")
        host = f"[{comfy_endpoint.hostname}]" if ":" in comfy_endpoint.hostname else comfy_endpoint.hostname
        data["comfyui"]["endpoint"] = urlunsplit((comfy_endpoint.scheme,
                                                   f"{host}:{comfy_port}", "", "", ""))
    try:
        validate_config(data)
    except ConfigError as error:
        raise InitializationError(str(error)) from error
    _replace_private_json(destination, data)
    return destination


def migrate_config(template_path: Path | None = None, config_path: Path | None = None) -> tuple[Path, list[str]]:
    template = (template_path or project_root() / "config.example.json").resolve()
    destination = (config_path or default_config_path()).resolve()
    current = _read_json(destination)
    defaults = _read_json(template)
    current_generation = current.get("generation", {})
    default_generation = defaults.get("generation", {})
    current_prompt_version = current_generation.get("prompt_skill_version", 0) if isinstance(current_generation, dict) else 0
    default_prompt_version = default_generation.get("prompt_skill_version", 0) if isinstance(default_generation, dict) else 0
    merged, added = _merge_missing(defaults, current)
    for obsolete in ("default_width", "default_height", "sampler_name", "scheduler"):
        if obsolete in merged.get("generation", {}):
            merged["generation"].pop(obsolete)
            added.append(f"generation.{obsolete} (obsolete)")
    if "system_prompt" in merged.get("generation", {}):
        merged["generation"].pop("system_prompt")
        added.append("generation.system_prompt (moved to file)")
    if isinstance(default_prompt_version, int) and isinstance(current_prompt_version, int) \
            and default_prompt_version > current_prompt_version:
        merged["generation"]["system_prompt_file"] = default_generation["system_prompt_file"]
        merged["generation"]["prompt_skill_version"] = default_prompt_version
        added.append("generation.system_prompt_file (prompt skill update)")
    try:
        validate_config(merged)
    except ConfigError as error:
        raise InitializationError(str(error)) from error
    if added:
        _replace_private_json(destination, merged)
    return destination, added


def _validate_single_slot_source(source: dict[str, Any], source_path: Path) -> None:
    args = source.get("args")
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise InitializationError(f"Strata config args must be a list of strings: {source_path}")
    for arg in args:
        if arg == "--batch" or arg.startswith("--batch=") or arg in {"--layer-split", "--peer-device"}:
            raise InitializationError(f"Strata session save/restore is incompatible with {arg}")
    for index, arg in enumerate(args[:-1]):
        if arg == "--prompt-cache" and args[index + 1] == "0":
            raise InitializationError("Strata session save/restore requires prompt cache to be enabled")


def prepare_runtime_files(settings: Settings) -> Path:
    """Create the private Strata runtime JSON from config.json."""
    strata = settings.section("strata")
    comfy = settings.section("comfyui")
    endpoint = urlsplit(strata["endpoint"])
    source_path = Path(strata["source_config_json"]).resolve()
    source = _read_json(source_path)
    _validate_single_slot_source(source, source_path)

    runtime = dict(source)
    runtime.pop("env", None)
    runtime.update({
        "api_key": strata["api_key"],
        "host": endpoint.hostname,
        "port": endpoint.port,
        "slot_save_path": str(settings.path("paths.strata_sessions").resolve()),
        "idle_unload_s": strata["idle_unload_s"],
        "lazy_load": True,
    })
    if strata["min_free_vram_mib"] is None:
        runtime.pop("min_free_vram_mib", None)
    else:
        runtime["min_free_vram_mib"] = strata["min_free_vram_mib"]

    runtime_path = settings.path("paths.strata_runtime_config")
    sessions_path = settings.path("paths.strata_sessions")
    temp_root = settings.path("paths.temp_root")
    for directory in (runtime_path.parent, sessions_path, temp_root / "comfyui"):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if directory == sessions_path:
            os.chmod(directory, 0o700)

    _replace_private_json(runtime_path, runtime)
    if not comfy["models"]["text_encoder"].startswith("qwen3vl_"):
        raise InitializationError("comfyui.models.text_encoder must be the MiniMax Qwen3-VL encoder")
    return runtime_path