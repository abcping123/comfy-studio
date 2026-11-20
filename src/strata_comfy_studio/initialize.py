from __future__ import annotations

import json
import os
import secrets
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


def initialize_config(template_path: Path | None = None, output_path: Path | None = None,
                      strata_repo_path: Path | None = None,
                      comfyui_repo_path: Path | None = None) -> Path:
    template = (template_path or project_root() / "data" / "config.example.json").resolve()
    destination = (output_path or default_config_path()).resolve()
    data = _read_json(template)
    source_value = data.get("strata", {}).get("source_config_json")
    if not isinstance(source_value, str) or not source_value.strip():
        raise InitializationError("strata.source_config_json must point to the existing Strata JSON config")
    if (strata_repo_path is None) != (comfyui_repo_path is None):
        raise InitializationError("both Strata and ComfyUI repository paths are required")
    if strata_repo_path is not None and comfyui_repo_path is not None:
        strata_repo = strata_repo_path.expanduser().resolve()
        comfyui_repo = comfyui_repo_path.expanduser().resolve()
        for repository, label in ((strata_repo, "Strata"), (comfyui_repo, "ComfyUI")):
            if not repository.is_dir():
                raise InitializationError(f"{label} repository does not exist: {repository}")

            terminal_root = strata_repo / "terminal"
            strata_root = terminal_root if (terminal_root / "serve/server.py").is_file() else strata_repo
            strata_python = strata_root / ".venv/bin/python"
            strata_server = strata_root / "serve/server.py"
            source_path = strata_root / Path(source_value).name
        comfy_python = comfyui_repo / ".venv/bin/python"
        comfy_main = comfyui_repo / "main.py"
        required_files = (
            (strata_python, "Strata Python"),
            (strata_server, "Strata server script"),
            (source_path, "Strata source config"),
            (comfy_python, "ComfyUI Python"),
            (comfy_main, "ComfyUI main script"),
        )
        for path, label in required_files:
            if not path.is_file():
                raise InitializationError(f"{label} does not exist: {path}")
        if not os.access(strata_python, os.X_OK) or not os.access(comfy_python, os.X_OK):
            raise InitializationError("repository Python interpreters must be executable")

        data["strata"].update({
            "python": str(strata_python),
            "server_script": str(strata_server),
            "source_config_json": str(source_path),
        })
        data["comfyui"].update({
            "python": str(comfy_python),
            "main_script": str(comfy_main),
            "working_directory": str(comfyui_repo),
            "output_directory": str(comfyui_repo / "output"),
        })

    configured_source = data["strata"]["source_config_json"]
    source_path = Path(configured_source)
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