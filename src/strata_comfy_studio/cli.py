from __future__ import annotations

import argparse
import json
import stat
import sys
from pathlib import Path
from typing import Any

from .initialize import (
    InitializationError,
    initialize_config,
    configure_listeners,
    migrate_config,
    prepare_runtime_files,
    set_vram_threshold,
)
from .settings import ConfigError, Settings, default_config_path, load_config
from .services import ServiceError, service_running, start_service, stop_service
from .workflow import WorkflowError, build_workflow


def _check_private_config(path: Path) -> None:
    permissions = stat.S_IMODE(path.stat().st_mode)
    if permissions & 0o077:
        raise ConfigError(f"{path} must have owner-only permissions (0600); found {permissions:04o}")


def _require_file(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise ConfigError(f"{label} does not exist or is not a file: {path}")
    return path


def _require_directory(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise ConfigError(f"{label} does not exist or is not a directory: {path}")
    return path


def validate_artifacts(settings: Settings) -> None:
    strata = settings.section("strata")
    comfy = settings.section("comfyui")
    if not strata["api_key"].strip():
        raise ConfigError("strata.api_key is empty; run `python3 studio.py init` or edit config.json")
    if not settings.section("app")["access_token"].strip():
        raise ConfigError("app.access_token is empty; set it in config.json")

    for key, label in (("python", "Strata Python"), ("server_script", "Strata server script"),
                       ("source_config_json", "Strata source config")):
        _require_file(strata[key], label)
    for key, label in (("python", "ComfyUI Python"), ("main_script", "ComfyUI main script")):
        _require_file(comfy[key], label)
    system_prompt_path = _require_file(
        str(settings.path("generation.system_prompt_file")), "Generation system prompt",
    )
    if not system_prompt_path.read_text(encoding="utf-8").strip():
        raise ConfigError("generation system prompt file is empty")
    _require_directory(comfy["working_directory"], "ComfyUI working directory")
    for key, model_key in (("diffusion_models_dir", "diffusion_model"),
                           ("text_encoders_dir", "text_encoder"), ("vae_dir", "vae")):
        directory = _require_directory(comfy[key], f"ComfyUI {key}")
        model_path = directory / comfy["models"][model_key]
        _require_file(str(model_path), f"ComfyUI model {model_key}")
    _require_directory(comfy["output_directory"], "ComfyUI output directory")
    workflow = _require_file(str(settings.path("paths.workflow_api")), "MiniMax API workflow")
    build_workflow(settings, "configuration validation", 1)
    try:
        graph = json.loads(workflow.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError(f"MiniMax API workflow is invalid JSON: {error}") from error
    serialized = json.dumps(graph).lower()
    if "qwen3.5" in serialized or "textgeneratev2" in serialized:
        raise ConfigError("MiniMax API workflow must not include Qwen3.5 prompt enhancement")


def _load_active_config() -> tuple[Path, Settings]:
    path = default_config_path()
    settings = load_config(path)
    _check_private_config(path)
    return path, settings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="studio", description="Configure and validate the local image studio")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init", help="create private config.json and import the Strata API key")
    subparsers.add_parser("prepare", help="generate private Strata runtime config")
    subparsers.add_parser("validate", help="validate config, model files, workflow and runtime paths")
    start = subparsers.add_parser("start", help="start one project-managed local model service")
    start.add_argument("service", choices=("strata", "comfyui"))
    stop = subparsers.add_parser("stop", help="stop one verified project-managed local model service")
    stop.add_argument("service", choices=("strata", "comfyui"))
    status = subparsers.add_parser("status", help="show whether each project-managed service is running")
    status.add_argument("service", choices=("strata", "comfyui", "all"), default="all", nargs="?")
    threshold = subparsers.add_parser("set-vram-threshold", help="set Strata's minimum free VRAM in JSON config")
    threshold.add_argument("mib", type=int)
    listeners = subparsers.add_parser("set-listeners", help="configure the LAN app listener and private Strata port")
    listeners.add_argument("--app-host", required=True)
    listeners.add_argument("--app-port", type=int, required=True)
    listeners.add_argument("--strata-port", type=int, required=True)
    listeners.add_argument("--allow-insecure-lan-http", action="store_true")
    listeners.add_argument("--comfy-port", type=int)
    subparsers.add_parser("migrate-config", help="add new example defaults without overwriting private settings")
    subparsers.add_parser("serve", help="run the authenticated browser application")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            path = initialize_config()
            print(f"Created private configuration: {path}")
            print("The Strata API key and app token were stored without displaying their values.")
            return 0
        if args.command == "set-vram-threshold":
            path = set_vram_threshold(args.mib)
            print(f"Updated strata.min_free_vram_mib in {path}")
            return 0
        if args.command == "set-listeners":
            path = configure_listeners(args.app_host, args.app_port, args.strata_port,
                                       args.allow_insecure_lan_http, comfy_port=args.comfy_port)
            print(f"Updated app and Strata listeners in {path}")
            return 0
        if args.command == "migrate-config":
            path, added = migrate_config()
            print(f"Updated configuration: {path}")
            print(f"Added {len(added)} missing default field(s); existing values and secrets were preserved.")
            return 0
        if args.command == "serve":
            from .app import run
            run()
            return 0
        config_path, settings = _load_active_config()
        if args.command == "prepare":
            validate_artifacts(settings)
            runtime_path = prepare_runtime_files(settings)
            print(f"Prepared private Strata config: {runtime_path}")
            return 0
        if args.command == "start":
            validate_artifacts(settings)
            prepare_runtime_files(settings)
            pid, log_path = start_service(settings, args.service)
            print(f"Started {args.service} with PID {pid}; log: {log_path}")
            return 0
        if args.command == "stop":
            stopped = stop_service(settings, args.service)
            print(f"Stopped {args.service}" if stopped else f"{args.service} was not running")
            return 0
        if args.command == "status":
            services = ("strata", "comfyui") if args.service == "all" else (args.service,)
            for service in services:
                print(f"{service}: {'running' if service_running(settings, service) else 'stopped'}")
            return 0
        validate_artifacts(settings)
        print(f"Configuration and model files are valid: {config_path}")
        print("No services were started and no credentials were displayed.")
        return 0
    except (ConfigError, InitializationError, ServiceError, WorkflowError, OSError) as error:
        print(f"studio: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())