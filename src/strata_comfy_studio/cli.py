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
    prepare_runtime_files,
)
from .settings import ConfigError, Settings, default_config_path, load_config
from .services import ServiceError, preflight_service_start, service_running, start_service, stop_service
from .workflow import WorkflowError, build_workflow

MANAGED_SERVICES = ("comfyui", "strata", "studio")


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
        raise ConfigError("strata.api_key is empty; set it in config.json")
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
    edit_prompt_path = _require_file(
        str(settings.path("generation.edit_system_prompt_file")), "Image edit system prompt",
    )
    if not edit_prompt_path.read_text(encoding="utf-8").strip():
        raise ConfigError("image edit system prompt file is empty")
    _require_directory(comfy["working_directory"], "ComfyUI working directory")
    for key, model_key in (("diffusion_models_dir", "diffusion_model"),
                           ("text_encoders_dir", "text_encoder"), ("vae_dir", "vae")):
        directory = _require_directory(comfy[key], f"ComfyUI {key}")
        model_path = directory / comfy["models"][model_key]
        _require_file(str(model_path), f"ComfyUI model {model_key}")
    _require_directory(str(settings.path("comfyui.output_directory")), "ComfyUI output directory")
    _require_directory(str(settings.path("comfyui.input_directory")), "ComfyUI input directory")
    workflow = _require_file(str(settings.path("paths.workflow_api")), "MiniMax API workflow")
    edit_workflow = _require_file(str(settings.path("paths.workflow_edit_api")), "MiniMax edit API workflow")
    build_workflow(settings, "configuration validation", 1)
    build_workflow(settings, "configuration validation", 1, image_mode="edit",
                   source_images=[{
                       "filename": "configuration-validation.png",
                       "subfolder": "0" * 32,
                       "width": 1024,
                       "height": 1024,
                   }])
    try:
        graph = json.loads(workflow.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError(f"MiniMax API workflow is invalid JSON: {error}") from error
    serialized = json.dumps(graph).lower()
    if "qwen3.5" in serialized or "textgeneratev2" in serialized:
        raise ConfigError("MiniMax API workflow must not include Qwen3.5 prompt enhancement")
    try:
        edit_graph = json.loads(edit_workflow.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError(f"MiniMax edit API workflow is invalid JSON: {error}") from error
    edit_serialized = json.dumps(edit_graph).lower()
    if "qwen3.5" in edit_serialized or "textgeneratev2" in edit_serialized:
        raise ConfigError("MiniMax edit API workflow must not include Qwen3.5 prompt enhancement")


def _prompt_repository_path(label: str) -> Path:
    try:
        value = input(f"{label} repository path: ").strip()
    except EOFError as error:
        raise InitializationError("first startup requires both repository paths") from error
    if not value:
        raise InitializationError(f"{label} repository path is required for first startup")
    return Path(value).expanduser()


def _load_active_config(initialize_if_missing: bool = False) -> tuple[Path, Settings]:
    path = default_config_path()
    if initialize_if_missing and not path.exists():
        print("No config.json found. Enter the local repository paths to configure the studio.")
        strata_repo = _prompt_repository_path("Strata")
        comfyui_repo = _prompt_repository_path("ComfyUI")
        initialize_config(
            output_path=path,
            strata_repo_path=strata_repo,
            comfyui_repo_path=comfyui_repo,
        )
        print("Created private config.json.")
    settings = load_config(path)
    _check_private_config(path)
    return path, settings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="studio", description="Manage the local studio services")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("start", help="start all project-managed services")
    subparsers.add_parser("status", help="show status for all project-managed services")
    subparsers.add_parser("stop", help="stop all verified project-managed services")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config_path, settings = _load_active_config(initialize_if_missing=args.command == "start")
        if args.command == "start":
            prepare_runtime_files(settings)
            validate_artifacts(settings)
            preflight_service_start(settings, MANAGED_SERVICES)
            failed = False
            for service in MANAGED_SERVICES:
                try:
                    pid, log_path = start_service(settings, service)
                except (ServiceError, OSError) as error:
                    print(f"studio: failed to start {service}: {error}", file=sys.stderr)
                    failed = True
                    continue
                if pid is None:
                    print(f"Using existing {service} service; it is not managed by Studio")
                else:
                    print(f"Started {service} with PID {pid}; log: {log_path}")
            return 2 if failed else 0
        if args.command == "stop":
            failed = False
            for service in reversed(MANAGED_SERVICES):
                try:
                    stopped = stop_service(settings, service)
                except (ServiceError, OSError) as error:
                    print(f"studio: failed to stop {service}: {error}", file=sys.stderr)
                    failed = True
                    continue
                if stopped is None:
                    print(f"{service} is running outside Studio; left untouched")
                else:
                    print(f"Stopped {service}" if stopped else f"{service} was not running")
            return 2 if failed else 0
        if args.command == "status":
            for service in MANAGED_SERVICES:
                print(f"{service}: {'running' if service_running(settings, service) else 'stopped'}")
            return 0
    except (ConfigError, InitializationError, ServiceError, WorkflowError, OSError) as error:
        print(f"studio: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())