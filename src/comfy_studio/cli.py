from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from .initialize import (
    InitializationError,
    initialize_config,
    prepare_runtime_files,
)
from .modes import IMAGE_MODES
from .permissions import set_private_permissions
from .settings import ConfigError, Settings, default_config_path, load_config
from .services import (
    ServiceError,
    preflight_service_start,
    service_running,
    start_service_tracked,
    stop_service,
)
from .workflow import WorkflowError, build_workflow

MANAGED_SERVICES = ("comfyui", "strata", "studio")


def _check_private_config(path: Path) -> None:
    try:
        set_private_permissions(path)
    except OSError as error:
        raise ConfigError(f"{path} must have owner-only permissions (0600): {error}") from error


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
    for mode in IMAGE_MODES:
        system_prompt_path = _require_file(
            str(settings.path(f"generation_modes.{mode.id}.system_prompt_file")),
            f"{mode.label} system prompt",
        )
        if not system_prompt_path.read_text(encoding="utf-8").strip():
            raise ConfigError(f"{mode.label} system prompt file is empty")
    _require_directory(comfy["working_directory"], "ComfyUI working directory")
    checked_models: set[Path] = set()
    for mode in IMAGE_MODES:
        for model_key, model in settings.data["generation_modes"][mode.id]["models"].items():
            directory = settings.path(f"comfyui.model_directories.{model['directory']}")
            directory = _require_directory(str(directory), f"ComfyUI model directory {model['directory']}")
            model_path = directory / model["file"]
            if model_path not in checked_models:
                _require_file(str(model_path), f"ComfyUI model {model_key}")
                checked_models.add(model_path)
    _require_directory(str(settings.path("comfyui.output_directory")), "ComfyUI output directory")
    _require_directory(str(settings.path("comfyui.input_directory")), "ComfyUI input directory")
    for mode in IMAGE_MODES:
        _require_file(
            str(settings.path(f"generation_modes.{mode.id}.workflow")),
            f"{mode.label} workflow",
        )
        sample_images = [
            {
                "filename": f"configuration-validation-{index}.png",
                "subfolder": "0" * 32,
                "width": 1024,
                "height": 1024,
            }
            for index in range(mode.source_image_min_count)
        ]
        build_workflow(
            settings,
            "configuration validation",
            1,
            image_mode=mode.id,
            source_images=sample_images,
        )


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
    _check_private_config(path)
    settings = load_config(path)
    return path, settings


def _stop_managed_services(settings: Settings, services: list[str]) -> bool:
    failed = False
    for service in reversed(services):
        try:
            stopped = stop_service(settings, service)
        except (ServiceError, OSError) as error:
            print(f"studio: failed to stop {service}: {error}", file=sys.stderr)
            failed = True
            continue
        if stopped is None:
            print(
                f"studio: failed to stop {service}; its process is no longer identifiable",
                file=sys.stderr,
            )
            failed = True
        else:
            print(f"Stopped {service}" if stopped else f"{service} was not running")
    return failed


def _start_foreground(settings: Settings) -> int:
    managed_services: list[str] = []
    exit_code = 0
    try:
        prepare_runtime_files(settings)
        validate_artifacts(settings)
        preflight_service_start(settings, MANAGED_SERVICES)
        for service in MANAGED_SERVICES:
            try:
                pid, log_path, started_here = start_service_tracked(settings, service)
            except (ServiceError, OSError) as error:
                print(f"studio: failed to start {service}: {error}", file=sys.stderr)
                exit_code = 2
                break

            if pid is None:
                raise ServiceError(f"{service} has no identifiable process to supervise")
            managed_services.append(service)
            if started_here:
                print(f"Started {service} with PID {pid}; log: {log_path}")
            else:
                print(f"Took over existing {service} with PID {pid}; it will stop with Studio")

        if exit_code == 0:
            if "studio" not in managed_services:
                raise ServiceError("Studio was not started or taken over by this invocation")
            if not service_running(settings, "studio"):
                raise ServiceError("Studio exited before foreground supervision began")

            print("Studio is running in the foreground; press Ctrl+C to stop managed services.")
            while service_running(settings, "studio"):
                time.sleep(0.5)
            print("Studio stopped; stopping all managed services.")
    except KeyboardInterrupt:
        print("\nCtrl+C received; stopping all managed services.")
        exit_code = 130
    except (ConfigError, InitializationError, ServiceError, WorkflowError, OSError) as error:
        print(f"studio: {error}", file=sys.stderr)
        exit_code = 2
    finally:
        if _stop_managed_services(settings, managed_services):
            exit_code = 2
    return exit_code


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments:
        print("studio: no arguments are supported; run main.py without arguments.", file=sys.stderr)
        return 2
    try:
        _, settings = _load_active_config(initialize_if_missing=True)
        return _start_foreground(settings)
    except (ConfigError, InitializationError, ServiceError, WorkflowError, OSError) as error:
        print(f"studio: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())