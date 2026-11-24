from __future__ import annotations

import os
import sys
from pathlib import Path

from .app import run as run_app
from .initialize import InitializationError, initialize_config, prepare_runtime_directories
from .chat_backends import CHAT_BACKEND_REGISTRY, get_chat_backend
from .modes import IMAGE_MODES
from .permissions import set_private_permissions
from .settings import CHAT_BACKENDS, ConfigError, Settings, default_config_path, load_config
from .services import (
    ServiceError,
    preflight_studio_start,
    preflight_service_start,
    start_service_tracked,
    service_running,
    stop_service,
)
from .workflow import WorkflowError, build_workflow

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


def validate_artifacts(settings: Settings, chat_backend: str | None = None) -> None:
    chat_backend = chat_backend or settings.chat_backend
    if chat_backend not in CHAT_BACKENDS:
        raise ConfigError(f"chat backend must be one of: {', '.join(CHAT_BACKENDS)}")
    backend_definition = get_chat_backend(chat_backend)
    backend_config = settings.backend_config(chat_backend)
    if not settings.section("app")["access_token"].strip():
        raise ConfigError("app.access_token is empty; set it in config.json")

    if backend_definition.requires_api_key:
        api_key = backend_config.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ConfigError(f"chat_backends.{chat_backend}.api_key is empty; set it in config.json")
    for key, label in backend_definition.required_files:
        path = _require_file(str(settings.path(f"{chat_backend}.{key}")), label)
        if key in backend_definition.executable_files and os.name != "nt" and not os.access(path, os.X_OK):
            raise ConfigError(f"{label} is not executable: {path}")

    for key, label in (("python", "ComfyUI Python"), ("main_script", "ComfyUI main script")):
        _require_file(str(settings.path(f"comfyui.{key}")), label)
    for mode in IMAGE_MODES:
        system_prompt_path = _require_file(
            str(settings.path(f"generation_modes.{mode.id}.system_prompt_file")),
            f"{mode.label} system prompt",
        )
        if not system_prompt_path.read_text(encoding="utf-8").strip():
            raise ConfigError(f"{mode.label} system prompt file is empty")
    _require_directory(
        str(settings.path("comfyui.working_directory")),
        "ComfyUI working directory",
    )
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
        raise InitializationError("first startup requires repository paths for the selected backend") from error
    if not value:
        raise InitializationError(f"{label} repository path is required for first startup")
    return Path(value).expanduser()


def _load_active_config(initialize_if_missing: bool = False,
                        chat_backend: str = "strata") -> tuple[Path, Settings]:
    path = default_config_path()
    if initialize_if_missing and not path.exists():
        print("No config.json found. Enter the local repository paths to configure the studio.")
        setup_adapter = get_chat_backend(chat_backend).setup_adapter
        strata_repo = _prompt_repository_path("Strata") if setup_adapter == "strata" else None
        comfyui_repo = _prompt_repository_path("ComfyUI")
        initialize_config(
            output_path=path,
            strata_repo_path=strata_repo,
            comfyui_repo_path=comfyui_repo,
            chat_backend=chat_backend,
        )
        print("Created private config.json.")
    _check_private_config(path)
    settings = load_config(path, chat_backend=chat_backend)
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


def _stop_unselected_backend(settings: Settings, selected_backend: str) -> None:
    for backend_id, backend_definition in CHAT_BACKEND_REGISTRY.items():
        if backend_id == selected_backend or not backend_definition.managed_service:
            continue
        if not service_running(settings, backend_id):
            continue
        stopped = stop_service(settings, backend_id)
        if stopped is None or service_running(settings, backend_id):
            raise ServiceError(
                f"could not stop the unselected {backend_id} backend"
            )
        if stopped:
            print(f"Stopped unselected {backend_id} backend.")


def _start_foreground(settings: Settings, chat_backend: str = "strata") -> int:
    managed_services: list[str] = []
    backend_definition = get_chat_backend(chat_backend)
    services = ("comfyui", chat_backend) if backend_definition.managed_service else ("comfyui",)
    all_managed_services = ("comfyui",) + tuple(
        backend_id
        for backend_id, definition in CHAT_BACKEND_REGISTRY.items()
        if definition.managed_service
    )
    exit_code = 0
    try:
        if backend_definition.runtime_preparer is not None:
            backend_definition.runtime_preparer(settings)
        else:
            prepare_runtime_directories(settings)
        validate_artifacts(settings, chat_backend)
        preflight_service_start(settings, all_managed_services)
        preflight_studio_start(settings)
        _stop_unselected_backend(settings, chat_backend)
        for service in services:
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
                print(f"Reusing tracked {service} with PID {pid}; it will stop with Studio")

        if exit_code == 0:
            print("Starting Studio in the foreground; press Ctrl+C to stop managed services.")
            run_app(settings)
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
    if arguments in (["-h"], ["--help"]):
        print("usage: main.py [chat-backend-id]")
        return 0
    if len(arguments) > 1 or (arguments and arguments[0] not in CHAT_BACKENDS):
        print("usage: main.py [chat-backend-id]", file=sys.stderr)
        return 2
    chat_backend = arguments[0] if arguments else "strata"
    try:
        _, settings = _load_active_config(
            initialize_if_missing=True,
            chat_backend=chat_backend,
        )
        return _start_foreground(settings, chat_backend)
    except (ConfigError, InitializationError, ServiceError, WorkflowError, OSError) as error:
        print(f"studio: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())