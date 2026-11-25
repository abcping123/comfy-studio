from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from collections.abc import Iterable
from http.client import HTTPException
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from .chat_backends import CHAT_BACKENDS, get_chat_backend
from .permissions import set_private_permissions
from .settings import Settings


class ServiceError(RuntimeError):
    """A managed local service cannot safely be started or stopped."""


def service_command(settings: Settings, service: str) -> tuple[list[str], Path, str, int]:
    if service == "comfyui":
        config = settings.section("comfyui")
        endpoint = urlsplit(config["endpoint"])
        command = [
            str(settings.path("comfyui.python")),
            str(settings.path("comfyui.main_script").resolve()),
            "--listen", endpoint.hostname or "127.0.0.1", "--port", str(endpoint.port),
            "--output-directory", str(settings.path("comfyui.output_directory").resolve()),
            "--input-directory", str(settings.path("comfyui.input_directory").resolve()),
        ]
        cwd = settings.path("comfyui.working_directory").resolve()
    else:
        try:
            backend = get_chat_backend(service)
        except ValueError as error:
            raise ServiceError(f"unknown managed service: {service}") from error
        if not backend.managed_service:
            raise ServiceError(f"chat backend {service} does not manage a local service")
        config = settings.backend_config(service)
        endpoint = urlsplit(config["endpoint"])
        if backend.service_adapter == "strata":
            script = settings.path(f"{service}.server_script").resolve()
            command = [
                str(settings.path(f"{service}.python")),
                str(script), "--engine", "strata",
                "--config", str(settings.path(f"{service}.runtime_config")),
                "--host", endpoint.hostname or "127.0.0.1", "--port", str(endpoint.port),
                "--lazy",
                "--idle-unload", str(config["idle_unload_s"]),
                "--slot-save-path", str(settings.path("paths.runtime_root").resolve()),
            ]
            cwd = script.parent.parent
        elif backend.service_adapter == "llama-server":
            binary = settings.path(f"{service}.server_binary").resolve()
            command = [
                str(binary),
                "--model", str(settings.path(f"{service}.model_file").resolve()),
                "--mmproj", str(settings.path(f"{service}.mmproj_file").resolve()),
                "--mmproj-offload" if config["mmproj_gpu_offload"] else "--no-mmproj-offload",
                "--ctx-size", str(config["ctx_size"]),
                "--n-gpu-layers", str(config["n_gpu_layers"]),
                "--flash-attn", config["flash_attn"],
                "--parallel", str(config["parallel"]),
                "--n-predict", str(config["n_predict"]),
                "--temp", str(config["temperature"]),
                "--top-p", str(config["top_p"]),
                "--top-k", str(config["top_k"]),
                "--min-p", str(config["min_p"]),
                "--presence-penalty", str(config["presence_penalty"]),
                "--repeat-penalty", str(config["repeat_penalty"]),
                "--host", endpoint.hostname or "127.0.0.1",
                "--port", str(endpoint.port),
            ]
            cwd = binary.parent
        else:
            raise ServiceError(
                f"no service command adapter is registered for chat backend {service}"
            )
    return command, cwd, endpoint.hostname or "127.0.0.1", endpoint.port or 0


def _service_environment(settings: Settings, service: str) -> dict[str, str] | None:
    if service != "strata":
        return None
    environment = os.environ.copy()
    environment["STRATA_API_KEY"] = settings.backend_config("strata")["api_key"]
    return environment


def _pid_path(settings: Settings, service: str) -> Path:
    runtime_root = settings.path("paths.runtime_root")
    return runtime_root / f"service_{service}.pid"


def _process_identity(pid: int) -> tuple[list[str], Path] | None:
    if os.name == "nt":
        import psutil

        try:
            process = psutil.Process(pid)
            args = process.cmdline()
            process_cwd = Path(process.cwd()).resolve()
        except (psutil.Error, OSError):
            return None
        return args, process_cwd

    proc = Path("/proc") / str(pid)
    try:
        args = [value.decode("utf-8", errors="replace") for value in (proc / "cmdline").read_bytes().split(b"\0") if value]
        process_cwd = (proc / "cwd").resolve()
    except OSError:
        return None
    return args, process_cwd


def _process_isolated_from_studio(pid: int) -> bool:
    if os.name == "nt":
        return False
    try:
        return os.getpgid(pid) != os.getpgrp()
    except ProcessLookupError:
        return False


def _iter_process_identities() -> Iterable[tuple[int, list[str], Path]]:
    if os.name == "nt":
        import psutil

        for process in psutil.process_iter():
            identity = _process_identity(process.pid)
            if identity is not None:
                yield process.pid, *identity
        return

    try:
        for process_dir in Path("/proc").iterdir():
            if not process_dir.name.isdecimal():
                continue
            pid = int(process_dir.name)
            identity = _process_identity(pid)
            if identity is not None:
                yield pid, *identity
    except OSError as error:
        raise ServiceError(f"could not inspect running processes: {error}") from error


def _option_value(args: list[str], option: str) -> str | None:
    for index, value in enumerate(args):
        if value == option:
            return args[index + 1] if index + 1 < len(args) else None
        if value.startswith(f"{option}="):
            return value[len(option) + 1:]
    return None


def _argument_path_matches(value: str | None, expected: Path, process_cwd: Path) -> bool:
    if value is None:
        return False
    actual = Path(value)
    if not actual.is_absolute():
        actual = process_cwd / actual
    try:
        return actual.resolve() == expected.resolve()
    except (OSError, ValueError):
        return False


def _service_command_matches(
    settings: Settings,
    service: str,
    args: list[str],
    process_cwd: Path,
) -> bool:
    if service != "comfyui" and service not in CHAT_BACKENDS:
        return False
    backend = get_chat_backend(service) if service != "comfyui" else None
    if backend is not None and not backend.managed_service:
        return False
    service_adapter = backend.service_adapter if backend is not None else "comfyui"
    command, _, _, port = service_command(settings, service)
    if service_adapter == "llama-server":
        if (not args
                or not _argument_path_matches(args[0], Path(command[0]), process_cwd)
                or _option_value(args, "--port") != str(port)):
            return False
        for option in ("--model", "--mmproj"):
            expected = _option_value(command, option)
            if expected is None or not _argument_path_matches(
                _option_value(args, option),
                Path(expected),
                process_cwd,
            ):
                return False
        return True

    script_key = "main_script" if service == "comfyui" else "server_script"
    script_path = settings.path(f"{service}.{script_key}").resolve()
    if _option_value(args, "--port") != str(port):
        return False
    if service_adapter == "strata" and _option_value(args, "--engine") != "strata":
        return False

    for argument in args[1:]:
        try:
            argument_path = Path(argument)
            if not argument_path.is_absolute():
                argument_path = process_cwd / argument_path
            if argument_path.resolve() == script_path:
                return True
        except (OSError, ValueError):
            continue
    return False


def _service_process_matches(settings: Settings, service: str, pid: int) -> bool:
    identity = _process_identity(pid)
    return identity is not None and _service_command_matches(settings, service, *identity)


def _find_service_processes(settings: Settings, service: str) -> list[int]:
    if service != "comfyui" and service not in CHAT_BACKENDS:
        return []
    if service != "comfyui" and not get_chat_backend(service).managed_service:
        return []
    return [
        pid
        for pid, args, process_cwd in _iter_process_identities()
        if _service_command_matches(settings, service, args, process_cwd)
    ]


def _process_matches(pid: int, command: list[str], cwd: Path) -> bool:
    identity = _process_identity(pid)
    return identity is not None and identity == (command, cwd.resolve())


def _pid_mismatch_error(
    pid_file: Path,
    pid: int,
    service: str,
    command: list[str],
    cwd: Path,
) -> ServiceError:
    identity = _process_identity(pid)
    if identity is None:
        details = f"could not read process details; expected cwd={cwd.resolve()}"
    else:
        actual_args, actual_cwd = identity
        mismatch_index = next(
            (index for index, (actual, expected) in enumerate(zip(actual_args, command))
             if actual != expected),
            min(len(actual_args), len(command)),
        )
        if actual_args == command:
            argv_details = "arguments match exactly"
        else:
            argv_details = (
                f"arguments differ at index {mismatch_index} "
                f"(actual {len(actual_args)} args, expected {len(command)} args)"
            )
        details = f"{argv_details}; actual cwd={actual_cwd}, expected cwd={cwd.resolve()}"
    return ServiceError(
        f"pid file {pid_file} points to PID {pid}, but it does not identify the expected "
        f"{service} process ({details}). No process was stopped and the PID file was left "
        f"untouched. To inspect arguments locally, run `ps -p {pid} -o pid=,lstart=,args=` "
        f"and `readlink /proc/{pid}/cwd`; full arguments are omitted here to avoid echoing secrets."
    )


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def _pid_exists(pid: int) -> bool:
    if os.name == "nt":
        import psutil

        try:
            return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False
        except psutil.Error:
            return True

    proc = Path("/proc") / str(pid)
    try:
        stat = (proc / "stat").read_text(encoding="ascii")
    except FileNotFoundError:
        return False
    except OSError:
        return proc.exists()
    _, separator, details = stat.rpartition(")")
    if not separator:
        return proc.exists()
    fields = details.split()
    if not fields:
        return proc.exists()
    return fields[0] not in {"Z", "X"}


def _terminate_process(pid: int) -> None:
    if os.name == "nt":
        import psutil

        try:
            psutil.Process(pid).terminate()
        except psutil.NoSuchProcess:
            return
        except psutil.Error as error:
            raise ServiceError(f"could not terminate process {pid}: {error}") from error
        return
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError as error:
        raise ServiceError(f"could not terminate process {pid}: {error}") from error


def _strata_endpoint_matches(settings: Settings, service: str = "strata") -> bool:
    config = settings.backend_config(service)
    api_key = config.get("api_key")
    if not isinstance(api_key, str) or not api_key.strip():
        return False
    request = Request(
        f"{config['endpoint'].rstrip('/')}/v1/models",
        headers={"Accept": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=1.0) as response:
            raw = response.read(65537)
        if len(raw) > 65536:
            return False
        payload = json.loads(raw)
    except (HTTPException, OSError, URLError, ValueError):
        return False
    models = payload.get("data") if isinstance(payload, dict) else None
    return isinstance(models, list) and any(
        isinstance(model, dict) and model.get("id") == config["model"] for model in models
    )


def _reject_unmanageable_endpoint_backend(settings: Settings, service: str) -> None:
    if service not in CHAT_BACKENDS:
        return
    backend = get_chat_backend(service)
    if backend.endpoint_probe == "strata-model" and _strata_endpoint_matches(settings, service):
        raise ServiceError(
            f"{backend.label} is responding with the configured model, but no "
            "matching local process was found; it cannot be taken over safely or stopped on Ctrl+C"
        )


def _check_service_start(
    settings: Settings,
    service: str,
) -> tuple[int | None, list[int]]:
    _, _, host, port = service_command(settings, service)
    pid_file = _pid_path(settings, service)
    old_pid: int | None = None

    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            pass

    matching_pids = _find_service_processes(settings, service)
    if matching_pids:
        if (len(matching_pids) == 1
                and matching_pids[0] == old_pid
                and _process_isolated_from_studio(old_pid)):
            return old_pid, []
        return None, matching_pids

    if _port_open(host, port):
        _reject_unmanageable_endpoint_backend(settings, service)
        command, cwd, _, _ = service_command(settings, service)
        if old_pid is not None and _pid_exists(old_pid):
            raise _pid_mismatch_error(pid_file, old_pid, service, command, cwd)
        raise ServiceError(f"refusing to start {service}: {host}:{port} is already in use")

    return None, []


def preflight_service_start(settings: Settings, services: Iterable[str]) -> None:
    errors: list[str] = []
    for service in services:
        try:
            _check_service_start(settings, service)
        except (OSError, ServiceError) as error:
            errors.append(f"{service}: {error}")
    if errors:
        details = "\n - ".join(errors)
        raise ServiceError(f"startup preflight failed; no services were started:\n - {details}")


def preflight_studio_start(settings: Settings) -> None:
    config = settings.section("app")
    host = config["listen_host"]
    port = config["port"]
    if _port_open(host, port):
        raise ServiceError(f"refusing to start studio: {host}:{port} is already in use")


def service_running(settings: Settings, service: str) -> bool:
    if service != "comfyui":
        if service not in CHAT_BACKENDS or not get_chat_backend(service).managed_service:
            return False
    pid_file = _pid_path(settings, service)
    try:
        pid = int(pid_file.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        pid = None
    if pid is not None and _service_process_matches(settings, service, pid):
        return True
    if _find_service_processes(settings, service):
        return True
    return (
        service in CHAT_BACKENDS
        and get_chat_backend(service).endpoint_probe == "strata-model"
        and _strata_endpoint_matches(settings, service)
    )


def start_service(settings: Settings, service: str) -> tuple[int | None, Path]:
    pid, log_path, _ = start_service_tracked(settings, service)
    return pid, log_path


def _stop_service_processes(
    settings: Settings,
    service: str,
    pids: Iterable[int],
    wait_s: float,
    wait_for_port: bool = False,
) -> None:
    _, _, host, port = service_command(settings, service)
    process_ids = list(dict.fromkeys(pids))
    for pid in process_ids:
        if _service_process_matches(settings, service, pid):
            _terminate_process(pid)

    deadline = time.monotonic() + wait_s
    while True:
        active_pids = [
            pid for pid in process_ids
            if _service_process_matches(settings, service, pid)
        ]
        port_open = wait_for_port and _port_open(host, port)
        if not active_pids and not port_open:
            return
        if time.monotonic() >= deadline:
            if active_pids:
                raise ServiceError(
                    f"{service} is still running after SIGTERM; no force kill was sent"
                )
            raise ServiceError(
                f"{service} endpoint {host}:{port} is still in use after SIGTERM"
            )
        time.sleep(0.1)


def start_service_tracked(
    settings: Settings,
    service: str,
) -> tuple[int | None, Path, bool]:
    command, cwd, _, _ = service_command(settings, service)
    environment = _service_environment(settings, service)
    pid_file = _pid_path(settings, service)
    log_path = pid_file.with_suffix(".log")
    pid_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    set_private_permissions(pid_file.parent, directory=True)
    existing_pid, restart_pids = _check_service_start(settings, service)
    if existing_pid is not None:
        return existing_pid, log_path, False
    if restart_pids:
        _stop_service_processes(
            settings,
            service,
            restart_pids,
            wait_s=10.0,
            wait_for_port=True,
        )

    with log_path.open("ab") as log:
        if os.name == "nt":
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
    if process.poll() is not None:
        raise ServiceError(f"{service} exited during startup; inspect {log_path}")
    try:
        with pid_file.open("w", encoding="ascii") as file:
            file.write(f"{process.pid}\n")
        set_private_permissions(pid_file)
    except OSError as error:
        if _process_matches(process.pid, command, cwd):
            _terminate_process(process.pid)
        raise ServiceError(f"could not record the {service} pid: {error}") from error
    return process.pid, log_path, True


def stop_service(settings: Settings, service: str, wait_s: float = 10.0) -> bool | None:
    command, cwd, _, _ = service_command(settings, service)
    pid_file = _pid_path(settings, service)
    matching_pids = _find_service_processes(settings, service)
    if matching_pids:
        _stop_service_processes(settings, service, matching_pids, wait_s)
        pid_file.unlink(missing_ok=True)
        return True

    if not pid_file.exists():
        if (service in CHAT_BACKENDS
                and get_chat_backend(service).endpoint_probe == "strata-model"
                and _strata_endpoint_matches(settings, service)):
            return None
        return False
    try:
        pid = int(pid_file.read_text(encoding="ascii").strip())
    except ValueError as error:
        if (service in CHAT_BACKENDS
                and get_chat_backend(service).endpoint_probe == "strata-model"
                and _strata_endpoint_matches(settings, service)):
            return None
        raise ServiceError(f"invalid pid file; leaving it untouched: {pid_file}") from error
    if not _service_process_matches(settings, service, pid):
        if not _pid_exists(pid):
            pid_file.unlink(missing_ok=True)
            if (service in CHAT_BACKENDS
                    and get_chat_backend(service).endpoint_probe == "strata-model"
                    and _strata_endpoint_matches(settings, service)):
                return None
            return False
        if (service in CHAT_BACKENDS
                and get_chat_backend(service).endpoint_probe == "strata-model"
                and _strata_endpoint_matches(settings, service)):
            return None
        raise _pid_mismatch_error(pid_file, pid, service, command, cwd)

    _terminate_process(pid)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if not _service_process_matches(settings, service, pid):
            pid_file.unlink(missing_ok=True)
            return True
        time.sleep(0.1)
    raise ServiceError(f"{service} is still running after SIGTERM; no force kill was sent")