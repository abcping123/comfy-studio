from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterable
from http.client import HTTPException
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from .permissions import set_private_permissions
from .settings import Settings, project_root


class ServiceError(RuntimeError):
    """A managed local service cannot safely be started or stopped."""


def service_command(settings: Settings, service: str) -> tuple[list[str], Path, str, int]:
    if service == "studio":
        config = settings.section("app")
        return ([sys.executable, "-m", "comfy_studio.app"], project_root(),
                config["listen_host"], config["port"])
    if service == "strata":
        config = settings.section("strata")
        endpoint = urlsplit(config["endpoint"])
        command = [
            config["python"], config["server_script"], "--engine", "strata",
            "--config", str(settings.path("paths.strata_runtime_config")),
            "--host", endpoint.hostname or "127.0.0.1", "--port", str(endpoint.port),
        ]
        cwd = Path(config["server_script"]).resolve().parent.parent
    elif service == "comfyui":
        config = settings.section("comfyui")
        endpoint = urlsplit(config["endpoint"])
        command = [
            config["python"], config["main_script"],
            "--listen", endpoint.hostname or "127.0.0.1", "--port", str(endpoint.port),
            "--output-directory", str(settings.path("comfyui.output_directory").resolve()),
            "--input-directory", str(settings.path("comfyui.input_directory").resolve()),
        ]
        cwd = Path(config["working_directory"]).resolve()
    else:
        raise ServiceError(f"unknown managed service: {service}")
    endpoint = urlsplit(settings.section(service)["endpoint"])
    return command, cwd, endpoint.hostname or "127.0.0.1", endpoint.port or 0


def _pid_path(settings: Settings, service: str) -> Path:
    temp_root = settings.path("paths.temp_root")
    service_dir = temp_root if service == "studio" else temp_root / service
    return service_dir / "service.pid"


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


def _service_command_matches(
    settings: Settings,
    service: str,
    args: list[str],
    process_cwd: Path,
) -> bool:
    if service not in ("studio", "strata", "comfyui"):
        return False
    command, service_cwd, _, port = service_command(settings, service)
    if service == "studio":
        return len(args) == len(command) and args[1:] == command[1:] and process_cwd == service_cwd

    config = settings.section(service)
    script_key = "server_script" if service == "strata" else "main_script"
    script_path = Path(config[script_key])
    if not script_path.is_absolute():
        script_path = service_cwd / script_path
    script_path = script_path.resolve()
    if _option_value(args, "--port") != str(port):
        return False
    if service == "strata" and _option_value(args, "--engine") != "strata":
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
    if service not in ("studio", "strata", "comfyui"):
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

        return psutil.pid_exists(pid)
    return (Path("/proc") / str(pid)).exists()


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
    os.kill(pid, signal.SIGTERM)


def _strata_endpoint_matches(settings: Settings) -> bool:
    config = settings.section("strata")
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


def _reject_unmanageable_strata(settings: Settings, service: str) -> None:
    if service == "strata" and _strata_endpoint_matches(settings):
        raise ServiceError(
            "Strata is responding with the configured model, but no matching local process "
            "was found; it cannot be taken over safely or stopped on Ctrl+C"
        )


def _check_service_start(
    settings: Settings,
    service: str,
) -> tuple[int | None, bool, int | None]:
    command, cwd, host, port = service_command(settings, service)
    pid_file = _pid_path(settings, service)
    stale_pid_file = False
    old_pid: int | None = None
    invalid_pid_file = False

    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text(encoding="ascii").strip())
        except ValueError:
            invalid_pid_file = True
        else:
            if _service_process_matches(settings, service, old_pid):
                return old_pid, False, None
            stale_pid_file = not _pid_exists(old_pid)

    matching_pids = _find_service_processes(settings, service)
    if matching_pids:
        if len(matching_pids) != 1:
            raise ServiceError(
                f"multiple {service} processes match the configured server script and port; "
                "refusing to take over any of them"
            )
        return None, stale_pid_file, matching_pids[0]

    if invalid_pid_file:
        _reject_unmanageable_strata(settings, service)
        raise ServiceError(f"invalid pid file; inspect before removing it: {pid_file}")

    if old_pid is not None and not stale_pid_file:
        _reject_unmanageable_strata(settings, service)
        raise _pid_mismatch_error(pid_file, old_pid, service, command, cwd)

    if _port_open(host, port):
        _reject_unmanageable_strata(settings, service)
        raise ServiceError(f"refusing to start {service}: {host}:{port} is already in use")

    return None, stale_pid_file, None


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


def service_running(settings: Settings, service: str) -> bool:
    pid_file = _pid_path(settings, service)
    try:
        pid = int(pid_file.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        pid = None
    if pid is not None and _service_process_matches(settings, service, pid):
        return True
    return service == "strata" and _strata_endpoint_matches(settings)


def start_service(settings: Settings, service: str) -> tuple[int | None, Path]:
    pid, log_path, _ = start_service_tracked(settings, service)
    return pid, log_path


def start_service_tracked(settings: Settings, service: str) -> tuple[int | None, Path, bool]:
    command, cwd, _, _ = service_command(settings, service)
    pid_file = _pid_path(settings, service)
    log_path = pid_file.parent / "service.log"
    pid_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    set_private_permissions(pid_file.parent, directory=True)
    existing_pid, stale_pid_file, takeover_pid = _check_service_start(settings, service)
    if stale_pid_file:
        pid_file.unlink(missing_ok=True)
    if existing_pid is not None:
        return existing_pid, log_path, False
    if takeover_pid is not None:
        if not _service_process_matches(settings, service, takeover_pid):
            raise ServiceError(
                f"{service} process PID {takeover_pid} changed before it could be taken over"
            )
        pid_file.write_text(f"{takeover_pid}\n", encoding="ascii")
        set_private_permissions(pid_file)
        return takeover_pid, log_path, False

    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if process.poll() is not None:
        raise ServiceError(f"{service} exited during startup; inspect {log_path}")
    try:
        with pid_file.open("x", encoding="ascii") as file:
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
    if not pid_file.exists():
        if service == "strata" and _strata_endpoint_matches(settings):
            return None
        return False
    try:
        pid = int(pid_file.read_text(encoding="ascii").strip())
    except ValueError as error:
        if service == "strata" and _strata_endpoint_matches(settings):
            return None
        raise ServiceError(f"invalid pid file; leaving it untouched: {pid_file}") from error
    if not _service_process_matches(settings, service, pid):
        if not _pid_exists(pid):
            pid_file.unlink(missing_ok=True)
            if service == "strata" and _strata_endpoint_matches(settings):
                return None
            return False
        if service == "strata" and _strata_endpoint_matches(settings):
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