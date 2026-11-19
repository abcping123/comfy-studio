from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit

from .settings import Settings


class ServiceError(RuntimeError):
    """A managed local service cannot safely be started or stopped."""


def service_command(settings: Settings, service: str) -> tuple[list[str], Path, str, int]:
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
        ]
        cwd = Path(config["working_directory"]).resolve()
    else:
        raise ServiceError(f"unknown managed service: {service}")
    endpoint = urlsplit(settings.section(service)["endpoint"])
    return command, cwd, endpoint.hostname or "127.0.0.1", endpoint.port or 0


def _pid_path(settings: Settings, service: str) -> Path:
    return settings.path("paths.temp_root") / service / "service.pid"


def _process_matches(pid: int, command: list[str], cwd: Path) -> bool:
    proc = Path("/proc") / str(pid)
    try:
        args = [value.decode("utf-8", errors="replace") for value in (proc / "cmdline").read_bytes().split(b"\0") if value]
        process_cwd = (proc / "cwd").resolve()
    except (OSError, PermissionError):
        return False
    return args == command and process_cwd == cwd.resolve()


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def service_running(settings: Settings, service: str) -> bool:
    command, cwd, _, _ = service_command(settings, service)
    pid_file = _pid_path(settings, service)
    try:
        pid = int(pid_file.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return False
    return _process_matches(pid, command, cwd)


def start_service(settings: Settings, service: str) -> tuple[int, Path]:
    command, cwd, host, port = service_command(settings, service)
    pid_file = _pid_path(settings, service)
    pid_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text(encoding="ascii").strip())
        except ValueError as error:
            raise ServiceError(f"invalid pid file; inspect before removing it: {pid_file}") from error
        if _process_matches(old_pid, command, cwd):
            return old_pid, pid_file.parent / "service.log"
        if (Path("/proc") / str(old_pid)).exists():
            raise ServiceError(f"pid file does not identify the expected {service} process: {pid_file}")
        pid_file.unlink()
    if _port_open(host, port):
        raise ServiceError(f"refusing to start {service}: {host}:{port} is already in use")

    log_path = pid_file.parent / "service.log"
    with log_path.open("ab") as log:
        process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    if process.poll() is not None:
        raise ServiceError(f"{service} exited during startup; inspect {log_path}")
    try:
        with pid_file.open("x", encoding="ascii") as file:
            file.write(f"{process.pid}\n")
        os.chmod(pid_file, 0o600)
    except OSError as error:
        if _process_matches(process.pid, command, cwd):
            os.kill(process.pid, signal.SIGTERM)
        raise ServiceError(f"could not record the {service} pid: {error}") from error
    return process.pid, log_path


def stop_service(settings: Settings, service: str, wait_s: float = 10.0) -> bool:
    command, cwd, _, _ = service_command(settings, service)
    pid_file = _pid_path(settings, service)
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text(encoding="ascii").strip())
    except ValueError as error:
        raise ServiceError(f"invalid pid file; leaving it untouched: {pid_file}") from error
    if not _process_matches(pid, command, cwd):
        if not Path("/proc").joinpath(str(pid)).exists():
            pid_file.unlink(missing_ok=True)
            return False
        raise ServiceError(f"pid file does not identify the expected {service} process; leaving it untouched")

    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if not _process_matches(pid, command, cwd):
            pid_file.unlink(missing_ok=True)
            return True
        time.sleep(0.1)
    raise ServiceError(f"{service} is still running after SIGTERM; no force kill was sent")