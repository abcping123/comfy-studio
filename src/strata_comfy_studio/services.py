from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from http.client import HTTPException
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from .settings import Settings, project_root


class ServiceError(RuntimeError):
    """A managed local service cannot safely be started or stopped."""


def service_command(settings: Settings, service: str) -> tuple[list[str], Path, str, int]:
    if service == "studio":
        config = settings.section("app")
        return ([sys.executable, "-m", "strata_comfy_studio.app"], project_root(),
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


def service_running(settings: Settings, service: str) -> bool:
    command, cwd, _, _ = service_command(settings, service)
    pid_file = _pid_path(settings, service)
    try:
        pid = int(pid_file.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        pid = None
    if pid is not None and _process_matches(pid, command, cwd):
        return True
    return service == "strata" and _strata_endpoint_matches(settings)


def start_service(settings: Settings, service: str) -> tuple[int | None, Path]:
    command, cwd, host, port = service_command(settings, service)
    pid_file = _pid_path(settings, service)
    log_path = pid_file.parent / "service.log"
    pid_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text(encoding="ascii").strip())
        except ValueError as error:
            if service == "strata" and _strata_endpoint_matches(settings):
                return None, log_path
            raise ServiceError(f"invalid pid file; inspect before removing it: {pid_file}") from error
        if _process_matches(old_pid, command, cwd):
            return old_pid, log_path
        if (Path("/proc") / str(old_pid)).exists():
            if service == "strata" and _strata_endpoint_matches(settings):
                return None, log_path
            raise ServiceError(f"pid file does not identify the expected {service} process: {pid_file}")
        pid_file.unlink()
    if _port_open(host, port):
        if service == "strata" and _strata_endpoint_matches(settings):
            return None, log_path
        raise ServiceError(f"refusing to start {service}: {host}:{port} is already in use")

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
    if not _process_matches(pid, command, cwd):
        if not Path("/proc").joinpath(str(pid)).exists():
            pid_file.unlink(missing_ok=True)
            if service == "strata" and _strata_endpoint_matches(settings):
                return None
            return False
        if service == "strata" and _strata_endpoint_matches(settings):
            return None
        raise ServiceError(f"pid file does not identify the expected {service} process; leaving it untouched")

    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if not _process_matches(pid, command, cwd):
            pid_file.unlink(missing_ok=True)
            return True
        time.sleep(0.1)
    raise ServiceError(f"{service} is still running after SIGTERM; no force kill was sent")