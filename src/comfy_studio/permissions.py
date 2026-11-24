from __future__ import annotations

import os
import subprocess
from pathlib import Path


def set_private_permissions(path: Path, *, directory: bool = False) -> None:
    if os.name != "nt":
        os.chmod(path, 0o700 if directory else 0o600)
        return

    username = os.environ.get("USERNAME")
    if not username:
        raise OSError("cannot determine the current Windows user for private permissions")
    domain = os.environ.get("USERDOMAIN")
    principal = f"{domain}\\{username}" if domain else username
    rights = "(OI)(CI)(F)" if directory else "(F)"
    try:
        result = subprocess.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"{principal}:{rights}"],
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise OSError(f"could not set private permissions for {path}: {error}") from error
    if result.returncode:
        details = result.stderr.decode(errors="replace").strip()
        if not details:
            details = result.stdout.decode(errors="replace").strip()
        if not details:
            details = f"icacls exited with status {result.returncode}"
        raise OSError(f"could not set private permissions for {path}: {details}")
