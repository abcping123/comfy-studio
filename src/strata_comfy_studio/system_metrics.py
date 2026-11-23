from __future__ import annotations

import csv
import math
import platform
import shutil
import subprocess
import threading
from pathlib import Path


class SystemMetrics:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self._nvidia_smi = shutil.which("nvidia-smi")
        self._cpu_lock = threading.Lock()
        self._previous_cpu: tuple[int, int] | None = None

    def snapshot(self) -> dict[str, float | None]:
        gpu_percent, vram_percent = self._gpu_metrics()
        return {
            "cpu_percent": self._cpu_percent(),
            "memory_percent": self._memory_percent(),
            "gpu_percent": gpu_percent,
            "vram_percent": vram_percent,
        }

    def _cpu_percent(self) -> float | None:
        current = self._read_cpu_times()
        with self._cpu_lock:
            previous = self._previous_cpu
            self._previous_cpu = current
        if current is None or previous is None:
            return None
        total_delta = current[0] - previous[0]
        idle_delta = current[1] - previous[1]
        if total_delta <= 0:
            return None
        return self._clamp_percent((total_delta - idle_delta) / total_delta * 100)

    @staticmethod
    def _read_cpu_times() -> tuple[int, int] | None:
        if platform.system() == "Windows":
            import psutil

            cpu_times = psutil.cpu_times()
            return round(sum(cpu_times) * 1000), round(cpu_times.idle * 1000)
        try:
            fields = Path("/proc/stat").read_text(encoding="ascii").splitlines()[0].split()
            values = [int(value) for value in fields[1:]]
        except (OSError, UnicodeError, ValueError, IndexError):
            return None
        if not fields or fields[0] != "cpu" or len(values) < 4:
            return None
        total = sum(values[:8])
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return total, idle

    @staticmethod
    def _memory_percent() -> float | None:
        if platform.system() == "Windows":
            import psutil

            return SystemMetrics._clamp_percent(psutil.virtual_memory().percent)
        values: dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                name, separator, value = line.partition(":")
                if separator and name in {"MemTotal", "MemAvailable"}:
                    values[name] = int(value.split()[0])
        except (OSError, UnicodeError, ValueError, IndexError):
            return None
        total = values.get("MemTotal", 0)
        available = values.get("MemAvailable")
        if total <= 0 or available is None:
            return None
        return SystemMetrics._clamp_percent((total - available) / total * 100)

    def _gpu_metrics(self) -> tuple[float | None, float | None]:
        if self._nvidia_smi is None:
            return None, None
        try:
            result = subprocess.run(
                [self._nvidia_smi, "--id", str(self.gpu_id),
                 "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None, None
        if result.returncode != 0:
            return None, None
        rows = list(csv.reader(line for line in result.stdout.splitlines() if line.strip()))
        if not rows or len(rows[0]) < 3:
            return None, None
        utilization = self._parse_number(rows[0][0])
        memory_used = self._parse_number(rows[0][1])
        memory_total = self._parse_number(rows[0][2])
        vram_percent = None
        if memory_used is not None and memory_total is not None and memory_total > 0:
            vram_percent = self._clamp_percent(memory_used / memory_total * 100)
        return (self._clamp_percent(utilization) if utilization is not None else None,
                vram_percent)

    @staticmethod
    def _parse_number(value: str) -> float | None:
        try:
            number = float(value.strip())
        except (AttributeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _clamp_percent(value: float) -> float:
        return round(min(100.0, max(0.0, value)), 1)