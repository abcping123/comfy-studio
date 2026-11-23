from __future__ import annotations

import sys
import unittest
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import Mock, patch

from strata_comfy_studio.system_metrics import SystemMetrics


CpuTimes = namedtuple("CpuTimes", ("user", "system", "idle", "interrupt", "dpc"))


class SystemMetricsTests(unittest.TestCase):
    def test_windows_cpu_percent_uses_psutil_cpu_times(self) -> None:
        psutil = SimpleNamespace(cpu_times=Mock(side_effect=[
            CpuTimes(1.0, 1.0, 8.0, 0.0, 0.0),
            CpuTimes(2.0, 1.0, 9.0, 0.0, 0.0),
        ]))
        metrics = SystemMetrics(0)

        with patch("strata_comfy_studio.system_metrics.platform.system", return_value="Windows"), \
                patch.dict(sys.modules, {"psutil": psutil}):
            self.assertIsNone(metrics._cpu_percent())
            self.assertEqual(metrics._cpu_percent(), 50.0)

    def test_windows_memory_percent_uses_psutil_virtual_memory(self) -> None:
        psutil = SimpleNamespace(
            virtual_memory=Mock(return_value=SimpleNamespace(percent=42.4))
        )

        with patch("strata_comfy_studio.system_metrics.platform.system", return_value="Windows"), \
                patch.dict(sys.modules, {"psutil": psutil}):
            self.assertEqual(SystemMetrics._memory_percent(), 42.4)


if __name__ == "__main__":
    unittest.main()
