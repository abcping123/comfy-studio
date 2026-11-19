from __future__ import annotations

import unittest

from strata_comfy_studio.services import ServiceError, _pid_path, service_command
from strata_comfy_studio.settings import load_config, project_root


class ServiceCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = load_config(project_root() / "config.example.json")

    def test_strata_command_uses_json_endpoint_and_private_runtime_config(self) -> None:
        command, cwd, host, port = service_command(self.settings, "strata")
        self.assertEqual((host, port), ("127.0.0.1", 8080))
        self.assertIn("--host", command)
        self.assertIn("--config", command)
        self.assertIn("temp/strata/runtime.json", " ".join(command))
        self.assertNotIn(self.settings.section("strata")["api_key"], command)
        self.assertEqual(cwd, project_root().parent.parent / "repos/Strata")

    def test_comfy_command_uses_default_model_paths(self) -> None:
        command, cwd, host, port = service_command(self.settings, "comfyui")
        self.assertEqual((host, port), ("127.0.0.1", 8188))
        self.assertIn("--listen", command)
        self.assertNotIn("--extra-model-paths-config", command)
        self.assertEqual(cwd, self.settings.path("comfyui.working_directory").resolve())

    def test_pid_files_use_service_specific_temporary_directories(self) -> None:
        self.assertEqual(
            _pid_path(self.settings, "strata"),
            project_root() / "temp" / "strata" / "service.pid",
        )
        self.assertEqual(
            _pid_path(self.settings, "comfyui"),
            project_root() / "temp" / "comfyui" / "service.pid",
        )

    def test_unknown_service_is_rejected(self) -> None:
        with self.assertRaisesRegex(ServiceError, "unknown managed service"):
            service_command(self.settings, "other")


if __name__ == "__main__":
    unittest.main()