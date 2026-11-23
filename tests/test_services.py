from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from strata_comfy_studio.services import (
    ServiceError,
    _pid_path,
    _strata_endpoint_matches,
    preflight_service_start,
    service_command,
    service_running,
    start_service,
    stop_service,
)
from strata_comfy_studio.settings import Settings, load_config, project_root


class ServiceCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = project_root()
        cls.settings = Settings(root, load_config(root / "data" / "config.example.json").data)

    def test_strata_command_uses_json_endpoint_and_private_runtime_config(self) -> None:
        command, cwd, host, port = service_command(self.settings, "strata")
        self.assertEqual((host, port), ("127.0.0.1", 8080))
        self.assertIn("--host", command)
        self.assertIn("--config", command)
        self.assertIn("temp/strata/runtime.json", " ".join(command))
        self.assertNotIn(self.settings.section("strata")["api_key"], command)
        self.assertEqual(cwd, project_root().parent.parent / "repos/Strata/terminal")

    def test_comfy_command_uses_default_model_paths(self) -> None:
        command, cwd, host, port = service_command(self.settings, "comfyui")
        self.assertEqual((host, port), ("127.0.0.1", 8188))
        self.assertIn("--listen", command)
        output_argument = command.index("--output-directory")
        self.assertEqual(command[output_argument + 1],
                 str(self.settings.path("comfyui.output_directory").resolve()))
        input_argument = command.index("--input-directory")
        self.assertEqual(command[input_argument + 1],
                 str(self.settings.path("comfyui.input_directory").resolve()))
        self.assertNotIn("--extra-model-paths-config", command)
        self.assertEqual(cwd, self.settings.path("comfyui.working_directory").resolve())

    def test_studio_command_uses_configured_app_listener(self) -> None:
        command, cwd, host, port = service_command(self.settings, "studio")
        self.assertEqual(command, [sys.executable, "-m", "strata_comfy_studio.app"])
        self.assertEqual(cwd, project_root())
        self.assertEqual((host, port), ("127.0.0.1", 8765))

    def test_pid_files_follow_service_directory_layout(self) -> None:
        self.assertEqual(
            _pid_path(self.settings, "strata"),
            project_root() / "temp" / "strata" / "service.pid",
        )
        self.assertEqual(
            _pid_path(self.settings, "comfyui"),
            project_root() / "temp" / "comfyui" / "service.pid",
        )
        self.assertEqual(
            _pid_path(self.settings, "studio"),
            project_root() / "temp" / "service.pid",
        )

    def test_unknown_service_is_rejected(self) -> None:
        with self.assertRaisesRegex(ServiceError, "unknown managed service"):
            service_command(self.settings, "other")


class ExternalStrataTests(unittest.TestCase):
    def _settings(self, root: Path) -> Settings:
        data = dict(load_config(project_root() / "data" / "config.example.json").data)
        data["strata"] = dict(data["strata"])
        data["strata"]["api_key"] = "test-api-key"
        return Settings(root, data)

    def test_strata_probe_requires_the_configured_model_and_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(Path(directory))
            response = MagicMock()
            response.__enter__.return_value = response
            response.read.return_value = json.dumps({
                "data": [{"id": settings.section("strata")["model"]}],
            }).encode("utf-8")
            opener = MagicMock()
            opener.open.return_value = response

            with patch("strata_comfy_studio.services.build_opener", return_value=opener):
                self.assertTrue(_strata_endpoint_matches(settings))
                request = opener.open.call_args.args[0]
                self.assertEqual(request.get_header("Authorization"), "Bearer test-api-key")

                response.read.return_value = json.dumps({"data": [{"id": "different-model"}]}).encode("utf-8")
                self.assertFalse(_strata_endpoint_matches(settings))

    def test_start_reuses_a_matching_external_strata_without_claiming_its_pid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(Path(directory))
            with (
                patch("strata_comfy_studio.services._port_open", return_value=True),
                patch("strata_comfy_studio.services._strata_endpoint_matches", return_value=True),
                patch("strata_comfy_studio.services.subprocess.Popen") as popen,
            ):
                pid, _ = start_service(settings, "strata")

            self.assertIsNone(pid)
            self.assertFalse(_pid_path(settings, "strata").exists())
            popen.assert_not_called()

    def test_start_still_rejects_an_unrelated_listener(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(Path(directory))
            with (
                patch("strata_comfy_studio.services._port_open", return_value=True),
                patch("strata_comfy_studio.services._strata_endpoint_matches", return_value=False),
                self.assertRaisesRegex(ServiceError, "already in use"),
            ):
                start_service(settings, "strata")

    def test_preflight_reports_live_pid_mismatch_without_echoing_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(Path(directory))
            pid_files = [_pid_path(settings, service) for service in ("comfyui", "studio")]
            for pid_file in pid_files:
                pid_file.parent.mkdir(parents=True, exist_ok=True)
                pid_file.write_text(f"{os.getpid()}\n", encoding="ascii")
            secret_argument = "do-not-print-this-token"
            with (
                patch("strata_comfy_studio.services._process_matches", return_value=False),
                patch(
                    "strata_comfy_studio.services._process_identity",
                    return_value=(["/usr/bin/python", secret_argument], Path("/tmp/other-service")),
                ),
                self.assertRaises(ServiceError) as error,
            ):
                preflight_service_start(settings, ("comfyui", "studio"))

            message = str(error.exception)
            self.assertIn("startup preflight failed; no services were started", message)
            self.assertIn("comfyui:", message)
            self.assertIn("studio:", message)
            self.assertIn(f"PID {os.getpid()}", message)
            self.assertIn("arguments differ at index", message)
            self.assertIn("actual cwd=/tmp/other-service", message)
            self.assertIn("expected cwd=", message)
            self.assertNotIn(secret_argument, message)
            self.assertTrue(all(pid_file.exists() for pid_file in pid_files))

    def test_status_and_stop_report_but_do_not_manage_external_strata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(Path(directory))
            with patch("strata_comfy_studio.services._strata_endpoint_matches", return_value=True):
                self.assertTrue(service_running(settings, "strata"))
                self.assertIsNone(stop_service(settings, "strata"))


if __name__ == "__main__":
    unittest.main()