from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from comfy_studio.services import (
    ServiceError,
    _pid_exists,
    _pid_path,
    _strata_endpoint_matches,
    preflight_service_start,
    preflight_studio_start,
    service_command,
    start_service_tracked,
    _service_process_matches,
    _process_matches,
    service_running,
    start_service,
    stop_service,
)
from comfy_studio.settings import Settings, load_config, project_root


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
        self.assertEqual(command[0], str(self.settings.path("strata.python")))
        self.assertEqual(command[1], str(self.settings.path("strata.server_script").resolve()))
        self.assertIn(str(self.settings.path("strata.runtime_config")), " ".join(command))
        self.assertNotIn(self.settings.section("strata")["api_key"], command)
        self.assertEqual(
            cwd,
            self.settings.path("strata.server_script").resolve().parent.parent,
        )

    def test_bonsai_command_matches_the_reference_llama_server_options(self) -> None:
        command, cwd, host, port = service_command(self.settings, "bonsai")
        bonsai = self.settings.section("bonsai")
        self.assertEqual((host, port), ("127.0.0.1", 9931))
        self.assertEqual(command, [
            str(self.settings.path("bonsai.server_binary").resolve()),
            "--model", str(self.settings.path("bonsai.model_file").resolve()),
            "--mmproj", str(self.settings.path("bonsai.mmproj_file").resolve()),
            "--ctx-size", "65536",
            "--n-gpu-layers", "all",
            "--flash-attn", "on",
            "--parallel", "1",
            "--n-predict", "16384",
            "--temp", "1.0",
            "--top-p", "0.95",
            "--top-k", "20",
            "--min-p", "0.05",
            "--presence-penalty", "0.0",
            "--repeat-penalty", "1.0",
            "--host", "127.0.0.1",
            "--port", "9931",
        ])
        self.assertEqual(cwd, Path(command[0]).resolve().parent)

    def test_comfy_command_uses_default_model_paths(self) -> None:
        command, cwd, host, port = service_command(self.settings, "comfyui")
        self.assertEqual((host, port), ("127.0.0.1", 8188))
        self.assertEqual(command[0], str(self.settings.path("comfyui.python")))
        self.assertEqual(command[1], str(self.settings.path("comfyui.main_script").resolve()))
        self.assertIn("--listen", command)
        output_argument = command.index("--output-directory")
        self.assertEqual(command[output_argument + 1],
                 str(self.settings.path("comfyui.output_directory").resolve()))
        input_argument = command.index("--input-directory")
        self.assertEqual(command[input_argument + 1],
                 str(self.settings.path("comfyui.input_directory").resolve()))
        self.assertNotIn("--extra-model-paths-config", command)
        self.assertEqual(cwd, self.settings.path("comfyui.working_directory").resolve())

    def test_studio_is_not_a_pid_managed_service(self) -> None:
        with self.assertRaisesRegex(ServiceError, "unknown managed service"):
            service_command(self.settings, "studio")

    def test_pid_files_are_flat_in_runtime_directory(self) -> None:
        self.assertEqual(
            _pid_path(self.settings, "strata"),
            project_root() / "temp" / "runtime" / "service_strata.pid",
        )
        self.assertEqual(
            _pid_path(self.settings, "comfyui"),
            project_root() / "temp" / "runtime" / "service_comfyui.pid",
        )

    def test_unknown_service_is_rejected(self) -> None:
        with self.assertRaisesRegex(ServiceError, "unknown managed service"):
            service_command(self.settings, "other")

    def test_starting_a_missing_service_uses_an_isolated_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(Path(directory), self.settings.data)
            process = MagicMock()
            process.poll.return_value = None
            process.pid = 404
            with (
                patch("comfy_studio.services._port_open", return_value=False),
                patch("comfy_studio.services._find_service_processes", return_value=[]),
                patch("comfy_studio.services.set_private_permissions"),
                patch("comfy_studio.services.subprocess.Popen", return_value=process) as popen,
            ):
                pid, _, started_here = start_service_tracked(settings, "bonsai")

        self.assertEqual(pid, 404)
        self.assertTrue(started_here)
        if os.name == "nt":
            self.assertIn("creationflags", popen.call_args.kwargs)
        else:
            self.assertTrue(popen.call_args.kwargs["start_new_session"])


class ExternalStrataTests(unittest.TestCase):
    def _settings(self, root: Path) -> Settings:
        data = dict(load_config(project_root() / "data" / "config.example.json").data)
        data["chat_backends"]["strata"] = dict(data["chat_backends"]["strata"])
        data["chat_backends"]["strata"]["api_key"] = "test-api-key"
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

            with patch("comfy_studio.services.build_opener", return_value=opener):
                self.assertTrue(_strata_endpoint_matches(settings))
                request = opener.open.call_args.args[0]
                self.assertEqual(request.get_header("Authorization"), "Bearer test-api-key")

                response.read.return_value = json.dumps({"data": [{"id": "different-model"}]}).encode("utf-8")
                self.assertFalse(_strata_endpoint_matches(settings))

    def test_start_rejects_an_unidentifiable_external_strata_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(Path(directory))
            with (
                patch("comfy_studio.services._port_open", return_value=True),
                patch("comfy_studio.services._strata_endpoint_matches", return_value=True),
                patch("comfy_studio.services._find_service_processes", return_value=[]),
                patch("comfy_studio.services.set_private_permissions"),
                patch("comfy_studio.services.subprocess.Popen") as popen,
                self.assertRaisesRegex(ServiceError, "cannot be taken over safely"),
            ):
                start_service_tracked(settings, "strata")

            self.assertFalse(_pid_path(settings, "strata").exists())
            popen.assert_not_called()

    def test_service_process_matching_uses_script_and_port(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(Path(directory))
            for service in ("comfyui", "strata", "bonsai"):
                command, cwd, host, port = service_command(settings, service)
                if service == "comfyui":
                    args = [command[0], command[1], "--listen", host, "--port", str(port)]
                else:
                    args = command
                with patch(
                    "comfy_studio.services._process_identity",
                    return_value=(args, cwd),
                ):
                    self.assertTrue(_service_process_matches(settings, service, 1234))

            bonsai_command, bonsai_cwd, _, _ = service_command(settings, "bonsai")
            tuned_bonsai_args = bonsai_command.copy()
            tuned_bonsai_args[tuned_bonsai_args.index("--ctx-size") + 1] = "4096"
            with patch(
                "comfy_studio.services._process_identity",
                return_value=(tuned_bonsai_args, bonsai_cwd),
            ):
                self.assertTrue(_service_process_matches(settings, "bonsai", 1234))

            wrong_model_args = bonsai_command.copy()
            wrong_model_args[wrong_model_args.index("--model") + 1] = str(
                Path(directory) / "different-model.gguf"
            )
            with patch(
                "comfy_studio.services._process_identity",
                return_value=(wrong_model_args, bonsai_cwd),
            ):
                self.assertFalse(_service_process_matches(settings, "bonsai", 1234))

            strata_command, _, _, strata_port = service_command(settings, "strata")
            wrong_port_args = strata_command.copy()
            wrong_port_args[wrong_port_args.index("--port") + 1] = str(strata_port + 1)
            with patch(
                "comfy_studio.services._process_identity",
                return_value=(wrong_port_args, cwd),
            ):
                self.assertFalse(_service_process_matches(settings, "strata", 1234))

    def test_start_restarts_untracked_matching_backend_and_comfyui_processes(self) -> None:
        for service in ("strata", "bonsai", "comfyui"):
            with self.subTest(service=service), tempfile.TemporaryDirectory() as directory:
                settings = self._settings(Path(directory))
                pid_file = _pid_path(settings, service)
                pid_file.parent.mkdir(parents=True)
                pid_file.write_text("202\n", encoding="ascii")
                process = MagicMock()
                process.poll.return_value = None
                process.pid = 404

                with (
                    patch("comfy_studio.services._find_service_processes", return_value=[303]),
                    patch(
                        "comfy_studio.services._service_process_matches",
                        side_effect=[True, False],
                    ),
                    patch("comfy_studio.services._terminate_process") as terminate,
                    patch("comfy_studio.services._port_open", return_value=False),
                    patch("comfy_studio.services.set_private_permissions"),
                    patch(
                        "comfy_studio.services.subprocess.Popen",
                        return_value=process,
                    ) as popen,
                ):
                    started_pid, _, started_here = start_service_tracked(settings, service)

                self.assertEqual(started_pid, 404)
                self.assertTrue(started_here)
                terminate.assert_called_once_with(303)
                popen.assert_called_once()
                self.assertEqual(pid_file.read_text(encoding="ascii"), "404\n")

    def test_start_restarts_all_duplicate_service_processes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(Path(directory))
            process = MagicMock()
            process.poll.return_value = None
            process.pid = 505

            with (
                patch("comfy_studio.services._find_service_processes", return_value=[303, 304]),
                patch(
                    "comfy_studio.services._service_process_matches",
                    side_effect=[True, True, False, False],
                ),
                patch("comfy_studio.services._terminate_process") as terminate,
                patch("comfy_studio.services._port_open", return_value=False),
                patch("comfy_studio.services.set_private_permissions"),
                patch("comfy_studio.services.subprocess.Popen", return_value=process),
            ):
                pid, _, started_here = start_service_tracked(settings, "strata")

        self.assertEqual(pid, 505)
        self.assertTrue(started_here)
        self.assertEqual(
            [call_args.args for call_args in terminate.call_args_list],
            [(303,), (304,)],
        )

    def test_start_reuses_a_unique_tracked_process_when_it_is_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(Path(directory))
            pid_file = _pid_path(settings, "strata")
            pid_file.parent.mkdir(parents=True)
            pid_file.write_text("202\n", encoding="ascii")

            with (
                patch("comfy_studio.services._find_service_processes", return_value=[202]),
                patch("comfy_studio.services._process_isolated_from_studio", return_value=True),
                patch("comfy_studio.services.subprocess.Popen") as popen,
            ):
                pid, _, started_here = start_service_tracked(settings, "strata")

        self.assertEqual(pid, 202)
        self.assertFalse(started_here)
        popen.assert_not_called()

    def test_start_restarts_a_tracked_process_sharing_the_studio_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(Path(directory))
            pid_file = _pid_path(settings, "strata")
            pid_file.parent.mkdir(parents=True)
            pid_file.write_text("202\n", encoding="ascii")
            process = MagicMock()
            process.poll.return_value = None
            process.pid = 404

            with (
                patch("comfy_studio.services._find_service_processes", return_value=[202]),
                patch("comfy_studio.services._process_isolated_from_studio", return_value=False),
                patch(
                    "comfy_studio.services._service_process_matches",
                    side_effect=[True, False],
                ),
                patch("comfy_studio.services._terminate_process") as terminate,
                patch("comfy_studio.services._port_open", return_value=False),
                patch("comfy_studio.services.set_private_permissions"),
                patch("comfy_studio.services.subprocess.Popen", return_value=process),
            ):
                pid, _, started_here = start_service_tracked(settings, "strata")

        self.assertEqual(pid, 404)
        self.assertTrue(started_here)
        terminate.assert_called_once_with(202)

    def test_stop_stops_a_matching_service_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(Path(directory))
            pid_file = _pid_path(settings, "comfyui")
            pid_file.parent.mkdir(parents=True)
            pid_file.write_text("303\n", encoding="ascii")

            with (
                patch("comfy_studio.services._find_service_processes", return_value=[303]),
                patch(
                    "comfy_studio.services._service_process_matches",
                    side_effect=[True, False],
                ),
                patch("comfy_studio.services._terminate_process") as terminate,
                patch("comfy_studio.services.time.monotonic", side_effect=[0, 0]),
            ):
                self.assertTrue(stop_service(settings, "comfyui"))

            terminate.assert_called_once_with(303)
            self.assertFalse(pid_file.exists())

    def test_start_still_rejects_an_unrelated_listener(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(Path(directory))
            with (
                patch("comfy_studio.services._port_open", return_value=True),
                patch("comfy_studio.services._strata_endpoint_matches", return_value=False),
                patch("comfy_studio.services._find_service_processes", return_value=[]),
                self.assertRaisesRegex(ServiceError, "already in use"),
            ):
                start_service(settings, "strata")

    def test_preflight_allows_a_pid_mismatch_when_the_service_port_is_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(Path(directory))
            pid_file = _pid_path(settings, "comfyui")
            pid_file.parent.mkdir(parents=True, exist_ok=True)
            pid_file.write_text(f"{os.getpid()}\n", encoding="ascii")
            with (
                patch("comfy_studio.services._find_service_processes", return_value=[]),
                patch("comfy_studio.services._port_open", return_value=False),
                patch("comfy_studio.services._terminate_process") as terminate,
            ):
                preflight_service_start(settings, ("comfyui",))

            self.assertTrue(pid_file.exists())
            terminate.assert_not_called()

    def test_start_replaces_a_mismatched_pid_without_signaling_that_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(Path(directory))
            pid_file = _pid_path(settings, "comfyui")
            pid_file.parent.mkdir(parents=True, exist_ok=True)
            pid_file.write_text(f"{os.getpid()}\n", encoding="ascii")
            process = MagicMock()
            process.poll.return_value = None
            process.pid = 505

            with (
                patch("comfy_studio.services._find_service_processes", return_value=[]),
                patch("comfy_studio.services._port_open", return_value=False),
                patch("comfy_studio.services._terminate_process") as terminate,
                patch("comfy_studio.services.set_private_permissions"),
                patch(
                    "comfy_studio.services.subprocess.Popen",
                    return_value=process,
                ) as popen,
            ):
                pid, _, started_here = start_service_tracked(settings, "comfyui")

            self.assertEqual(pid, 505)
            self.assertTrue(started_here)
            self.assertEqual(pid_file.read_text(encoding="ascii"), "505\n")
            terminate.assert_not_called()
            popen.assert_called_once()

    @unittest.skipUnless(
        hasattr(os, "fork") and hasattr(os, "waitid"),
        "POSIX zombie process inspection only",
    )
    def test_zombie_pid_is_not_treated_as_a_live_process(self) -> None:
        child_pid = os.fork()
        if child_pid == 0:
            os._exit(0)
        try:
            os.waitid(os.P_PID, child_pid, os.WEXITED | os.WNOWAIT)
            self.assertFalse(_pid_exists(child_pid))
        finally:
            os.waitpid(child_pid, 0)

    def test_studio_preflight_checks_its_port_without_tracking_a_pid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(Path(directory))
            app = settings.section("app")
            with (
                patch("comfy_studio.services._port_open", return_value=True) as port_open,
                self.assertRaisesRegex(ServiceError, "studio: .* is already in use"),
            ):
                preflight_studio_start(settings)

            port_open.assert_called_once_with(app["listen_host"], app["port"])

    def test_status_and_stop_report_but_do_not_manage_external_strata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(Path(directory))
            with patch("comfy_studio.services._strata_endpoint_matches", return_value=True):
                self.assertTrue(service_running(settings, "strata"))
                self.assertIsNone(stop_service(settings, "strata"))

    @unittest.skipUnless(os.name == "nt", "Windows process inspection only")
    def test_windows_process_matching_checks_command_and_working_directory(self) -> None:
        import psutil

        process = psutil.Process()
        command = process.cmdline()
        cwd = Path(process.cwd())
        self.assertTrue(_process_matches(process.pid, command, cwd))
        self.assertFalse(_process_matches(process.pid, command + ["unexpected"], cwd))
        self.assertFalse(_process_matches(process.pid, command, cwd / "unexpected"))


if __name__ == "__main__":
    unittest.main()