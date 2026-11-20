from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import call, patch

from strata_comfy_studio.cli import _load_active_config, main
from strata_comfy_studio.services import ServiceError


class CliTests(unittest.TestCase):
    def test_start_starts_all_services_comfyui_first(self) -> None:
        settings = object()
        with (
            patch("strata_comfy_studio.cli._load_active_config", return_value=(Path("config.json"), settings)) as load_active_config,
            patch("strata_comfy_studio.cli.validate_artifacts") as validate_artifacts,
            patch("strata_comfy_studio.cli.prepare_runtime_files") as prepare_runtime_files,
            patch("strata_comfy_studio.cli.start_service", side_effect=[
                (101, Path("comfyui.log")),
                (202, Path("strata.log")),
                (303, Path("studio.log")),
            ]) as start_service,
            redirect_stdout(io.StringIO()),
        ):
            result = main(["start"])

        self.assertEqual(result, 0)
        load_active_config.assert_called_once_with(initialize_if_missing=True)
        validate_artifacts.assert_called_once_with(settings)
        prepare_runtime_files.assert_called_once_with(settings)
        self.assertEqual(start_service.call_args_list, [
            call(settings, "comfyui"),
            call(settings, "strata"),
            call(settings, "studio"),
        ])

    def test_start_initializes_only_when_config_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            settings = object()
            strata_repo = Path(directory) / "Strata"
            comfyui_repo = Path(directory) / "ComfyUI"
            with (
                patch("strata_comfy_studio.cli.default_config_path", return_value=config_path),
                patch("strata_comfy_studio.cli.initialize_config") as initialize_config,
                patch("strata_comfy_studio.cli.load_config", return_value=settings),
                patch("strata_comfy_studio.cli._check_private_config"),
                patch("builtins.input", side_effect=[str(strata_repo), str(comfyui_repo)]) as prompt,
                redirect_stdout(io.StringIO()),
            ):
                loaded_path, loaded_settings = _load_active_config(initialize_if_missing=True)

            self.assertEqual((loaded_path, loaded_settings), (config_path, settings))
            initialize_config.assert_called_once_with(
                output_path=config_path,
                strata_repo_path=strata_repo,
                comfyui_repo_path=comfyui_repo,
            )
            self.assertEqual(prompt.call_count, 2)

    def test_start_does_not_prompt_or_overwrite_existing_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.touch()
            with (
                patch("strata_comfy_studio.cli.default_config_path", return_value=config_path),
                patch("strata_comfy_studio.cli.initialize_config") as initialize_config,
                patch("strata_comfy_studio.cli.load_config", return_value=object()),
                patch("strata_comfy_studio.cli._check_private_config"),
                patch("builtins.input") as prompt,
            ):
                _load_active_config(initialize_if_missing=True)

            initialize_config.assert_not_called()
            prompt.assert_not_called()

    def test_start_continues_after_a_service_fails(self) -> None:
        settings = object()
        with (
            patch("strata_comfy_studio.cli._load_active_config", return_value=(Path("config.json"), settings)) as load_active_config,
            patch("strata_comfy_studio.cli.validate_artifacts"),
            patch("strata_comfy_studio.cli.prepare_runtime_files"),
            patch("strata_comfy_studio.cli.start_service", side_effect=[
                ServiceError("ComfyUI failed"),
                (202, Path("strata.log")),
                (303, Path("studio.log")),
            ]) as start_service,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            result = main(["start"])

        self.assertEqual(result, 2)
        self.assertEqual(start_service.call_count, 3)

    def test_start_reuses_external_strata_without_failing(self) -> None:
        settings = object()
        output = io.StringIO()
        with (
            patch("strata_comfy_studio.cli._load_active_config", return_value=(Path("config.json"), settings)),
            patch("strata_comfy_studio.cli.validate_artifacts"),
            patch("strata_comfy_studio.cli.prepare_runtime_files"),
            patch("strata_comfy_studio.cli.start_service", side_effect=[
                (101, Path("comfyui.log")),
                (None, Path("strata.log")),
                (303, Path("studio.log")),
            ]),
            redirect_stdout(output),
        ):
            result = main(["start"])

        self.assertEqual(result, 0)
        self.assertIn("Using existing strata service", output.getvalue())

    def test_status_checks_all_services(self) -> None:
        settings = object()
        with (
            patch("strata_comfy_studio.cli._load_active_config", return_value=(Path("config.json"), settings)) as load_active_config,
            patch("strata_comfy_studio.cli.service_running", side_effect=[True, False, True]) as service_running,
            redirect_stdout(io.StringIO()),
        ):
            result = main(["status"])

        self.assertEqual(result, 0)
        load_active_config.assert_called_once_with(initialize_if_missing=False)
        self.assertEqual(service_running.call_args_list, [
            call(settings, "comfyui"),
            call(settings, "strata"),
            call(settings, "studio"),
        ])

    def test_stop_stops_all_services_in_reverse_order(self) -> None:
        settings = object()
        with (
            patch("strata_comfy_studio.cli._load_active_config", return_value=(Path("config.json"), settings)) as load_active_config,
            patch("strata_comfy_studio.cli.stop_service", side_effect=[True, False, True]) as stop_service,
            redirect_stdout(io.StringIO()),
        ):
            result = main(["stop"])

        self.assertEqual(result, 0)
        load_active_config.assert_called_once_with(initialize_if_missing=False)
        self.assertEqual(stop_service.call_args_list, [
            call(settings, "studio"),
            call(settings, "strata"),
            call(settings, "comfyui"),
        ])

    def test_stop_leaves_external_strata_running(self) -> None:
        settings = object()
        output = io.StringIO()
        with (
            patch("strata_comfy_studio.cli._load_active_config", return_value=(Path("config.json"), settings)),
            patch("strata_comfy_studio.cli.stop_service", side_effect=[True, None, True]),
            redirect_stdout(output),
        ):
            result = main(["stop"])

        self.assertEqual(result, 0)
        self.assertIn("strata is running outside Studio; left untouched", output.getvalue())

    def test_service_commands_reject_arguments(self) -> None:
        for command in ("start", "status", "stop"):
            with self.subTest(command=command), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main([command, "comfyui"])
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()