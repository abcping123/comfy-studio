from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import call, patch

from comfy_studio.cli import _load_active_config, main, validate_artifacts
from comfy_studio.modes import IMAGE_MODES
from comfy_studio.settings import load_config, project_root
from comfy_studio.services import ServiceError


class CliTests(unittest.TestCase):
    def test_artifact_preflight_checks_models_for_each_mode(self) -> None:
        settings = load_config(project_root() / "data" / "config.example.json")
        settings.data["strata"]["api_key"] = "test-api-key"
        settings.data["app"]["access_token"] = "test-token"
        checked_files: list[Path] = []
        checked_directories: list[Path] = []

        def check_file(value: str, _label: str) -> Path:
            path = Path(value)
            checked_files.append(path)
            return path

        def check_directory(value: str, _label: str) -> Path:
            path = Path(value)
            checked_directories.append(path)
            return path

        with (
            patch("comfy_studio.cli._require_file", side_effect=check_file),
            patch("comfy_studio.cli._require_directory", side_effect=check_directory),
            patch("comfy_studio.cli.build_workflow"),
            patch.object(Path, "read_text", return_value="system prompt"),
        ):
            validate_artifacts(settings)

        expected_models = set()
        expected_directories = set()
        for mode in IMAGE_MODES:
            for model in settings.data["generation_modes"][mode.id]["models"].values():
                directory = settings.path(
                    f"comfyui.model_directories.{model['directory']}"
                )
                expected_directories.add(directory)
                expected_models.add(directory / model["file"])
        self.assertTrue(expected_directories.issubset(set(checked_directories)))
        self.assertTrue(expected_models.issubset(set(checked_files)))

    def test_parameterless_main_blocks_and_stops_started_services_on_interrupt(self) -> None:
        settings = object()
        lifecycle: list[str] = []
        with (
            patch("comfy_studio.cli._load_active_config", return_value=(Path("config.json"), settings)) as load_active_config,
            patch("comfy_studio.cli.validate_artifacts",
                  side_effect=lambda _: lifecycle.append("validate")) as validate_artifacts,
            patch("comfy_studio.cli.prepare_runtime_files",
                  side_effect=lambda _: lifecycle.append("prepare")) as prepare_runtime_files,
            patch("comfy_studio.cli.preflight_service_start",
                  side_effect=lambda *_: lifecycle.append("preflight")) as preflight_service_start,
            patch("comfy_studio.cli.service_running", side_effect=[True, True]),
            patch("comfy_studio.cli.start_service_tracked", side_effect=[
                (101, Path("comfyui.log"), True),
                (202, Path("strata.log"), True),
                (303, Path("studio.log"), True),
            ]) as start_service,
            patch("comfy_studio.cli.stop_service", side_effect=[True, True, True]) as stop_service,
            patch("comfy_studio.cli.time.sleep", side_effect=KeyboardInterrupt),
            patch("sys.argv", ["main.py"]),
            redirect_stdout(io.StringIO()),
        ):
            result = main()

        self.assertEqual(result, 130)
        load_active_config.assert_called_once_with(initialize_if_missing=True)
        validate_artifacts.assert_called_once_with(settings)
        prepare_runtime_files.assert_called_once_with(settings)
        preflight_service_start.assert_called_once_with(settings, ("comfyui", "strata", "studio"))
        self.assertEqual(lifecycle, ["prepare", "validate", "preflight"])
        self.assertEqual(start_service.call_args_list, [
            call(settings, "comfyui"),
            call(settings, "strata"),
            call(settings, "studio"),
        ])
        self.assertEqual(stop_service.call_args_list, [
            call(settings, "studio"),
            call(settings, "strata"),
            call(settings, "comfyui"),
        ])

    def test_start_initializes_only_when_config_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            settings = object()
            strata_repo = Path(directory) / "Strata"
            comfyui_repo = Path(directory) / "ComfyUI"
            with (
                patch("comfy_studio.cli.default_config_path", return_value=config_path),
                patch("comfy_studio.cli.initialize_config") as initialize_config,
                patch("comfy_studio.cli.load_config", return_value=settings),
                patch("comfy_studio.cli._check_private_config"),
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
                patch("comfy_studio.cli.default_config_path", return_value=config_path),
                patch("comfy_studio.cli.initialize_config") as initialize_config,
                patch("comfy_studio.cli.load_config", return_value=object()),
                patch("comfy_studio.cli._check_private_config"),
                patch("builtins.input") as prompt,
            ):
                _load_active_config(initialize_if_missing=True)

            initialize_config.assert_not_called()
            prompt.assert_not_called()

    def test_start_rolls_back_services_when_a_later_service_fails(self) -> None:
        settings = object()
        with (
            patch("comfy_studio.cli._load_active_config", return_value=(Path("config.json"), settings)) as load_active_config,
            patch("comfy_studio.cli.validate_artifacts"),
            patch("comfy_studio.cli.prepare_runtime_files"),
            patch("comfy_studio.cli.preflight_service_start"),
            patch("comfy_studio.cli.service_running", return_value=False),
            patch("comfy_studio.cli.start_service_tracked", side_effect=[
                (101, Path("comfyui.log"), True),
                ServiceError("Strata failed"),
            ]) as start_service,
            patch("comfy_studio.cli.stop_service", return_value=True) as stop_service,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            result = main([])

        self.assertEqual(result, 2)
        self.assertEqual(start_service.call_count, 2)
        stop_service.assert_called_once_with(settings, "comfyui")

    def test_start_preflight_failure_starts_no_services(self) -> None:
        settings = object()
        error_output = io.StringIO()
        with (
            patch("comfy_studio.cli._load_active_config", return_value=(Path("config.json"), settings)),
            patch("comfy_studio.cli.validate_artifacts"),
            patch("comfy_studio.cli.prepare_runtime_files"),
            patch("comfy_studio.cli.preflight_service_start",
                  side_effect=ServiceError("stale service PID")),
            patch("comfy_studio.cli.service_running", return_value=False),
            patch("comfy_studio.cli.start_service_tracked") as start_service,
            redirect_stderr(error_output),
        ):
            result = main([])

        self.assertEqual(result, 2)
        self.assertIn("stale service PID", error_output.getvalue())
        start_service.assert_not_called()

    def test_existing_services_are_taken_over_and_stopped_on_interrupt(self) -> None:
        settings = object()
        output = io.StringIO()
        with (
            patch("comfy_studio.cli._load_active_config", return_value=(Path("config.json"), settings)),
            patch("comfy_studio.cli.validate_artifacts"),
            patch("comfy_studio.cli.prepare_runtime_files"),
            patch("comfy_studio.cli.preflight_service_start"),
            patch("comfy_studio.cli.service_running", side_effect=[True, True]),
            patch("comfy_studio.cli.start_service_tracked", side_effect=[
                (101, Path("comfyui.log"), False),
                (202, Path("strata.log"), False),
                (303, Path("studio.log"), False),
            ]),
            patch("comfy_studio.cli.stop_service", side_effect=[True, True, True]) as stop_service,
            patch("comfy_studio.cli.time.sleep", side_effect=KeyboardInterrupt),
            redirect_stdout(output),
        ):
            result = main([])

        self.assertEqual(result, 130)
        self.assertIn("Took over existing strata with PID 202", output.getvalue())
        self.assertIn("Took over existing studio with PID 303", output.getvalue())
        self.assertEqual(stop_service.call_args_list, [
            call(settings, "studio"),
            call(settings, "strata"),
            call(settings, "comfyui"),
        ])

    def test_all_command_arguments_are_rejected(self) -> None:
        for arguments in (["start"], ["status"], ["stop"], ["anything"]):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                self.assertEqual(main(arguments), 2)


if __name__ == "__main__":
    unittest.main()