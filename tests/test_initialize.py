from __future__ import annotations

import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from comfy_studio.initialize import (
    InitializationError,
    configure_listeners,
    initialize_config,
    prepare_runtime_files,
)
from test_settings import sample_config
from comfy_studio.settings import load_config


def _venv_python(root: Path) -> Path:
    root = root.resolve()
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def _assert_private_file(test_case: unittest.TestCase, path: Path) -> None:
    if os.name == "nt":
        result = subprocess.run(
            ["icacls", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        entries = [line.strip() for line in result.stdout.splitlines() if ":(" in line]
        test_case.assertEqual(len(entries), 1)
        test_case.assertTrue(entries[0].endswith(":(F)"))
    else:
        test_case.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)


class InitializeTests(unittest.TestCase):
    def test_bonsai_initialization_does_not_require_a_strata_repository_or_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comfyui_repo = root / "ComfyUI"
            python = _venv_python(comfyui_repo)
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("python", encoding="utf-8")
            if os.name != "nt":
                python.chmod(0o755)
            (comfyui_repo / "main.py").write_text("", encoding="utf-8")
            template = sample_config()
            template["chat_backends"]["strata"]["api_key"] = ""
            template_path = root / "config.example.json"
            template_path.write_text(json.dumps(template), encoding="utf-8")
            destination = root / "config.json"

            initialize_config(
                template_path,
                destination,
                comfyui_repo_path=comfyui_repo,
                chat_backend="bonsai",
            )

            active = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(active["comfyui"]["python"], str(python.resolve()))
            self.assertEqual(active["comfyui"]["working_directory"], str(comfyui_repo.resolve()))
            self.assertEqual(active["chat_backends"]["strata"]["api_key"], "")
            self.assertTrue(active["app"]["access_token"])

    def test_initialization_imports_secret_without_printing_and_sets_private_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"HOME": directory, "USERPROFILE": directory}
        ):
            root = Path(directory)
            source_path = root / "strata.json"
            source_path.write_text(json.dumps({"api_key": "private-test-key"}), encoding="utf-8")
            template = sample_config()
            template["chat_backends"]["strata"]["source_config_json"] = "~/strata.json"
            template["chat_backends"]["strata"]["api_key"] = ""
            template_path = root / "config.example.json"
            template_path.write_text(json.dumps(template), encoding="utf-8")
            destination = root / "config.json"

            initialize_config(template_path, destination)

            active = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(active["chat_backends"]["strata"]["api_key"], "private-test-key")
            self.assertTrue(active["app"]["access_token"])
            _assert_private_file(self, destination)

    def test_initialization_generates_private_api_key_when_source_has_none(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "strata.json"
            source_path.write_text(json.dumps({"args": []}), encoding="utf-8")
            template = sample_config()
            template["chat_backends"]["strata"]["source_config_json"] = str(source_path)
            template["chat_backends"]["strata"]["api_key"] = ""
            template_path = root / "config.example.json"
            template_path.write_text(json.dumps(template), encoding="utf-8")
            destination = root / "config.json"

            initialize_config(template_path, destination)

            active = json.loads(destination.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(active["chat_backends"]["strata"]["api_key"]), 32)
            self.assertTrue(active["app"]["access_token"])
            _assert_private_file(self, destination)

    def test_initialization_refuses_to_overwrite_active_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "strata.json"
            source.write_text(json.dumps({"api_key": "private-test-key"}), encoding="utf-8")
            template = sample_config()
            template["chat_backends"]["strata"]["source_config_json"] = str(source)
            template_path = root / "config.example.json"
            template_path.write_text(json.dumps(template), encoding="utf-8")
            destination = root / "config.json"
            destination.write_text("existing", encoding="utf-8")

            with self.assertRaisesRegex(InitializationError, "refusing to overwrite"):
                initialize_config(template_path, destination)

    def test_initialization_derives_paths_from_repository_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            strata_repo = root / "Strata"
            comfyui_repo = root / "ComfyUI"
            for path in (_venv_python(strata_repo), _venv_python(comfyui_repo)):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("python", encoding="utf-8")
                if os.name != "nt":
                    path.chmod(0o755)
            strata_server = strata_repo / "serve/server.py"
            strata_server.parent.mkdir(parents=True)
            strata_server.write_text("", encoding="utf-8")
            (comfyui_repo / "main.py").write_text("", encoding="utf-8")
            (strata_repo / "strata-q2_0.json").write_text(
                json.dumps({"api_key": "private-test-key"}), encoding="utf-8",
            )

            template = sample_config()
            template["chat_backends"]["strata"]["source_config_json"] = "/old/path/strata-q2_0.json"
            template["chat_backends"]["strata"]["api_key"] = ""
            template_path = root / "config.example.json"
            template_path.write_text(json.dumps(template), encoding="utf-8")
            destination = root / "config.json"

            initialize_config(template_path, destination, strata_repo, comfyui_repo)

            active = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(active["chat_backends"]["strata"]["python"], str(_venv_python(strata_repo)))
            self.assertEqual(active["chat_backends"]["strata"]["server_script"], str(strata_server.resolve()))
            self.assertEqual(active["chat_backends"]["strata"]["source_config_json"],
                             str((strata_repo / "strata-q2_0.json").resolve()))
            self.assertEqual(active["chat_backends"]["strata"]["api_key"], "private-test-key")
            self.assertEqual(active["comfyui"]["python"], str(_venv_python(comfyui_repo)))
            self.assertEqual(active["comfyui"]["main_script"],
                             str((comfyui_repo / "main.py").resolve()))
            self.assertEqual(active["comfyui"]["working_directory"], str(comfyui_repo.resolve()))
            self.assertEqual(active["comfyui"]["output_directory"], "temp/session")
            self.assertEqual(active["comfyui"]["input_directory"], "temp/session")

    def test_initialization_derives_paths_from_terminal_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            strata_repo = root / "Strata"
            terminal_root = strata_repo / "terminal"
            comfyui_repo = root / "ComfyUI"
            for path in (_venv_python(terminal_root), _venv_python(comfyui_repo)):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("python", encoding="utf-8")
                if os.name != "nt":
                    path.chmod(0o755)
            strata_server = terminal_root / "serve/server.py"
            strata_server.parent.mkdir(parents=True)
            strata_server.write_text("", encoding="utf-8")
            (terminal_root / "strata-q2_0.json").write_text(
                json.dumps({"api_key": "private-test-key"}), encoding="utf-8",
            )
            (comfyui_repo / "main.py").write_text("", encoding="utf-8")

            template = sample_config()
            template["chat_backends"]["strata"]["source_config_json"] = "/old/path/strata-q2_0.json"
            template["chat_backends"]["strata"]["api_key"] = ""
            template_path = root / "config.example.json"
            template_path.write_text(json.dumps(template), encoding="utf-8")
            destination = root / "config.json"

            initialize_config(template_path, destination, strata_repo, comfyui_repo)

            active = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(active["chat_backends"]["strata"]["python"], str(_venv_python(terminal_root)))
            self.assertEqual(active["chat_backends"]["strata"]["server_script"], str(strata_server.resolve()))
            self.assertEqual(active["chat_backends"]["strata"]["source_config_json"],
                             str((terminal_root / "strata-q2_0.json").resolve()))

    def test_initialization_uses_repository_venv_for_terminal_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            strata_repo = root / "Strata"
            terminal_root = strata_repo / "terminal"
            comfyui_repo = root / "ComfyUI"
            for path in (_venv_python(strata_repo), _venv_python(comfyui_repo)):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("python", encoding="utf-8")
                if os.name != "nt":
                    path.chmod(0o755)
            strata_server = terminal_root / "serve/server.py"
            strata_server.parent.mkdir(parents=True)
            strata_server.write_text("", encoding="utf-8")
            source_path = terminal_root / "strata-q2_0.json"
            source_path.write_text(json.dumps({"api_key": "private-test-key"}), encoding="utf-8")
            (comfyui_repo / "main.py").write_text("", encoding="utf-8")

            template = sample_config()
            template["chat_backends"]["strata"]["source_config_json"] = "/old/path/strata-q2_0.json"
            template["chat_backends"]["strata"]["api_key"] = ""
            template_path = root / "config.example.json"
            template_path.write_text(json.dumps(template), encoding="utf-8")
            destination = root / "config.json"

            initialize_config(template_path, destination, strata_repo, comfyui_repo)

            active = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(active["chat_backends"]["strata"]["python"], str(_venv_python(strata_repo)))
            self.assertEqual(active["chat_backends"]["strata"]["server_script"], str(strata_server.resolve()))
            self.assertEqual(active["chat_backends"]["strata"]["source_config_json"], str(source_path.resolve()))

    def test_listener_config_keeps_model_backends_private_and_preserves_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            data = sample_config()
            data["chat_backends"]["strata"]["api_key"] = "private-api-key"
            data["app"]["access_token"] = "private-app-token"
            config_path.write_text(json.dumps(data), encoding="utf-8")

            configure_listeners("0.0.0.0", 8443, 8081, allow_insecure_lan_http=True,
                                config_path=config_path, comfy_port=8188)

            updated = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["app"]["listen_host"], "0.0.0.0")
            self.assertEqual(updated["app"]["port"], 8443)
            self.assertTrue(updated["app"]["allow_insecure_lan_http"])
            for key in ("tls_cert_file", "tls_key_file", "http_redirect_host",
                        "http_redirect_port", "public_host"):
                self.assertNotIn(key, updated["app"])
            self.assertEqual(updated["chat_backends"]["strata"]["endpoint"], "http://127.0.0.1:8081")
            self.assertEqual(updated["comfyui"]["endpoint"], "http://127.0.0.1:8188")
            self.assertEqual(updated["app"]["access_token"], "private-app-token")
            self.assertEqual(updated["chat_backends"]["strata"]["api_key"], "private-api-key")
            _assert_private_file(self, config_path)

    def test_listener_config_omits_removed_transport_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            data = sample_config()
            data["chat_backends"]["strata"]["api_key"] = "private-api-key"
            data["app"]["access_token"] = "private-app-token"
            config_path.write_text(json.dumps(data), encoding="utf-8")

            configure_listeners("0.0.0.0", 8765, 8080, allow_insecure_lan_http=True,
                                config_path=config_path, comfy_port=8188)

            updated = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertTrue(updated["app"]["allow_insecure_lan_http"])
            for key in ("tls_cert_file", "tls_key_file", "http_redirect_host",
                        "http_redirect_port", "public_host"):
                self.assertNotIn(key, updated["app"])
            self.assertEqual(updated["chat_backends"]["strata"]["endpoint"], "http://127.0.0.1:8080")
            self.assertEqual(updated["comfyui"]["endpoint"], "http://127.0.0.1:8188")
            self.assertEqual(updated["chat_backends"]["strata"]["api_key"], "private-api-key")
            self.assertEqual(updated["app"]["access_token"], "private-app-token")

    def test_listener_config_can_target_a_registered_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(sample_config()), encoding="utf-8")

            configure_listeners(
                "127.0.0.1",
                8765,
                9940,
                config_path=config_path,
                chat_backend="bonsai",
            )

            updated = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(
                updated["chat_backends"]["bonsai"]["endpoint"],
                "http://127.0.0.1:9940",
            )
            self.assertEqual(
                updated["chat_backends"]["strata"]["endpoint"],
                "http://127.0.0.1:8080",
            )

    def test_runtime_config_is_copied_exactly_once_and_kept_private(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"HOME": directory, "USERPROFILE": directory}
        ):
            root = Path(directory)
            source_path = root / "strata-source.json"
            source_contents = json.dumps({
                "api_key": "source-key",
                "min_free_vram_mib": 4096,
                "args": ["--max-context", "32768"],
                "vision": {
                    "exe": "strata-vision",
                    "mmproj": "mmproj.gguf",
                    "model": "model.gguf",
                    "gpu": True,
                },
                "env": {"SHOULD_NOT_COPY": "value"},
            }, indent=2).encode("utf-8")
            source_path.write_bytes(source_contents)
            data = sample_config()
            data["chat_backends"]["strata"]["source_config_json"] = "~/strata-source.json"
            data["chat_backends"]["strata"]["api_key"] = "private-key"
            config_path = root / "config.json"
            config_path.write_text(json.dumps(data), encoding="utf-8")
            settings = load_config(config_path)

            runtime_path = prepare_runtime_files(settings)

            self.assertEqual(runtime_path, root / "temp" / "runtime" / "strata_runtime.json")
            self.assertTrue((root / "temp" / "runtime").is_dir())
            self.assertTrue((root / "temp" / "session").is_dir())
            self.assertTrue((root / "output").is_dir())
            self.assertTrue(settings.path("comfyui.output_directory").is_dir())
            self.assertTrue(settings.path("comfyui.input_directory").is_dir())
            self.assertEqual(runtime_path.read_bytes(), source_contents)
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            self.assertEqual(runtime["api_key"], "source-key")
            self.assertEqual(runtime["min_free_vram_mib"], 4096)
            self.assertEqual(runtime["vision"], {
                "exe": "strata-vision",
                "mmproj": "mmproj.gguf",
                "model": "model.gguf",
                "gpu": True,
            })
            self.assertEqual(runtime["env"], {"SHOULD_NOT_COPY": "value"})
            self.assertNotIn("host", runtime)
            self.assertNotIn("slot_save_path", runtime)
            _assert_private_file(self, runtime_path)
            self.assertFalse((root / "config" / "model_paths.yaml").exists())
            self.assertTrue(
                data["generation_modes"]["text-to-image"]["models"]["text_encoder"]["file"]
                .startswith("qwen3vl_")
            )

            source_path.unlink()
            prepare_runtime_files(settings)
            self.assertEqual(runtime_path.read_bytes(), source_contents)

    def test_existing_runtime_config_does_not_depend_on_or_overwrite_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_path = root / "temp" / "runtime" / "strata_runtime.json"
            runtime_path.parent.mkdir(parents=True)
            runtime_contents = b'{\n  "args": [],\n  "api_key": "runtime-key"\n}\n'
            runtime_path.write_bytes(runtime_contents)

            data = sample_config()
            data["chat_backends"]["strata"].pop("source_config_json")
            config_path = root / "config.json"
            config_path.write_text(json.dumps(data), encoding="utf-8")
            settings = load_config(config_path)

            self.assertEqual(prepare_runtime_files(settings), runtime_path)
            self.assertEqual(runtime_path.read_bytes(), runtime_contents)

    def test_invalid_existing_runtime_config_is_not_reseeded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "strata-source.json"
            source_path.write_text(json.dumps({"args": []}), encoding="utf-8")
            runtime_path = root / "temp" / "runtime" / "strata_runtime.json"
            runtime_path.parent.mkdir(parents=True)
            runtime_contents = b"{invalid json"
            runtime_path.write_bytes(runtime_contents)

            data = sample_config()
            data["chat_backends"]["strata"]["source_config_json"] = str(source_path)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(data), encoding="utf-8")
            settings = load_config(config_path)

            with self.assertRaisesRegex(InitializationError, "cannot read JSON config"):
                prepare_runtime_files(settings)
            self.assertEqual(runtime_path.read_bytes(), runtime_contents)

    def test_initialization_uses_existing_runtime_without_source_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            strata_repo = root / "Strata"
            comfyui_repo = root / "ComfyUI"
            for path in (_venv_python(strata_repo), _venv_python(comfyui_repo)):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("python", encoding="utf-8")
                if os.name != "nt":
                    path.chmod(0o755)
            strata_server = strata_repo / "serve/server.py"
            strata_server.parent.mkdir(parents=True)
            strata_server.write_text("", encoding="utf-8")
            (comfyui_repo / "main.py").write_text("", encoding="utf-8")

            runtime_path = root / "temp" / "runtime" / "strata_runtime.json"
            runtime_path.parent.mkdir(parents=True)
            runtime_path.write_text(
                json.dumps({"api_key": "runtime-key", "args": []}),
                encoding="utf-8",
            )
            template = sample_config()
            template["chat_backends"]["strata"].pop("source_config_json")
            template_path = root / "config.example.json"
            template_path.write_text(json.dumps(template), encoding="utf-8")

            destination = initialize_config(
                template_path,
                root / "config.json",
                strata_repo_path=strata_repo,
                comfyui_repo_path=comfyui_repo,
            )

            active = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(active["chat_backends"]["strata"]["api_key"], "runtime-key")
            self.assertNotIn("source_config_json", active["chat_backends"]["strata"])

    def test_initialization_checks_both_repositories_before_inspecting_strata_python(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            strata_repo = root / "Strata"
            strata_repo.mkdir()
            missing_comfy_repo = root / "MissingComfyUI"
            template = sample_config()
            template_path = root / "config.example.json"
            template_path.write_text(json.dumps(template), encoding="utf-8")

            with self.assertRaisesRegex(InitializationError, "ComfyUI repository does not exist"):
                initialize_config(
                    template_path,
                    root / "config.json",
                    strata_repo_path=strata_repo,
                    comfyui_repo_path=missing_comfy_repo,
                )

    def test_runtime_rejects_strata_batch_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "strata-source.json"
            source_path.write_text(json.dumps({"args": ["--batch", "2"]}), encoding="utf-8")
            data = sample_config()
            data["chat_backends"]["strata"]["source_config_json"] = str(source_path)
            data["chat_backends"]["strata"]["api_key"] = "private-key"
            config_path = root / "config.json"
            config_path.write_text(json.dumps(data), encoding="utf-8")
            settings = load_config(config_path)

            with self.assertRaisesRegex(InitializationError, "incompatible"):
                prepare_runtime_files(settings)


if __name__ == "__main__":
    unittest.main()