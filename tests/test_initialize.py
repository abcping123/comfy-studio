from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from strata_comfy_studio.initialize import (
    InitializationError,
    configure_listeners,
    initialize_config,
    prepare_runtime_files,
)
from test_settings import sample_config
from strata_comfy_studio.settings import load_config


class InitializeTests(unittest.TestCase):
    def test_initialization_imports_secret_without_printing_and_sets_private_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "strata.json"
            source_path.write_text(json.dumps({"api_key": "private-test-key"}), encoding="utf-8")
            template = sample_config()
            template["strata"]["source_config_json"] = str(source_path)
            template["strata"]["api_key"] = ""
            template_path = root / "config.example.json"
            template_path.write_text(json.dumps(template), encoding="utf-8")
            destination = root / "config.json"

            initialize_config(template_path, destination)

            active = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(active["strata"]["api_key"], "private-test-key")
            self.assertTrue(active["app"]["access_token"])
            self.assertEqual(destination.stat().st_mode & 0o777, 0o600)

    def test_initialization_refuses_to_overwrite_active_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "strata.json"
            source.write_text(json.dumps({"api_key": "private-test-key"}), encoding="utf-8")
            template = sample_config()
            template["strata"]["source_config_json"] = str(source)
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
            for path in (strata_repo / ".venv/bin/python", comfyui_repo / ".venv/bin/python"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("python", encoding="utf-8")
                path.chmod(0o755)
            strata_server = strata_repo / "serve/server.py"
            strata_server.parent.mkdir(parents=True)
            strata_server.write_text("", encoding="utf-8")
            (comfyui_repo / "main.py").write_text("", encoding="utf-8")
            (strata_repo / "strata-q2_0.json").write_text(
                json.dumps({"api_key": "private-test-key"}), encoding="utf-8",
            )

            template = sample_config()
            template["strata"]["source_config_json"] = "/old/path/strata-q2_0.json"
            template["strata"]["api_key"] = ""
            template_path = root / "config.example.json"
            template_path.write_text(json.dumps(template), encoding="utf-8")
            destination = root / "config.json"

            initialize_config(template_path, destination, strata_repo, comfyui_repo)

            active = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(active["strata"]["python"], str(strata_repo / ".venv/bin/python"))
            self.assertEqual(active["strata"]["server_script"], str(strata_server))
            self.assertEqual(active["strata"]["source_config_json"],
                             str(strata_repo / "strata-q2_0.json"))
            self.assertEqual(active["strata"]["api_key"], "private-test-key")
            self.assertEqual(active["comfyui"]["python"], str(comfyui_repo / ".venv/bin/python"))
            self.assertEqual(active["comfyui"]["main_script"], str(comfyui_repo / "main.py"))
            self.assertEqual(active["comfyui"]["working_directory"], str(comfyui_repo))
            self.assertEqual(active["comfyui"]["output_directory"], str(comfyui_repo / "output"))
            self.assertEqual(active["comfyui"]["input_directory"], "temp/comfyui/input")

    def test_initialization_derives_paths_from_terminal_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            strata_repo = root / "Strata"
            terminal_root = strata_repo / "terminal"
            comfyui_repo = root / "ComfyUI"
            for path in (terminal_root / ".venv/bin/python", comfyui_repo / ".venv/bin/python"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("python", encoding="utf-8")
                path.chmod(0o755)
            strata_server = terminal_root / "serve/server.py"
            strata_server.parent.mkdir(parents=True)
            strata_server.write_text("", encoding="utf-8")
            (terminal_root / "strata-q2_0.json").write_text(
                json.dumps({"api_key": "private-test-key"}), encoding="utf-8",
            )
            (comfyui_repo / "main.py").write_text("", encoding="utf-8")

            template = sample_config()
            template["strata"]["source_config_json"] = "/old/path/strata-q2_0.json"
            template["strata"]["api_key"] = ""
            template_path = root / "config.example.json"
            template_path.write_text(json.dumps(template), encoding="utf-8")
            destination = root / "config.json"

            initialize_config(template_path, destination, strata_repo, comfyui_repo)

            active = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(active["strata"]["python"], str(terminal_root / ".venv/bin/python"))
            self.assertEqual(active["strata"]["server_script"], str(strata_server))
            self.assertEqual(active["strata"]["source_config_json"],
                             str(terminal_root / "strata-q2_0.json"))

    def test_listener_config_keeps_model_backends_private_and_preserves_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            data = sample_config()
            data["strata"]["api_key"] = "private-api-key"
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
            self.assertEqual(updated["strata"]["endpoint"], "http://127.0.0.1:8081")
            self.assertEqual(updated["comfyui"]["endpoint"], "http://127.0.0.1:8188")
            self.assertEqual(updated["app"]["access_token"], "private-app-token")
            self.assertEqual(updated["strata"]["api_key"], "private-api-key")
            self.assertEqual(config_path.stat().st_mode & 0o777, 0o600)

    def test_listener_config_omits_removed_transport_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            data = sample_config()
            data["strata"]["api_key"] = "private-api-key"
            data["app"]["access_token"] = "private-app-token"
            config_path.write_text(json.dumps(data), encoding="utf-8")

            configure_listeners("0.0.0.0", 8765, 8080, allow_insecure_lan_http=True,
                                config_path=config_path, comfy_port=8188)

            updated = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertTrue(updated["app"]["allow_insecure_lan_http"])
            for key in ("tls_cert_file", "tls_key_file", "http_redirect_host",
                        "http_redirect_port", "public_host"):
                self.assertNotIn(key, updated["app"])
            self.assertEqual(updated["strata"]["endpoint"], "http://127.0.0.1:8080")
            self.assertEqual(updated["comfyui"]["endpoint"], "http://127.0.0.1:8188")
            self.assertEqual(updated["strata"]["api_key"], "private-api-key")
            self.assertEqual(updated["app"]["access_token"], "private-app-token")

    def test_runtime_files_are_derived_from_json_and_exclude_qwen35(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "strata-source.json"
            source_path.write_text(json.dumps({
                "api_key": "source-key",
                "args": ["--max-context", "32768"],
                "vision": {
                    "exe": "strata-vision",
                    "mmproj": "mmproj.gguf",
                    "model": "model.gguf",
                    "gpu": True,
                },
                "env": {"SHOULD_NOT_COPY": "value"},
            }), encoding="utf-8")
            data = sample_config()
            data["strata"]["source_config_json"] = str(source_path)
            data["strata"]["api_key"] = "private-key"
            config_path = root / "config.json"
            config_path.write_text(json.dumps(data), encoding="utf-8")
            settings = load_config(config_path)

            runtime_path = prepare_runtime_files(settings)

            self.assertEqual(runtime_path, root / "temp" / "strata" / "runtime.json")
            self.assertTrue((root / "temp" / "comfyui").is_dir())
            self.assertTrue(settings.path("comfyui.output_directory").is_dir())
            self.assertTrue(settings.path("comfyui.input_directory").is_dir())
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            self.assertEqual(runtime["api_key"], "private-key")
            self.assertEqual(runtime["host"], "127.0.0.1")
            self.assertEqual(runtime["port"], 8080)
            self.assertEqual(runtime["vision"], {
                "exe": "strata-vision",
                "mmproj": "mmproj.gguf",
                "model": "model.gguf",
                "gpu": True,
            })
            self.assertNotIn("env", runtime)
            self.assertEqual(runtime_path.stat().st_mode & 0o777, 0o600)
            self.assertFalse((root / "config" / "model_paths.yaml").exists())
            self.assertTrue(data["comfyui"]["models"]["text_encoder"].startswith("qwen3vl_"))

    def test_runtime_rejects_strata_batch_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "strata-source.json"
            source_path.write_text(json.dumps({"args": ["--batch", "2"]}), encoding="utf-8")
            data = sample_config()
            data["strata"]["source_config_json"] = str(source_path)
            data["strata"]["api_key"] = "private-key"
            config_path = root / "config.json"
            config_path.write_text(json.dumps(data), encoding="utf-8")
            settings = load_config(config_path)

            with self.assertRaisesRegex(InitializationError, "incompatible"):
                prepare_runtime_files(settings)


if __name__ == "__main__":
    unittest.main()