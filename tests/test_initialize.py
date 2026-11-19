from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from strata_comfy_studio.initialize import (
    InitializationError,
    configure_listeners,
    initialize_config,
    migrate_config,
    prepare_runtime_files,
    set_vram_threshold,
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

    def test_vram_threshold_updates_only_json_and_keeps_private_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = sample_config()
            data["strata"]["api_key"] = "private-test-key"
            config_path = root / "config.json"
            config_path.write_text(json.dumps(data), encoding="utf-8")

            set_vram_threshold(12000, config_path)

            updated = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["strata"]["min_free_vram_mib"], 12000)
            self.assertEqual(updated["strata"]["api_key"], "private-test-key")
            self.assertEqual(config_path.stat().st_mode & 0o777, 0o600)

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

    def test_config_migration_adds_defaults_and_preserves_existing_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = sample_config()
            current.pop("coordination")
            current["generation"]["prompt_skill_version"] = 2
            current["generation"].pop("system_prompt_file")
            current["generation"]["system_prompt"] = "Old prompt rules"
            current["generation"].update({"default_width": 2048, "default_height": 2048,
                                          "sampler_name": "euler", "scheduler": "simple"})
            current["strata"]["api_key"] = "keep-this-secret"
            current["app"]["access_token"] = "keep-this-token"
            config_path = root / "config.json"
            config_path.write_text(json.dumps(current), encoding="utf-8")
            template = sample_config()
            template["coordination"] = {
                "handoff_timeout_s": 120,
                "generation_timeout_s": 1800,
                "poll_interval_s": 1.0,
            }
            template["generation"]["prompt_skill_version"] = 3
            template["generation"]["system_prompt_file"] = "data/strata/minimax_h3_image_t2i.txt"
            template_path = root / "config.example.json"
            template_path.write_text(json.dumps(template), encoding="utf-8")

            migrated_path, added = migrate_config(template_path, config_path)

            migrated = json.loads(migrated_path.read_text(encoding="utf-8"))
            self.assertIn("coordination", added)
            self.assertEqual(migrated["strata"]["api_key"], "keep-this-secret")
            self.assertEqual(migrated["app"]["access_token"], "keep-this-token")
            self.assertEqual(migrated["generation"]["system_prompt_file"],
                             "data/strata/minimax_h3_image_t2i.txt")
            self.assertNotIn("system_prompt", migrated["generation"])
            self.assertEqual(migrated["generation"]["prompt_skill_version"], 3)
            self.assertNotIn("default_width", migrated["generation"])
            self.assertNotIn("default_height", migrated["generation"])
            self.assertEqual(config_path.stat().st_mode & 0o777, 0o600)

    def test_runtime_files_are_derived_from_json_and_exclude_qwen35(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "strata-source.json"
            source_path.write_text(json.dumps({
                "api_key": "source-key",
                "args": ["--max-context", "32768"],
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
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            self.assertEqual(runtime["api_key"], "private-key")
            self.assertEqual(runtime["host"], "127.0.0.1")
            self.assertEqual(runtime["port"], 8080)
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