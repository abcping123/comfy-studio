from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from comfy_studio.settings import ConfigError, Settings, load_config, validate_config


def sample_config() -> dict:
    model_directories = {
        "diffusion_models": "/models/diffusion_models",
        "text_encoders": "/models/text_encoders",
        "vae": "/models/vae",
    }
    minimax_models = {
        "diffusion_model": {
            "directory": "diffusion_models",
            "file": "minimax.safetensors",
        },
        "text_encoder": {
            "directory": "text_encoders",
            "file": "qwen3vl_minimax.safetensors",
        },
        "vae": {
            "directory": "vae",
            "file": "vae.safetensors",
        },
    }
    return {
        "schema_version": 8,
        "app": {
            "listen_host": "127.0.0.1",
            "port": 8765,
            "access_token": "test-token",
        },
        "paths": {
            "runtime_root": "temp/runtime",
            "session_root": "temp/session",
            "output_archive": "output",
        },
        "chat_backends": {
            "strata": {
                "endpoint": "http://127.0.0.1:8080",
                "python": "/srv/strata/.venv/bin/python",
                "server_script": "/srv/strata/serve/server.py",
                "source_config_json": "/srv/strata/config.json",
                "runtime_config": "temp/runtime/strata_runtime.json",
                "api_key": "test-api-key",
                "model": "qwen-q2",
                "idle_unload_s": 600,
                "max_tokens": 8192,
                "request_timeout_s": 900,
            },
            "bonsai": {
                "endpoint": "http://127.0.0.1:9931",
                "server_binary": "/srv/prism-llama.cpp/bin/llama-server",
                "model_file": "/models/Bonsai-2-27B-PTQ1_0-CRACK.gguf",
                "mmproj_file": "/models/Ternary-Bonsai-2-27B-mmproj-BF16.gguf",
                "mmproj_gpu_offload": False,
                "model": "local-model",
                "ctx_size": 32768,
                "n_gpu_layers": "all",
                "flash_attn": "on",
                "parallel": 1,
                "n_predict": 16384,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.05,
                "presence_penalty": 0.0,
                "repeat_penalty": 1.0,
                "max_tokens": 8192,
                "request_timeout_s": 900,
                "startup_timeout_s": 900,
            },
        },
        "comfyui": {
            "endpoint": "http://127.0.0.1:8188",
            "python": "/srv/comfy/.venv/bin/python",
            "main_script": "/srv/comfy/main.py",
            "working_directory": "/srv/comfy",
            "model_directories": model_directories,
            "input_directory": "temp/session",
            "output_directory": "temp/session",
            "gpu_id": 0,
            "request_timeout_s": 1800,
            "poll_interval_s": 1.0,
        },
        "generation_modes": {
            "text-to-image": {
                "workflow": "data/comfyui/minimax_h3_image_t2i.json",
                "system_prompt_file": "data/prompts/minimax_h3_image_t2i.txt",
                "models": minimax_models,
            },
            "edit": {
                "workflow": "data/comfyui/minimax_h3_image_edit.json",
                "system_prompt_file": "data/prompts/minimax_h3_image_edit.txt",
                "models": {
                    key: dict(value) for key, value in minimax_models.items()
                },
            },
        },
        "generation": {
            "prompt_enhancement": False,
            "prompt_skill_version": 1,
            "aspect_ratios": [
                {"label": "1:1 (Square)", "width": 1, "height": 1},
                {"label": "4:3 (Standard)", "width": 4, "height": 3},
            ],
            "default_aspect_ratio": "1:1 (Square)",
            "default_megapixels": 4.0,
            "resolution_multiple": 32,
            "steps": 6,
            "cfg": 1.0,
            "sampler_options": ["euler", "euler_ancestral"],
            "scheduler_options": ["simple", "normal"],
            "default_sampler": "euler",
            "default_scheduler": "simple",
        },
        "coordination": {
            "generation_timeout_s": 1800,
            "poll_interval_s": 1.0,
        },
    }


class SettingsTests(unittest.TestCase):
    def test_valid_local_config(self) -> None:
        self.assertEqual(validate_config(sample_config())["schema_version"], 8)

    def test_chat_max_tokens_must_be_a_positive_integer_for_each_backend(self) -> None:
        for backend_id in ("strata", "bonsai"):
            data = sample_config()
            data["chat_backends"][backend_id]["max_tokens"] = 4096
            self.assertEqual(validate_config(data)["chat_backends"][backend_id]["max_tokens"], 4096)

            for value in (True, 0, -1, 1.5, "8192", None):
                data = sample_config()
                data["chat_backends"][backend_id]["max_tokens"] = value
                with self.subTest(backend=backend_id, value=value):
                    with self.assertRaisesRegex(
                            ConfigError,
                            f"chat_backends\\.{backend_id}\\.max_tokens must be a positive integer"):
                        validate_config(data)

            data = sample_config()
            del data["chat_backends"][backend_id]["max_tokens"]
            with self.subTest(backend=backend_id, value="missing"):
                with self.assertRaisesRegex(
                        ConfigError,
                        f"chat_backends\\.{backend_id}\\.max_tokens must be a positive integer"):
                    validate_config(data)

    def test_strata_source_config_is_optional_after_runtime_bootstrap(self) -> None:
        data = sample_config()
        data["chat_backends"]["strata"].pop("source_config_json")

        self.assertEqual(validate_config(data)["schema_version"], 8)

    def test_bonsai_mmproj_gpu_offload_must_be_boolean(self) -> None:
        data = sample_config()
        del data["chat_backends"]["bonsai"]["mmproj_gpu_offload"]
        with self.assertRaisesRegex(ConfigError, "mmproj_gpu_offload must be a boolean"):
            validate_config(data)

        for value in ("true", 1, None):
            data = sample_config()
            data["chat_backends"]["bonsai"]["mmproj_gpu_offload"] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ConfigError, "mmproj_gpu_offload must be a boolean"):
                    validate_config(data)

        data = sample_config()
        data["chat_backends"]["bonsai"]["mmproj_gpu_offload"] = False
        self.assertFalse(validate_config(data)["chat_backends"]["bonsai"]["mmproj_gpu_offload"])

    def test_only_schema_eight_is_supported(self) -> None:
        for version in range(1, 8):
            data = sample_config()
            data["schema_version"] = version
            with self.subTest(version=version):
                with self.assertRaisesRegex(ConfigError, "schema_version must be 8"):
                    validate_config(data)

    def test_default_megapixels_must_be_between_half_and_twenty(self) -> None:
        data = sample_config()
        data["generation"]["default_megapixels"] = 0.5
        self.assertEqual(validate_config(data)["generation"]["default_megapixels"], 0.5)
        data = sample_config()
        data["generation"]["default_megapixels"] = 20.0
        self.assertEqual(validate_config(data)["generation"]["default_megapixels"], 20.0)
        for megapixels in (0.49, 20.01):
            data = sample_config()
            data["generation"]["default_megapixels"] = megapixels
            with self.subTest(megapixels=megapixels):
                with self.assertRaisesRegex(ConfigError, "between 0.5 and 20.0"):
                    validate_config(data)

    def test_relative_paths_resolve_from_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(sample_config()), encoding="utf-8")
            settings = load_config(config_path)
            self.assertEqual(settings.path("paths.session_root"),
                             Path(directory) / "temp/session")

    def test_home_relative_paths_resolve_from_user_home(self) -> None:
        data = sample_config()
        data["paths"]["session_root"] = "~/studio-sessions"
        settings = Settings(Path.cwd(), validate_config(data))

        self.assertEqual(settings.path("paths.session_root"), Path.home() / "studio-sessions")

    def test_unregistered_backend_config_is_rejected(self) -> None:
        data = sample_config()
        data["chat_backends"]["future-model"] = {}

        with self.assertRaisesRegex(ConfigError, "unregistered backend"):
            validate_config(data)

    def test_selected_backend_can_load_without_unselected_backend_config(self) -> None:
        data = sample_config()
        data["chat_backends"].pop("bonsai")

        self.assertEqual(validate_config(data, required_backend="strata")["schema_version"], 8)
        with self.assertRaisesRegex(ConfigError, "chat_backends.bonsai is required"):
            validate_config(data, required_backend="bonsai")

    def test_load_config_records_the_selected_chat_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(sample_config()), encoding="utf-8")

            settings = load_config(config_path, chat_backend="bonsai")

        self.assertEqual(settings.chat_backend, "bonsai")

    def test_mode_workflow_and_prompt_paths_are_required(self) -> None:
        for key in ("workflow", "system_prompt_file"):
            data = sample_config()
            data["generation_modes"]["text-to-image"].pop(key)
            with self.subTest(key=key):
                with self.assertRaisesRegex(
                    ConfigError,
                    f"generation_modes.text-to-image.{key} must be a non-empty string",
                ):
                    validate_config(data)

    def test_comfy_input_directory_is_required(self) -> None:
        data = sample_config()
        data["comfyui"].pop("input_directory")

        with self.assertRaisesRegex(ConfigError, "comfyui.input_directory"):
            validate_config(data)

    def test_each_mode_can_select_its_own_models(self) -> None:
        data = sample_config()
        data["generation_modes"]["edit"]["models"]["diffusion_model"]["file"] = "edit.safetensors"

        validated = validate_config(data)

        self.assertEqual(
            validated["generation_modes"]["edit"]["models"]["diffusion_model"]["file"],
            "edit.safetensors",
        )
        self.assertEqual(
            validated["generation_modes"]["text-to-image"]["models"]["diffusion_model"]["file"],
            "minimax.safetensors",
        )

    def test_unregistered_mode_configuration_is_rejected(self) -> None:
        data = sample_config()
        data["generation_modes"]["future-mode"] = {}
        with self.assertRaisesRegex(ConfigError, "unregistered mode"):
            validate_config(data)

    def test_model_filenames_must_match_mode_requirements(self) -> None:
        data = sample_config()
        data["generation_modes"]["edit"]["models"]["text_encoder"]["file"] = "other-encoder.gguf"
        with self.assertRaisesRegex(ConfigError, "must start with qwen3vl_"):
            validate_config(data)

    def test_non_loopback_backend_is_rejected(self) -> None:
        data = sample_config()
        data["comfyui"]["endpoint"] = "http://192.168.1.5:8188"
        with self.assertRaisesRegex(ConfigError, "must be loopback"):
            validate_config(data)

    def test_lan_app_requires_http_opt_in_and_token(self) -> None:
        data = sample_config()
        data["app"]["listen_host"] = "192.168.1.20"
        with self.assertRaisesRegex(ConfigError, "allow_insecure_lan_http"):
            validate_config(data)

    def test_lan_app_accepts_explicit_http_opt_in_and_token(self) -> None:
        data = sample_config()
        data["app"].update({
            "listen_host": "192.168.1.20",
            "access_token": "test-token",
            "allow_insecure_lan_http": True,
        })
        self.assertEqual(validate_config(data)["app"]["listen_host"], "192.168.1.20")

    def test_plaintext_lan_listener_requires_explicit_opt_in_and_token(self) -> None:
        data = sample_config()
        data["app"].update({"listen_host": "0.0.0.0", "access_token": "test-token",
                            "allow_insecure_lan_http": True})
        self.assertEqual(validate_config(data)["app"]["listen_host"], "0.0.0.0")
        data["app"]["access_token"] = ""
        with self.assertRaisesRegex(ConfigError, "requires app.access_token"):
            validate_config(data)

    def test_invalid_url_port_is_rejected(self) -> None:
        data = sample_config()
        data["chat_backends"]["strata"]["endpoint"] = "http://127.0.0.1:99999"
        with self.assertRaisesRegex(ConfigError, "invalid"):
            validate_config(data)

    def test_prompt_enhancement_cannot_be_enabled(self) -> None:
        data = sample_config()
        data["generation"]["prompt_enhancement"] = True
        with self.assertRaisesRegex(ConfigError, "must be false"):
            validate_config(data)

    def test_invalid_port_is_rejected(self) -> None:
        data = sample_config()
        data["comfyui"]["endpoint"] = "http://127.0.0.1:0"
        with self.assertRaises(ConfigError):
            validate_config(data)

    def test_invalid_json_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "invalid JSON"):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()