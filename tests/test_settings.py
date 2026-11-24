from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from comfy_studio.modes import IMAGE_MODES
from comfy_studio.settings import ConfigError, load_config, validate_config


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
        "schema_version": 3,
        "app": {
            "listen_host": "127.0.0.1",
            "port": 8765,
            "access_token": "test-token",
        },
        "paths": {
            "strata_runtime_config": "temp/strata/runtime.json",
            "strata_sessions": "temp/strata",
            "database": "temp/studio.sqlite3",
            "temp_root": "temp",
        },
        "strata": {
            "endpoint": "http://127.0.0.1:8080",
            "python": "/srv/strata/.venv/bin/python",
            "server_script": "/srv/strata/serve/server.py",
            "source_config_json": "/srv/strata/config.json",
            "api_key": "test-api-key",
            "model": "qwen-q2",
            "idle_unload_s": 600,
            "request_timeout_s": 900,
        },
        "comfyui": {
            "endpoint": "http://127.0.0.1:8188",
            "python": "/srv/comfy/.venv/bin/python",
            "main_script": "/srv/comfy/main.py",
            "working_directory": "/srv/comfy",
            "model_directories": model_directories,
            "input_directory": "temp/comfyui/input",
            "output_directory": "temp/comfyui",
            "gpu_id": 0,
            "request_timeout_s": 1800,
            "poll_interval_s": 1.0,
        },
        "generation_modes": {
            "text-to-image": {
                "workflow": "data/comfyui/minimax_h3_image_t2i.json",
                "system_prompt_file": "data/strata/minimax_h3_image_t2i.txt",
                "models": minimax_models,
            },
            "edit": {
                "workflow": "data/comfyui/minimax_h3_image_edit.json",
                "system_prompt_file": "data/strata/minimax_h3_image_edit.txt",
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


def legacy_sample_config() -> dict:
    data = sample_config()
    generation_modes = data.pop("generation_modes")
    data["schema_version"] = 1
    comfy = data["comfyui"]
    model_directories = comfy.pop("model_directories")
    comfy.update({
        "diffusion_models_dir": model_directories["diffusion_models"],
        "text_encoders_dir": model_directories["text_encoders"],
        "vae_dir": model_directories["vae"],
        "models": {
            key: model["file"]
            for key, model in generation_modes["text-to-image"]["models"].items()
        },
    })
    for mode in IMAGE_MODES:
        if mode.legacy_workflow_path_key is not None:
            data["paths"][mode.legacy_workflow_path_key] = generation_modes[mode.id]["workflow"]
        if mode.legacy_system_prompt_key is not None:
            data["generation"][mode.legacy_system_prompt_key] = (
                generation_modes[mode.id]["system_prompt_file"]
            )
    return data


class SettingsTests(unittest.TestCase):
    def test_valid_local_config(self) -> None:
        self.assertEqual(validate_config(sample_config())["schema_version"], 3)

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
            self.assertEqual(settings.path("paths.database"),
                             Path(directory) / "temp/studio.sqlite3")

    def test_version_one_config_migrates_to_per_mode_configuration(self) -> None:
        data = legacy_sample_config()
        del data["paths"]["workflow_edit_api"]
        del data["comfyui"]["input_directory"]
        del data["generation"]["edit_system_prompt_file"]

        validated = validate_config(data)

        self.assertEqual(validated["schema_version"], 3)
        self.assertEqual(validated["generation_modes"]["edit"]["workflow"],
                         "data/comfyui/minimax_h3_image_edit.json")
        self.assertEqual(validated["comfyui"]["input_directory"], "temp/comfyui/input")
        self.assertEqual(validated["generation_modes"]["edit"]["system_prompt_file"],
                         "data/strata/minimax_h3_image_edit.txt")
        self.assertEqual(
            validated["generation_modes"]["edit"]["models"]["text_encoder"]["file"],
            "qwen3vl_minimax.safetensors",
        )

    def test_version_two_config_migrates_to_generation_modes_name(self) -> None:
        data = sample_config()
        data["schema_version"] = 2
        data["image_modes"] = data.pop("generation_modes")

        validated = validate_config(data)

        self.assertEqual(validated["schema_version"], 3)
        self.assertIn("generation_modes", validated)
        self.assertNotIn("image_modes", validated)

    def test_mode_workflows_and_prompts_default_from_the_registry(self) -> None:
        data = sample_config()
        for mode in IMAGE_MODES:
            data["generation_modes"][mode.id].pop("workflow")
            data["generation_modes"][mode.id].pop("system_prompt_file")

        validated = validate_config(data)

        for mode in IMAGE_MODES:
            with self.subTest(mode=mode.id):
                self.assertEqual(validated["generation_modes"][mode.id]["workflow"],
                                 mode.default_workflow_path)
                self.assertEqual(validated["generation_modes"][mode.id]["system_prompt_file"],
                                 mode.default_system_prompt_file)

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
        data["strata"]["endpoint"] = "http://127.0.0.1:99999"
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