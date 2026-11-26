from __future__ import annotations

import unittest

from comfy_studio.modes import (
    DEFAULT_MODE_ID,
    IMAGE_MODES,
    MAX_SOURCE_IMAGES,
    get_mode,
    public_mode_options,
    validate_source_image_count,
)
from comfy_studio.settings import load_config, project_root


class ImageModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = project_root()
        cls.settings = load_config(root / "data" / "config.example.json")

    def test_registered_modes_have_complete_configuration(self) -> None:
        for mode in IMAGE_MODES:
            config = self.settings.data["generation_modes"][mode.id]
            with self.subTest(mode=mode.id):
                self.assertTrue(config["workflow"])
                self.assertTrue(config["system_prompt_file"])
                self.assertTrue(set(mode.required_model_keys).issubset(config["models"]))

    def test_public_mode_options_include_capabilities_and_presentation(self) -> None:
        generation = self.settings.section("generation")
        options = public_mode_options(generation)

        self.assertEqual([item["id"] for item in options], [mode.id for mode in IMAGE_MODES])
        self.assertEqual(DEFAULT_MODE_ID, get_mode(DEFAULT_MODE_ID).id)
        edit = next(item for item in options if item["id"] == "edit")
        self.assertEqual(edit["source_images"]["min_count"], 1)
        self.assertEqual(edit["source_images"]["max_count"], MAX_SOURCE_IMAGES)
        self.assertEqual(edit["resolution"]["strategy"], "source-image")
        self.assertEqual(edit["ui"]["generate_label"], "Edit image")
        self.assertNotIn("legacy_local_storage_key", edit)
        self.assertNotIn("legacy_options_key", edit)
        character_swap = next(item for item in options if item["id"] == "character-swap")
        self.assertEqual(character_swap["source_images"]["min_count"], 2)
        self.assertEqual(character_swap["source_images"]["max_count"], MAX_SOURCE_IMAGES)
        self.assertEqual(character_swap["resolution"]["strategy"], "source-image")
        self.assertEqual(character_swap["resolution"]["megapixels"]["default"], 1.0)
        self.assertEqual(character_swap["ui"]["generate_label"], "Swap character")
        for item in options:
            self.assertNotIn("Strata", item["ui"]["composer_hint"])

    def test_source_image_limits_are_owned_by_the_selected_mode(self) -> None:
        text_to_image = get_mode(DEFAULT_MODE_ID)
        edit = get_mode("edit")

        validate_source_image_count(text_to_image, 0)
        with self.assertRaisesRegex(ValueError, "does not accept source images"):
            validate_source_image_count(text_to_image, 1)
        with self.assertRaisesRegex(ValueError, "requires one base image"):
            validate_source_image_count(edit, 0)
        validate_source_image_count(edit, MAX_SOURCE_IMAGES)
        character_swap = get_mode("character-swap")
        with self.assertRaisesRegex(ValueError, "at least one character reference"):
            validate_source_image_count(character_swap, 1)
        validate_source_image_count(character_swap, MAX_SOURCE_IMAGES)

    def test_unknown_mode_is_not_silently_mapped_to_the_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported image mode"):
            get_mode("unregistered-mode")


if __name__ == "__main__":
    unittest.main()
