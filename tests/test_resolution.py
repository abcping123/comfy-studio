from __future__ import annotations

import unittest

from strata_comfy_studio.resolution import calculate_edit_resolution, calculate_resolution, normalize_aspect_ratio
from strata_comfy_studio.settings import load_config, project_root


class ResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generation = load_config(project_root() / "data" / "config.example.json").data["generation"]

    def test_comfy_square_four_megapixels_matches_2048(self) -> None:
        self.assertEqual(calculate_resolution("1:1 (Square)", 4.0, 32,
                                              self.generation["aspect_ratios"]), (2048, 2048))

    def test_comfy_standard_landscape_matches_rounded_multiple(self) -> None:
        self.assertEqual(calculate_resolution("4:3 (Standard)", 4.0, 32,
                                              self.generation["aspect_ratios"]), (2368, 1760))

    def test_megapixel_target_supports_half_to_twenty_mp(self) -> None:
        self.assertEqual(calculate_resolution("1:1 (Square)", 0.5, 32,
                                              self.generation["aspect_ratios"]), (736, 736))
        self.assertEqual(calculate_resolution("1:1 (Square)", 20.0, 32,
                                              self.generation["aspect_ratios"]), (4576, 4576))
        for megapixels in (0.49, 20.01):
            with self.subTest(megapixels=megapixels):
                with self.assertRaisesRegex(ValueError, "between 0.5 and 20.0"):
                    calculate_resolution("1:1 (Square)", megapixels, 32,
                                         self.generation["aspect_ratios"])

    def test_every_ratio_is_aligned_to_configured_multiple(self) -> None:
        multiple = self.generation["resolution_multiple"]
        for option in self.generation["aspect_ratios"]:
            with self.subTest(ratio=option["label"]):
                width, height = calculate_resolution(option["label"], 4.0, multiple,
                                                     self.generation["aspect_ratios"])
                self.assertEqual(width % multiple, 0)
                self.assertEqual(height % multiple, 0)

    def test_edit_dimensions_follow_source_ratio_and_are_aligned(self) -> None:
        width, height = calculate_edit_resolution(1920, 1080, 4.0, 32)
        self.assertEqual(width % 32, 0)
        self.assertEqual(height % 32, 0)
        self.assertAlmostEqual(width / height, 16 / 9, delta=0.03)
        self.assertLessEqual(width * height, 4 * 1024 * 1024)
        with self.assertRaisesRegex(ValueError, "between 0.5 and 16.0"):
            calculate_edit_resolution(1920, 1080, 16.01, 32)

    def test_unsupported_ratio_maps_to_nearest_native_option(self) -> None:
        options = self.generation["aspect_ratios"]
        self.assertEqual(normalize_aspect_ratio("5:7", options), ("3:4 (Portrait Standard)", True))
        self.assertEqual(normalize_aspect_ratio("4:3 (Standard)", options), ("4:3 (Standard)", False))
        self.assertEqual(normalize_aspect_ratio("3:4", options), ("3:4 (Portrait Standard)", False))


if __name__ == "__main__":
    unittest.main()