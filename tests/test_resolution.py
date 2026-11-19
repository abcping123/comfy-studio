from __future__ import annotations

import unittest

from strata_comfy_studio.resolution import calculate_resolution, normalize_aspect_ratio
from strata_comfy_studio.settings import load_config, project_root


class ResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generation = load_config(project_root() / "config.example.json").data["generation"]

    def test_comfy_square_four_megapixels_matches_2048(self) -> None:
        self.assertEqual(calculate_resolution("1:1 (Square)", 4.0, 32,
                                              self.generation["aspect_ratios"]), (2048, 2048))

    def test_comfy_standard_landscape_matches_rounded_multiple(self) -> None:
        self.assertEqual(calculate_resolution("4:3 (Standard)", 4.0, 32,
                                              self.generation["aspect_ratios"]), (2368, 1760))

    def test_every_ratio_is_aligned_to_configured_multiple(self) -> None:
        multiple = self.generation["resolution_multiple"]
        for option in self.generation["aspect_ratios"]:
            with self.subTest(ratio=option["label"]):
                width, height = calculate_resolution(option["label"], 4.0, multiple,
                                                     self.generation["aspect_ratios"])
                self.assertEqual(width % multiple, 0)
                self.assertEqual(height % multiple, 0)

    def test_unsupported_ratio_maps_to_nearest_native_option(self) -> None:
        options = self.generation["aspect_ratios"]
        self.assertEqual(normalize_aspect_ratio("5:7", options), ("3:4 (Portrait Standard)", True))
        self.assertEqual(normalize_aspect_ratio("4:3 (Standard)", options), ("4:3 (Standard)", False))
        self.assertEqual(normalize_aspect_ratio("3:4", options), ("3:4 (Portrait Standard)", False))


if __name__ == "__main__":
    unittest.main()