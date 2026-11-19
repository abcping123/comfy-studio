from __future__ import annotations

import unittest

from strata_comfy_studio.settings import load_config, project_root
from strata_comfy_studio.workflow import WorkflowError, build_workflow


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = load_config(project_root() / "config.example.json")

    def test_prompt_flows_directly_to_minimax_conditioning(self) -> None:
        graph = build_workflow(self.settings, "A glass greenhouse at dawn", 7)
        self.assertEqual(graph["4"]["inputs"]["prompt"], "A glass greenhouse at dawn")
        self.assertEqual(graph["4"]["inputs"]["width"], 2048)
        self.assertEqual(graph["4"]["inputs"]["height"], 2048)
        self.assertEqual(graph["2"]["inputs"]["clip_name"],
                         self.settings.data["comfyui"]["models"]["text_encoder"])

    def test_qwen35_and_prompt_enhancement_branch_are_absent(self) -> None:
        graph = build_workflow(self.settings, "A red bicycle", 99)
        serialized = str(graph).lower()
        self.assertNotIn("qwen3.5", serialized)
        self.assertNotIn("textgeneratev2", serialized)
        self.assertNotIn("comfyswitchnode", serialized)
        self.assertEqual(graph["4"]["class_type"], "MiniMaxH3ImageConditioning")

    def test_dimensions_seed_and_prompt_are_validated(self) -> None:
        with self.assertRaisesRegex(WorkflowError, "unsupported aspect ratio"):
            build_workflow(self.settings, "A scene", 1, aspect_ratio="5:7")
        with self.assertRaisesRegex(WorkflowError, "between 0.1 and 16.0"):
            build_workflow(self.settings, "A scene", 1, megapixels=20)
        with self.assertRaisesRegex(WorkflowError, "unsigned 64-bit"):
            build_workflow(self.settings, "A scene", -1)
        with self.assertRaisesRegex(WorkflowError, "non-empty"):
            build_workflow(self.settings, "  ", 1)

    def test_ratio_megapixels_and_sampler_map_to_workflow(self) -> None:
        graph = build_workflow(self.settings, "A landscape", 5, "4:3 (Standard)", 4.0,
                               "euler_ancestral", "normal")
        self.assertEqual(graph["4"]["inputs"]["width"], 2368)
        self.assertEqual(graph["4"]["inputs"]["height"], 1760)
        self.assertEqual(graph["6"]["inputs"]["sampler_name"], "euler_ancestral")
        self.assertEqual(graph["6"]["inputs"]["scheduler"], "normal")

    def test_unconfigured_sampler_is_rejected(self) -> None:
        with self.assertRaisesRegex(WorkflowError, "KSampler options"):
            build_workflow(self.settings, "A landscape", 5, sampler_name="not-a-sampler")


if __name__ == "__main__":
    unittest.main()