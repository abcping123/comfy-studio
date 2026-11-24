from __future__ import annotations

import copy
import unittest

from comfy_studio.settings import Settings, load_config, project_root
from comfy_studio.workflow import WorkflowError, build_workflow


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = project_root()
        cls.settings = Settings(root, load_config(root / "data" / "config.example.json").data)

    def test_prompt_flows_directly_to_minimax_conditioning(self) -> None:
        graph = build_workflow(self.settings, "A glass greenhouse at dawn", 7)
        self.assertEqual(graph["4"]["inputs"]["prompt"], "A glass greenhouse at dawn")
        self.assertEqual(graph["4"]["inputs"]["width"], 2048)
        self.assertEqual(graph["4"]["inputs"]["height"], 2048)
        self.assertEqual(graph["6"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(graph["6"]["inputs"]["scheduler"], "simple")
        text_encoder = self.settings.data["generation_modes"]["text-to-image"]["models"]["text_encoder"]
        self.assertEqual(graph["2"]["inputs"]["clip_name"], text_encoder["file"])

    def test_edit_workflow_uses_its_mode_specific_model_configuration(self) -> None:
        config = copy.deepcopy(self.settings.data)
        config["generation_modes"]["edit"]["models"]["text_encoder"]["file"] = "edit-encoder.safetensors"
        settings = Settings(self.settings.root, config)
        graph = build_workflow(
            settings,
            "Make the image brighter",
            7,
            image_mode="edit",
            source_images=[{
                "filename": "base.png",
                "subfolder": "a" * 32,
                "width": 1024,
                "height": 1024,
            }],
        )

        self.assertEqual(graph["2"]["inputs"]["clip_name"], "edit-encoder.safetensors")

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
        with self.assertRaisesRegex(WorkflowError, "between 0.5 and 20.0"):
            build_workflow(self.settings, "A scene", 1, megapixels=20.01)
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

    def test_edit_workflow_resizes_base_and_maps_ordered_reference_images(self) -> None:
        source_images = [
            {"filename": "base.png", "subfolder": "a" * 32, "width": 1920, "height": 1080},
            {"filename": "ref2.jpg", "subfolder": "a" * 32, "width": 800, "height": 1200},
            {"filename": "ref3.png", "subfolder": "a" * 32, "width": 640, "height": 480},
            {"filename": "ref4.webp", "subfolder": "a" * 32, "width": 400, "height": 400},
            {"filename": "ref5.png", "subfolder": "a" * 32, "width": 1280, "height": 720},
        ]
        graph = build_workflow(
            self.settings, "Edit <Picture 1> using <Picture 2>.", 7,
            megapixels=4.0, image_mode="edit", source_images=source_images,
        )

        self.assertEqual(graph["9"]["inputs"]["image"], f"{'a' * 32}/base.png")
        self.assertEqual(graph["10"]["inputs"]["resize_type"], "scale total pixels")
        self.assertEqual(graph["10"]["inputs"]["resize_type.megapixels"], 4.0)
        self.assertEqual(graph["11"]["inputs"]["resize_type"], "scale to multiple")
        self.assertEqual(graph["11"]["inputs"]["resize_type.multiple"], 32)
        for index, image in enumerate(source_images, start=1):
            image_node = str(11 + index) if index > 1 else "11"
            self.assertEqual(graph["4"]["inputs"][f"images.image_{index}"], [image_node, 0])
            if index > 1:
                self.assertEqual(
                    graph[image_node]["inputs"]["image"],
                    f"{'a' * 32}/{image['filename']}",
                )
        self.assertEqual(graph["4"]["inputs"]["width"], ["12", 0])
        self.assertEqual(graph["4"]["inputs"]["height"], ["12", 1])

    def test_edit_workflow_requires_base_image_and_obeys_resize_limit(self) -> None:
        with self.assertRaisesRegex(WorkflowError, "requires one base image"):
            build_workflow(self.settings, "Edit", 7, image_mode="edit", source_images=[])
        with self.assertRaisesRegex(WorkflowError, "between 0.5 and 16.0"):
            build_workflow(
                self.settings, "Edit", 7, megapixels=16.01, image_mode="edit",
                source_images=[{
                    "filename": "base.png", "subfolder": "a" * 32, "width": 1024, "height": 1024,
                }],
            )
        with self.assertRaisesRegex(WorkflowError, "unsafe"):
            build_workflow(
                self.settings, "Edit", 7, image_mode="edit",
                source_images=[{
                    "filename": "../base.png", "subfolder": "a" * 32, "width": 1024, "height": 1024,
                }],
            )

    def test_unconfigured_sampler_is_rejected(self) -> None:
        with self.assertRaisesRegex(WorkflowError, "KSampler options"):
            build_workflow(self.settings, "A landscape", 5, sampler_name="not-a-sampler")

    def test_unregistered_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(WorkflowError, "unsupported image mode"):
            build_workflow(self.settings, "A scene", 1, image_mode="not-registered")


if __name__ == "__main__":
    unittest.main()