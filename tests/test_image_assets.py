from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from PIL import Image

from comfy_studio.image_assets import SourceImageError, inspect_source_image


class ImageAssetTests(unittest.TestCase):
    def make_image(self, image_format: str) -> bytes:
        content = io.BytesIO()
        Image.new("RGB", (12, 8), color=(30, 80, 120)).save(content, format=image_format)
        return content.getvalue()

    def test_allowed_formats_are_verified_and_report_dimensions(self) -> None:
        for image_format, content_type, extension in (
            ("PNG", "image/png", ".png"),
            ("JPEG", "image/jpeg", ".jpg"),
            ("WEBP", "image/webp", ".webp"),
        ):
            with self.subTest(image_format=image_format):
                result = inspect_source_image(self.make_image(image_format), content_type)
                self.assertEqual((result.content_type, result.extension), (content_type, extension))
                self.assertEqual((result.width, result.height), (12, 8))

    def test_malformed_and_mismatched_images_are_rejected(self) -> None:
        with self.assertRaisesRegex(SourceImageError, "not a valid"):
            inspect_source_image(b"not an image", "image/png")
        with self.assertRaisesRegex(SourceImageError, "does not match"):
            inspect_source_image(self.make_image("PNG"), "image/jpeg")

    def test_upload_size_limit_is_enforced(self) -> None:
        with patch("comfy_studio.image_assets.MAX_SOURCE_IMAGE_BYTES", 1):
            with self.assertRaisesRegex(SourceImageError, "25 MiB"):
                inspect_source_image(self.make_image("PNG"), "image/png")


if __name__ == "__main__":
    unittest.main()
