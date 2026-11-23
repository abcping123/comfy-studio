from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from PIL import Image, UnidentifiedImageError


MAX_SOURCE_IMAGE_BYTES = 25 * 1024 * 1024
MAX_SOURCE_IMAGE_PIXELS = 100_000_000
Image.MAX_IMAGE_PIXELS = MAX_SOURCE_IMAGE_PIXELS


class SourceImageError(ValueError):
    """An uploaded image is unsupported, malformed, or exceeds its limits."""


@dataclass(frozen=True)
class InspectedImage:
    content_type: str
    extension: str
    width: int
    height: int


_FORMATS = {
    "PNG": ("image/png", ".png"),
    "JPEG": ("image/jpeg", ".jpg"),
    "WEBP": ("image/webp", ".webp"),
}


def inspect_source_image(content: bytes, declared_type: str) -> InspectedImage:
    if not content or len(content) > MAX_SOURCE_IMAGE_BYTES:
        raise SourceImageError("source images must be non-empty and no larger than 25 MiB")
    try:
        with Image.open(BytesIO(content)) as image:
            image_format = image.format
            width, height = image.size
            image.verify()
        with Image.open(BytesIO(content)) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise SourceImageError("animated images are not supported")
            if image.getexif().get(274) in {5, 6, 7, 8}:
                width, height = height, width
    except SourceImageError:
        raise
    except (UnidentifiedImageError, OSError, RuntimeError, SyntaxError,
            ValueError, Image.DecompressionBombError) as error:
        raise SourceImageError("uploaded file is not a valid PNG, JPEG, or WebP image") from error

    image_info = _FORMATS.get(image_format or "")
    if image_info is None or image_info[0] != declared_type:
        raise SourceImageError("image content does not match its PNG, JPEG, or WebP content type")
    if width <= 0 or height <= 0 or width * height > MAX_SOURCE_IMAGE_PIXELS:
        raise SourceImageError("source images must not exceed 100 megapixels")
    return InspectedImage(image_info[0], image_info[1], width, height)
