from __future__ import annotations

import base64
import hashlib
import os
import re
import secrets
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .clients import ComfyUIClient
from .coordination import CoordinationError
from .modes import ModeDefinition, source_image_role
from .permissions import set_private_permissions
from .settings import Settings
from .storage import ConversationStore


class SessionAssets:
    def __init__(
        self,
        settings: Settings,
        store: ConversationStore,
        comfy: ComfyUIClient,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.settings = settings
        self.store = store
        self.comfy = comfy
        self._sleep = sleeper

    def session_asset_directory(self, conversation_id: str) -> Path:
        conversation_id = self.store.validate_id(conversation_id)
        root = self.settings.path("paths.session_root").resolve(strict=True)
        input_root = self.settings.path("comfyui.input_directory").resolve(strict=True)
        output_root = self.settings.path("comfyui.output_directory").resolve(strict=True)
        if input_root != root or output_root != root:
            raise CoordinationError("ComfyUI input and output directories must be the session root")
        directory = self.store.session_directory(conversation_id)
        if directory is None or directory.parent != root:
            raise CoordinationError("session directory is missing or outside Studio storage")
        return directory

    @staticmethod
    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        try:
            with path.open("rb") as image_file:
                for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as error:
            raise CoordinationError(f"could not read image for hashing: {error}") from error
        return digest.hexdigest()

    def relocate_generated_image(
        self, conversation_id: str, image: dict[str, str],
    ) -> dict[str, str]:
        conversation_id = self.store.validate_id(conversation_id)
        filename = image.get("filename", "")
        if (image.get("subfolder") != conversation_id or image.get("type") != "output"
                or not filename or Path(filename).name != filename or "\\" in filename
                or Path(filename).suffix.lower() != ".png"):
            raise CoordinationError("ComfyUI returned an unsafe generated image path")

        source_directory = self.session_asset_directory(conversation_id)
        source_path = source_directory / filename
        if source_path.is_symlink():
            raise CoordinationError("refusing to move a symbolic-link generated image")
        source_path = source_path.resolve(strict=True)
        if source_path.parent != source_directory or not source_path.is_file():
            raise CoordinationError("ComfyUI generated image is missing or unsafe")

        content_hash = self.sha256_file(source_path)
        destination_path = source_directory / f"{content_hash}.png"
        if destination_path.is_symlink():
            raise CoordinationError("refusing to overwrite a symbolic-link generated image")
        if destination_path.exists():
            if (not destination_path.is_file()
                    or self.sha256_file(destination_path) != content_hash):
                raise CoordinationError("generated image hash path contains different content")
            try:
                source_path.unlink()
            except OSError as error:
                raise CoordinationError(
                    f"could not remove the duplicate generated image: {error}"
                ) from error
        else:
            try:
                os.link(source_path, destination_path)
                source_path.unlink()
            except FileExistsError:
                if (destination_path.is_symlink() or not destination_path.is_file()
                        or self.sha256_file(destination_path) != content_hash):
                    raise CoordinationError("generated image hash path contains different content")
                try:
                    source_path.unlink()
                except OSError as error:
                    raise CoordinationError(
                        f"could not remove the duplicate generated image: {error}"
                    ) from error
            except OSError as error:
                raise CoordinationError(
                    f"could not store the generated image in its session: {error}"
                ) from error
        self.archive_generated_image(destination_path)
        return {"filename": destination_path.name, "subfolder": conversation_id, "type": "output"}

    def archive_generated_image(self, image_path: Path) -> Path:
        output_root = self.settings.path("paths.output_archive")
        if output_root.is_symlink():
            raise CoordinationError("refusing to write to a symbolic-link output archive")
        try:
            output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            output_root = output_root.resolve(strict=True)
            if image_path.is_symlink() or not image_path.is_file():
                raise CoordinationError("generated image is missing or unsafe")
            while True:
                timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%Hh%M_%S")
                destination = output_root / f"{timestamp}.png"
                if destination.is_symlink() or (destination.exists() and not destination.is_file()):
                    raise CoordinationError("output archive target is not a regular file")
                if destination.exists():
                    self._sleep(max(0.01, 1.0 - (time.time() % 1.0)))
                    continue
                temporary = output_root / f".{destination.name}.{secrets.token_hex(8)}.tmp"
                try:
                    with image_path.open("rb") as source, temporary.open("xb") as target:
                        shutil.copyfileobj(source, target)
                        target.flush()
                        os.fsync(target.fileno())
                    set_private_permissions(temporary)
                    os.link(temporary, destination)
                    temporary.unlink()
                    set_private_permissions(destination)
                    return destination
                except FileExistsError:
                    temporary.unlink(missing_ok=True)
                    self._sleep(max(0.01, 1.0 - (time.time() % 1.0)))
                except OSError as error:
                    temporary.unlink(missing_ok=True)
                    raise CoordinationError(f"could not archive generated image: {error}") from error
        except CoordinationError:
            raise
        except OSError as error:
            raise CoordinationError(f"could not archive generated image: {error}") from error

    def generated_image_path(self, generation: dict[str, Any]) -> Path:
        conversation_id = self.store.validate_id(generation.get("conversation_id", ""))
        filename = generation.get("filename")
        if (generation.get("subfolder") != conversation_id or generation.get("image_type") != "output"
                or not isinstance(filename, str) or not filename or Path(filename).name != filename
                or "\\" in filename or re.fullmatch(r"[0-9a-f]{64}\.png", filename) is None):
            raise CoordinationError("generated image metadata has an invalid path")
        directory = self.session_asset_directory(conversation_id)
        image_path = directory / filename
        if image_path.is_symlink():
            raise CoordinationError("refusing to serve a symbolic-link generated image")
        try:
            image_path = image_path.resolve(strict=True)
        except OSError as error:
            raise CoordinationError("generated image file is missing") from error
        if image_path.parent != directory or not image_path.is_file():
            raise CoordinationError("generated image file is missing or unsafe")
        return image_path

    def source_image_file(self, conversation_id: str, filename: str) -> Path:
        directory = self.session_asset_directory(conversation_id)
        if filename in {"", ".", ".."} or Path(filename).name != filename or "\\" in filename:
            raise CoordinationError("refusing an unsafe source image filename")
        path = directory / filename
        if path.is_symlink():
            raise CoordinationError("refusing to access a symbolic-link source image")
        return path

    def source_image_path(self, conversation_id: str, source_image: dict[str, Any]) -> Path:
        conversation_id = self.store.validate_id(conversation_id)
        content_type = source_image.get("content_type")
        extension = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
        }.get(content_type) if isinstance(content_type, str) else None
        content_hash = source_image.get("content_hash")
        filename = source_image.get("filename")
        if (extension is None or not isinstance(content_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", content_hash) is None
                or filename != f"{content_hash}{extension}"
                or source_image.get("subfolder") != conversation_id):
            raise CoordinationError("source image metadata has an invalid path")

        try:
            directory = self.session_asset_directory(conversation_id)
            if directory.is_symlink():
                raise CoordinationError("refusing to access a symbolic-link source image directory")
            image_path = self.source_image_file(conversation_id, filename)
            session_root = self.settings.path("paths.session_root").resolve(strict=True)
            resolved_directory = directory.resolve(strict=True)
            resolved_image_path = image_path.resolve(strict=True)
        except OSError as error:
            raise CoordinationError("source image file is missing") from error
        if (resolved_directory.parent != session_root
                or resolved_image_path.parent != resolved_directory
                or not resolved_image_path.is_file()):
            raise CoordinationError("source image file is missing or unsafe")
        return resolved_image_path

    def ensure_source_image_file(
        self,
        conversation_id: str,
        content: bytes,
        content_type: str,
        extension: str,
        content_hash: str,
    ) -> tuple[Path, bool]:
        filename = f"{content_hash}{extension}"
        image_path = self.source_image_file(conversation_id, filename)
        if image_path.exists():
            if not image_path.is_file() or self.sha256_file(image_path) != content_hash:
                raise CoordinationError("source image hash path contains different content")
            return image_path, False

        response = self.comfy.upload_image(filename, content, content_type, conversation_id)
        uploaded_name = response.get("name")
        uploaded_subfolder = response.get("subfolder")
        uploaded_type = response.get("type")
        if (uploaded_name != filename or uploaded_subfolder != conversation_id
                or uploaded_type != "input"):
            raise CoordinationError("ComfyUI returned an invalid source image path")
        image_path = self.source_image_file(conversation_id, filename)
        if not image_path.is_file() or self.sha256_file(image_path) != content_hash:
            raise CoordinationError("ComfyUI did not store the source image at its hash path")
        return image_path, True

    def encode_source_images_for_chat(
        self,
        conversation_id: str,
        mode: ModeDefinition,
        source_images: list[dict[str, Any]],
        backend_label: str,
        user_message: str,
    ) -> list[dict[str, Any]]:
        image_labels = [
            f"<Picture {index + 1}> ({source_image_role(mode, index)})"
            for index in range(len(source_images))
        ]
        prompt_text = user_message + "\n\nattached_source_images_in_order:\n" + "\n".join(image_labels)
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        image_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }
        for image in source_images:
            image_path = self.source_image_file(conversation_id, image["filename"])
            if not image_path.is_file():
                raise CoordinationError(f"source image file is missing: {image['filename']}")
            try:
                resolved_image_path = image_path.resolve(strict=True)
            except OSError as error:
                raise CoordinationError(
                    f"source image file could not be resolved: {image['filename']}"
                ) from error
            image_type = image_types.get(resolved_image_path.suffix.lower())
            if image_type is None:
                raise CoordinationError(
                    f"source image format is unsupported by {backend_label}: {image['filename']}"
                )
            try:
                encoded_image = base64.b64encode(resolved_image_path.read_bytes()).decode("ascii")
            except OSError as error:
                raise CoordinationError(
                    f"source image file could not be read: {image['filename']}"
                ) from error
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{image_type};base64,{encoded_image}"},
            })
        return content_parts
