from __future__ import annotations

import base64
import hashlib
import re
import threading
import time
import json
import os
import secrets
import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from .chat_backends import CHAT_BACKENDS, get_chat_backend
from .clients import BackendError, ChatBackend, ComfyUIClient, NativeSessionBackend
from .modes import (
    DEFAULT_MODE_ID,
    MAX_SOURCE_IMAGES,
    calculate_mode_dimensions,
    get_mode,
    source_image_role,
    validate_source_image_count,
)
from .permissions import set_private_permissions
from .settings import Settings
from .services import ServiceError, service_running, start_service_tracked, stop_service
from .resolution import normalize_aspect_ratio
from .storage import ConversationStore, StorageError
from .workflow import WorkflowError, build_workflow


class CoordinationError(RuntimeError):
    """The requested model handoff is unsafe or cannot complete."""


@dataclass(frozen=True)
class ChatOutcome:
    answer: str
    image_prompt: str
    aspect_ratio: str
    context_saved: bool
    warning: str | None = None
    reasoning_content: str = ""


@dataclass(frozen=True)
class GenerationOutcome:
    generation_id: str
    prompt_id: str
    image: dict[str, str]
    cleanup_warning: str | None = None


class Orchestrator:
    def __init__(self, settings: Settings, store: ConversationStore,
                 chat_backend: ChatBackend, comfy: ComfyUIClient,
                 clock: Callable[[], float] = time.monotonic,
                 sleeper: Callable[[float], None] = time.sleep):
        self.settings = settings
        self.store = store
        self.chat_backend = chat_backend
        self.comfy = comfy
        self._clock = clock
        self._sleep = sleeper
        self._operation_lock = threading.Lock()
        self._comfy_models_retained = False

    def _enter(self) -> None:
        if not self._operation_lock.acquire(blocking=False):
            raise CoordinationError("another model operation is already in progress")

    def _leave(self) -> None:
        self._operation_lock.release()

    def _queue(self) -> tuple[list[Any], list[Any]]:
        response = self.comfy.queue()
        running = response.get("queue_running")
        pending = response.get("queue_pending")
        if not isinstance(running, list) or not isinstance(pending, list):
            raise CoordinationError("ComfyUI queue state is unknown; refusing GPU handoff")
        return running, pending

    def _require_idle_queue(self) -> None:
        running, pending = self._queue()
        if running or pending:
            raise CoordinationError("ComfyUI queue is busy with other work; no model switch was made")

    def _release_comfy(self) -> None:
        self._require_idle_queue()
        self._comfy_models_retained = False
        self.comfy.free_memory()
        self._require_idle_queue()

    def _chat_backend_loaded(self) -> bool:
        backend = get_chat_backend(self.settings.chat_backend)
        if backend.loaded_state == "process":
            return service_running(self.settings, self.settings.chat_backend)
        if backend.loaded_state == "stateless":
            return False
        response = self.chat_backend.status()
        loaded = response.get("loaded")
        if not isinstance(loaded, bool):
            raise CoordinationError("chat backend load state is unknown; refusing GPU handoff")
        return loaded

    def _session_path(self, conversation_id: str) -> Path:
        directory = self.store.session_directory(conversation_id)
        if directory is None:
            raise CoordinationError("session directory is missing")
        return directory / "chat_context.bin"

    def _native_session_backend(self) -> NativeSessionBackend:
        if not isinstance(self.chat_backend, NativeSessionBackend):
            raise CoordinationError(
                f"the selected {get_chat_backend(self.settings.chat_backend).label} client "
                "does not support native sessions"
            )
        return self.chat_backend

    def _session_asset_directory(self, conversation_id: str) -> Path:
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

    def _conversation_input_directory(self, conversation_id: str) -> Path:
        return self._session_asset_directory(conversation_id)

    def _studio_output_directory(self, conversation_id: str) -> Path:
        return self._session_asset_directory(conversation_id)

    def _relocate_generated_image(self, conversation_id: str,
                                  image: dict[str, str]) -> dict[str, str]:
        conversation_id = self.store.validate_id(conversation_id)
        filename = image.get("filename", "")
        if (image.get("subfolder") != conversation_id or image.get("type") != "output"
                or not filename or Path(filename).name != filename or "\\" in filename
                or Path(filename).suffix.lower() != ".png"):
            raise CoordinationError("ComfyUI returned an unsafe generated image path")

        source_directory = self._session_asset_directory(conversation_id)
        source_path = source_directory / filename
        if source_path.is_symlink():
            raise CoordinationError("refusing to move a symbolic-link generated image")
        source_path = source_path.resolve(strict=True)
        if source_path.parent != source_directory or not source_path.is_file():
            raise CoordinationError("ComfyUI generated image is missing or unsafe")

        content_hash = self._sha256_file(source_path)
        destination_path = source_directory / f"{content_hash}.png"
        if destination_path.is_symlink():
            raise CoordinationError("refusing to overwrite a symbolic-link generated image")
        if destination_path.exists():
            if (not destination_path.is_file()
                    or self._sha256_file(destination_path) != content_hash):
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
                        or self._sha256_file(destination_path) != content_hash):
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
        self._archive_generated_image(destination_path)
        return {"filename": destination_path.name, "subfolder": conversation_id, "type": "output"}

    def _archive_generated_image(self, image_path: Path) -> Path:
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
                    time.sleep(max(0.01, 1.0 - (time.time() % 1.0)))
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
                    time.sleep(max(0.01, 1.0 - (time.time() % 1.0)))
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
        directory = self._studio_output_directory(conversation_id)
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

    def source_image_path(self, conversation_id: str,
                          source_image: dict[str, Any]) -> Path:
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
            directory = self._conversation_input_directory(conversation_id)
            if directory.is_symlink():
                raise CoordinationError("refusing to access a symbolic-link source image directory")
            image_path = self._source_image_file(conversation_id, filename)
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

    def _source_image_file(self, conversation_id: str, filename: str) -> Path:
        directory = self._conversation_input_directory(conversation_id)
        if filename in {"", ".", ".."} or Path(filename).name != filename or "\\" in filename:
            raise CoordinationError("refusing an unsafe source image filename")
        path = directory / filename
        if path.is_symlink():
            raise CoordinationError("refusing to access a symbolic-link source image")
        return path

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        try:
            with path.open("rb") as image_file:
                for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as error:
            raise CoordinationError(f"could not read image for hashing: {error}") from error
        return digest.hexdigest()

    def _ensure_source_image_file(self, conversation_id: str, content: bytes,
                                  content_type: str, extension: str,
                                  content_hash: str) -> tuple[Path, bool]:
        filename = f"{content_hash}{extension}"
        image_path = self._source_image_file(conversation_id, filename)
        if image_path.exists():
            if not image_path.is_file() or self._sha256_file(image_path) != content_hash:
                raise CoordinationError("source image hash path contains different content")
            return image_path, False

        response = self.comfy.upload_image(filename, content, content_type, conversation_id)
        uploaded_name = response.get("name")
        uploaded_subfolder = response.get("subfolder")
        uploaded_type = response.get("type")
        if (uploaded_name != filename or uploaded_subfolder != conversation_id
                or uploaded_type != "input"):
            raise CoordinationError("ComfyUI returned an invalid source image path")
        image_path = self._source_image_file(conversation_id, filename)
        if not image_path.is_file() or self._sha256_file(image_path) != content_hash:
            raise CoordinationError("ComfyUI did not store the source image at its hash path")
        return image_path, True

    def upload_source_image(self, conversation_id: str, content: bytes, content_type: str,
                            extension: str, width: int, height: int) -> dict[str, Any]:
        conversation_id = self.store.validate_id(conversation_id)
        if self.store.get_conversation(conversation_id) is None:
            raise StorageError("conversation does not exist")
        if extension not in {".png", ".jpg", ".webp"}:
            raise CoordinationError("unsupported source image extension")
        content_hash = hashlib.sha256(content).hexdigest()
        self._enter()
        try:
            existing = self.store.source_image_by_hash(conversation_id, content_hash)
            if existing is not None:
                if (existing["removed_at"] is not None
                        and len(self.store.source_images(conversation_id)) >= MAX_SOURCE_IMAGES):
                    raise CoordinationError(
                        f"a conversation can have at most {MAX_SOURCE_IMAGES} source images"
                    )
                self._ensure_source_image_file(
                    conversation_id, content, content_type, extension, content_hash,
                )
                if existing["removed_at"] is None:
                    return existing
                restored = self.store.restore_source_image(conversation_id, existing["id"])
                if restored is None:
                    raise StorageError("source image could not be restored")
                return restored

            if len(self.store.source_images(conversation_id)) >= MAX_SOURCE_IMAGES:
                raise CoordinationError(
                    f"a conversation can have at most {MAX_SOURCE_IMAGES} source images"
                )
            source_image_id = secrets.token_hex(16)
            image_path, uploaded = self._ensure_source_image_file(
                conversation_id, content, content_type, extension, content_hash,
            )
            try:
                return self.store.create_source_image(
                    source_image_id, conversation_id, image_path.name, conversation_id,
                    content_type, content_hash, len(content), width, height,
                )
            except (StorageError, sqlite3.Error) as error:
                if uploaded:
                    try:
                        image_path.unlink(missing_ok=True)
                    except OSError as cleanup_error:
                        raise CoordinationError(
                            f"source image could not be recorded or removed: {cleanup_error}"
                        ) from cleanup_error
                raise
        finally:
            self._leave()

    def reorder_source_images(self, conversation_id: str,
                              source_image_ids: list[str]) -> list[dict[str, Any]]:
        self._enter()
        try:
            return self.store.reorder_source_images(conversation_id, source_image_ids)
        finally:
            self._leave()

    def set_conversation_mode(self, conversation_id: str, image_mode: str) -> None:
        self._enter()
        try:
            self.store.set_conversation_mode(conversation_id, image_mode)
        finally:
            self._leave()

    def delete_source_image(self, conversation_id: str, source_image_id: str) -> None:
        conversation_id = self.store.validate_id(conversation_id)
        self._enter()
        try:
            if self.store.soft_delete_source_image(conversation_id, source_image_id) is None:
                raise StorageError("source image does not exist in this conversation")
        finally:
            self._leave()

    def delete_conversation(self, conversation_id: str) -> str | None:
        conversation_id = self.store.validate_id(conversation_id)
        if self.store.get_conversation(conversation_id) is None:
            raise StorageError("conversation does not exist")
        self._enter()
        try:
            if self.store.active_conversation_id() == conversation_id:
                backend = get_chat_backend(self.settings.chat_backend)
                if backend.context_strategy == "native-session" and self._chat_backend_loaded():
                    self._native_session_backend().unload(wait_s=300)
                    if self._chat_backend_loaded():
                        raise CoordinationError(
                            f"{backend.label} did not unload; the conversation was not deleted"
                        )
                self.store.set_active_conversation_id(None)
            try:
                deleted = self.store.delete_conversation(conversation_id)
            except StorageError as error:
                raise CoordinationError(
                    f"could not remove the conversation session: {error}"
                ) from error
            if not deleted:
                raise StorageError("conversation does not exist")
            return None
        finally:
            self._leave()

    def _save_session(self, conversation_id: str) -> None:
        backend = get_chat_backend(self.settings.chat_backend)
        filename = self.store.session_filename(conversation_id)
        staging_path = self.settings.path("paths.runtime_root") / filename
        if staging_path.is_symlink() or staging_path.exists():
            raise CoordinationError("a native session staging file already exists")
        try:
            response = self._native_session_backend().save_session(filename)
            token_count = response.get("n_saved")
            byte_count = response.get("n_written")
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                   for value in (token_count, byte_count)):
                raise CoordinationError(f"{backend.label} did not confirm the session save")
            if staging_path.is_symlink() or not staging_path.is_file():
                raise CoordinationError(f"{backend.label} did not write the native session file")
            session_path = self._session_path(conversation_id)
            if session_path.is_symlink():
                raise CoordinationError("refusing to replace a symbolic-link native session")
            try:
                os.replace(staging_path, session_path)
            except OSError as error:
                raise CoordinationError(
                    f"could not move the native session into its session directory: {error}"
                ) from error
            self.store.mark_context_saved(
                conversation_id, token_count, byte_count, backend=backend.id
            )
        finally:
            try:
                staging_path.unlink(missing_ok=True)
            except OSError as error:
                raise CoordinationError(f"could not clean up native session staging file: {error}") from error

    def _restore_session(self, conversation_id: str) -> None:
        conversation = self.store.get_conversation(conversation_id)
        if conversation is None:
            raise CoordinationError("conversation does not exist")
        if (conversation.get("context_saved_at")
                and conversation.get("context_backend") != self.settings.chat_backend):
            return
        if not conversation.get("context_saved_at"):
            if self.store.messages(conversation_id):
                raise CoordinationError("conversation has messages but no native session file")
            return
        if not self.store.context_is_current(conversation_id, self.settings.chat_backend):
            raise CoordinationError("saved native context does not match the latest transcript")
        session_path = self._session_path(conversation_id)
        if session_path.is_symlink() or not session_path.is_file():
            raise CoordinationError("saved native session file is missing")
        backend = get_chat_backend(self.settings.chat_backend)
        filename = self.store.session_filename(conversation_id)
        staging_path = self.settings.path("paths.runtime_root") / filename
        if staging_path.is_symlink() or staging_path.exists():
            raise CoordinationError("a native session staging file already exists")
        try:
            with session_path.open("rb") as source, staging_path.open("xb") as target:
                shutil.copyfileobj(source, target)
                target.flush()
                os.fsync(target.fileno())
            set_private_permissions(staging_path)
            response = self._native_session_backend().restore_session(filename)
            restored = response.get("n_restored")
            if isinstance(restored, bool) or not isinstance(restored, int) or restored <= 0:
                raise CoordinationError(f"{backend.label} did not confirm the session restore")
        except OSError as error:
            raise CoordinationError(f"could not stage the native session for restore: {error}") from error
        finally:
            try:
                staging_path.unlink(missing_ok=True)
            except OSError as error:
                raise CoordinationError(f"could not clean up native session staging file: {error}") from error

    def _save_context(self, conversation_id: str) -> None:
        backend = get_chat_backend(self.settings.chat_backend)
        if backend.context_strategy == "native-session":
            self._save_session(conversation_id)
            return
        transcript = json.dumps(
            self.store.messages(conversation_id),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.store.mark_context_saved(
            conversation_id,
            None,
            len(transcript),
            backend=backend.id,
        )

    def _wait_for_managed_backend(self, log_path: Path) -> None:
        backend = get_chat_backend(self.settings.chat_backend)
        if backend.startup_timeout_key is None:
            raise CoordinationError(f"{backend.label} has no configured service startup timeout")
        timeout = self.settings.backend_config()[backend.startup_timeout_key]
        deadline = self._clock() + timeout
        while True:
            try:
                status = self.chat_backend.status()
                if status.get("loaded") is True:
                    return
                last_error = "health endpoint did not report a ready model"
            except BackendError as error:
                last_error = str(error)
            if not service_running(self.settings, backend.id):
                raise CoordinationError(
                    f"{backend.label} exited before becoming ready ({last_error}); inspect {log_path}"
                )
            if self._clock() >= deadline:
                raise CoordinationError(
                    f"{backend.label} did not become ready within {timeout:g} seconds "
                    f"({last_error}); inspect {log_path}"
                )
            self._sleep(min(0.5, max(0.0, deadline - self._clock())))

    def _prepare_on_demand_backend(self) -> None:
        backend = get_chat_backend(self.settings.chat_backend)
        if not service_running(self.settings, backend.id):
            self._release_comfy()
            try:
                pid, log_path, _ = start_service_tracked(self.settings, backend.id)
            except (ServiceError, OSError) as error:
                raise CoordinationError(f"could not start {backend.label}: {error}") from error
            if pid is None:
                raise CoordinationError(f"{backend.label} has no identifiable process to supervise")
        else:
            log_path = self.settings.path("paths.runtime_root") / f"service_{backend.id}.log"
        self._wait_for_managed_backend(log_path)

    def _stop_backend_service(self) -> None:
        backend = get_chat_backend(self.settings.chat_backend)
        try:
            stopped = stop_service(self.settings, backend.id, wait_s=30.0)
        except (ServiceError, OSError) as error:
            raise CoordinationError(
                f"could not stop {backend.label} before ComfyUI generation: {error}"
            ) from error
        if stopped is None or service_running(self.settings, backend.id):
            raise CoordinationError(
                f"{backend.label} is still running; image generation was not started"
            )

    def _prepare_chat_backend(self, conversation_id: str) -> None:
        self._require_idle_queue()
        backend = get_chat_backend(self.settings.chat_backend)
        if backend.chat_preparation == "on-demand-service":
            self._prepare_on_demand_backend()
            self.store.set_active_conversation_id(conversation_id)
            return
        if backend.chat_preparation == "stateless":
            self.store.set_active_conversation_id(conversation_id)
            return

        loaded = self._chat_backend_loaded()
        active_id = self.store.active_conversation_id()
        if loaded and active_id is None:
            raise CoordinationError(f"{backend.label} is loaded but its active conversation is unknown")
        if loaded and active_id == conversation_id:
            return
        if loaded and active_id != conversation_id:
            if active_id is not None:
                self._save_session(active_id)
            self._native_session_backend().unload(wait_s=300)
            if self._chat_backend_loaded():
                raise CoordinationError(f"{backend.label} did not unload; refusing to replace its context")
            loaded = False
        if not loaded:
            self._release_comfy()
            self._native_session_backend().load()
            if not self._chat_backend_loaded():
                raise CoordinationError(f"{backend.label} load did not complete")
            self._restore_session(conversation_id)
            self.store.set_active_conversation_id(conversation_id)

    @staticmethod
    def _chat_content(response: dict[str, Any]) -> tuple[str, str]:
        try:
            message = response["choices"][0]["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise CoordinationError("chat backend returned no assistant message") from error
        if isinstance(content, str):
            answer = content
        elif isinstance(content, list):
            answer = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        else:
            raise CoordinationError("chat backend returned an unsupported message content type")
        if not answer.strip():
            raise CoordinationError("chat backend returned an empty assistant message")
        reasoning = message.get("reasoning_content") or ""
        if not isinstance(reasoning, str):
            raise CoordinationError("chat backend returned unsupported reasoning content")
        return answer, reasoning

    @classmethod
    def _parse_chat_response(cls, response: dict[str, Any]) -> tuple[str, str, str, str, str]:
        raw, reasoning = cls._chat_content(response)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            raise CoordinationError("chat backend response was not the configured JSON object") from error
        if not isinstance(parsed, dict) or set(parsed) != {"reply", "image_prompt", "aspect_ratio"}:
            raise CoordinationError("chat backend response must contain only reply, image_prompt, and aspect_ratio")
        reply, image_prompt, aspect_ratio = parsed["reply"], parsed["image_prompt"], parsed["aspect_ratio"]
        if (not isinstance(reply, str) or not reply.strip() or not isinstance(image_prompt, str)
                or not isinstance(aspect_ratio, str)):
            raise CoordinationError("chat backend reply, image_prompt, and aspect_ratio must be strings")
        return reply.strip(), image_prompt.strip(), aspect_ratio.strip(), raw, reasoning

    def _build_chat_messages(self, conversation_id: str, user_text: str,
                             current_image_prompt: str | None,
                             current_aspect_ratio: str | None
                             ) -> tuple[str, str, list[dict[str, Any]], str | None]:
        conversation_id = self.store.validate_id(conversation_id)
        conversation = self.store.get_conversation(conversation_id)
        if conversation is None:
            raise CoordinationError("conversation does not exist")
        if not isinstance(user_text, str) or not user_text.strip() or len(user_text) > 20000:
            raise CoordinationError("message must contain 1 to 20000 characters")
        if current_image_prompt is None:
            current_image_prompt = conversation["image_prompt"]
        if not isinstance(current_image_prompt, str) or len(current_image_prompt) > 20000:
            raise CoordinationError("current image prompt must contain at most 20000 characters")
        generation = self.settings.data["generation"]
        if current_aspect_ratio is None or not current_aspect_ratio.strip():
            current_aspect_ratio = conversation["aspect_ratio"] or generation["default_aspect_ratio"]
        current_aspect_ratio = current_aspect_ratio.strip()
        available_ratios = {item["label"] for item in generation["aspect_ratios"]}
        if current_aspect_ratio not in available_ratios:
            raise CoordinationError("current aspect ratio is not supported")
        image_mode = conversation.get("image_mode", DEFAULT_MODE_ID)
        try:
            mode = get_mode(image_mode)
        except ValueError as error:
            raise CoordinationError(str(error)) from error
        source_images = (
            self.store.source_images(conversation_id)
            if mode.send_source_images_to_chat else []
        )
        source_image_count = len(source_images)
        system_prompt_path = self.settings.path(
            f"generation_modes.{mode.id}.system_prompt_file"
        )
        try:
            system_prompt_text = system_prompt_path.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise CoordinationError(f"cannot read system prompt file: {system_prompt_path}") from error
        if not system_prompt_text:
            raise CoordinationError(f"system prompt file is empty: {system_prompt_path}")
        system_prompt = system_prompt_text + (
            "\n最新用户消息开头包含 current_editor_state JSON，其中 current_image_prompt 是右侧当前草稿，"
            "current_aspect_ratio 是当前画幅选择。除非用户本轮明确要求更改，否则延续草稿并保留该画幅。"
        )
        editor_state = {
            "current_image_prompt": current_image_prompt,
            "current_aspect_ratio": current_aspect_ratio,
        }
        preserve_aspect_ratio = None
        if mode.send_source_images_to_chat:
            editor_state.update({"image_mode": mode.id, "source_image_count": source_image_count})
        if mode.resolution_strategy == "source-image":
            preserve_aspect_ratio = current_aspect_ratio
        current_editor_state = json.dumps(editor_state, ensure_ascii=False)
        current_user_message = (
            f"current_editor_state:\n{current_editor_state}\n\n"
            f"user_request:\n{user_text.strip()}"
        )
        message_content: str | list[dict[str, Any]] = current_user_message
        if mode.send_source_images_to_chat and source_images:
            backend = get_chat_backend(self.settings.chat_backend)
            if backend.image_transport == "none":
                raise CoordinationError(
                    f"{backend.label} does not accept source images for chat"
                )
            image_labels = [
                f"<Picture {index + 1}> ({source_image_role(mode, index)})"
                for index in range(source_image_count)
            ]
            current_user_message += "\n\nattached_source_images_in_order:\n" + "\n".join(image_labels)
            content_parts: list[dict[str, Any]] = [{"type": "text", "text": current_user_message}]
            for image in source_images:
                image_path = self._source_image_file(conversation_id, image["filename"])
                if not image_path.is_file():
                    raise CoordinationError(f"source image file is missing: {image['filename']}")
                try:
                    resolved_image_path = image_path.resolve(strict=True)
                except OSError as error:
                    raise CoordinationError(
                        f"source image file could not be resolved: {image['filename']}"
                    ) from error
                image_types = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".webp": "image/webp",
                }
                image_type = image_types.get(resolved_image_path.suffix.lower())
                if image_type is None:
                    raise CoordinationError(
                        f"source image format is unsupported by {backend.label}: "
                        f"{image['filename']}"
                    )
                try:
                    encoded_image = base64.b64encode(resolved_image_path.read_bytes()).decode("ascii")
                except OSError as error:
                    raise CoordinationError(
                        f"source image file could not be read: {image['filename']}"
                    ) from error
                image_url = f"data:{image_type};base64,{encoded_image}"
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": image_url},
                })
            message_content = content_parts
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.store.messages(conversation_id))
        messages.append({"role": "user", "content": message_content})
        return conversation_id, user_text.strip(), messages, preserve_aspect_ratio

    def _finish_chat(self, conversation_id: str, user_text: str,
                     response: dict[str, Any], preserve_aspect_ratio: str | None = None) -> ChatOutcome:
        answer, image_prompt, suggested_ratio, raw_answer, reasoning_content = self._parse_chat_response(response)
        if preserve_aspect_ratio is not None:
            aspect_ratio, was_adjusted = preserve_aspect_ratio, False
        else:
            try:
                aspect_ratio, was_adjusted = normalize_aspect_ratio(
                    suggested_ratio, self.settings.data["generation"]["aspect_ratios"]
                )
            except ValueError as error:
                raise CoordinationError(str(error)) from error
        display_answer = answer
        ratio_short = aspect_ratio.split(" ", 1)[0]
        explained_adjustment = suggested_ratio in answer and ratio_short in answer
        if was_adjusted and not explained_adjustment:
            display_answer += f"（{suggested_ratio} 不在原生选项中，已采用最接近的 {aspect_ratio}。）"
        self.store.append_exchange(conversation_id, user_text, raw_answer, image_prompt,
                                   aspect_ratio, display_answer, reasoning_content)
        try:
            self._save_context(conversation_id)
            return ChatOutcome(answer=display_answer, image_prompt=image_prompt, aspect_ratio=aspect_ratio,
                               context_saved=True, reasoning_content=reasoning_content)
        except (BackendError, StorageError, CoordinationError) as error:
            return ChatOutcome(answer=display_answer, image_prompt=image_prompt, aspect_ratio=aspect_ratio,
                               context_saved=False,
                               warning=f"Reply received, but the conversation context was not saved: {error}",
                               reasoning_content=reasoning_content)

    def chat(self, conversation_id: str, user_text: str, max_tokens: int = 8192, *,
             current_image_prompt: str | None = None,
             current_aspect_ratio: str | None = None,
             reasoning_effort: str | None = None) -> ChatOutcome:
        conversation_id, user_text, messages, preserve_aspect_ratio = self._build_chat_messages(
            conversation_id, user_text, current_image_prompt, current_aspect_ratio
        )
        self._enter()
        try:
            self._prepare_chat_backend(conversation_id)
            response = self.chat_backend.chat(messages, max_tokens=max_tokens,
                                              reasoning_effort=reasoning_effort)
            return self._finish_chat(conversation_id, user_text, response, preserve_aspect_ratio)
        finally:
            self._leave()

    def chat_stream(self, conversation_id: str, user_text: str, max_tokens: int = 8192, *,
                    current_image_prompt: str | None = None,
                    current_aspect_ratio: str | None = None,
                    reasoning_effort: str | None = None) -> Iterator[dict[str, Any]]:
        conversation_id, user_text, messages, preserve_aspect_ratio = self._build_chat_messages(
            conversation_id, user_text, current_image_prompt, current_aspect_ratio
        )
        self._enter()
        stream = None
        try:
            self._prepare_chat_backend(conversation_id)
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            stream = self.chat_backend.chat_stream(messages, max_tokens=max_tokens,
                                                   reasoning_effort=reasoning_effort)
            try:
                for delta in stream:
                    content = delta.get("content", "")
                    reasoning = delta.get("reasoning_content", "")
                    if not isinstance(content, str) or not isinstance(reasoning, str):
                        raise CoordinationError("chat backend returned non-text streaming content")
                    if content:
                        content_parts.append(content)
                    if reasoning:
                        reasoning_parts.append(reasoning)
                        yield {"type": "reasoning", "text": reasoning}
            finally:
                close_stream = getattr(stream, "close", None)
                if close_stream is not None:
                    close_stream()
            response = {"choices": [{"message": {
                "content": "".join(content_parts),
                "reasoning_content": "".join(reasoning_parts),
            }}]}
            outcome = self._finish_chat(conversation_id, user_text, response, preserve_aspect_ratio)
            yield {
                "type": "done",
                "answer": outcome.answer,
                "image_prompt": outcome.image_prompt,
                "aspect_ratio": outcome.aspect_ratio,
                "reasoning_content": outcome.reasoning_content,
                "context_saved": outcome.context_saved,
                "warning": outcome.warning,
            }
        finally:
            self._leave()

    def _report_generation_event(self, callback: Callable[[dict[str, Any]], None] | None,
                                event: dict[str, Any]) -> None:
        if callback is not None:
            callback(event)

    @staticmethod
    def _generation_node_label(workflow: dict[str, Any], node_id: str,
                               event_data: dict[str, Any]) -> str:
        node = workflow.get(node_id, {})
        metadata = node.get("_meta", {}) if isinstance(node, dict) else {}
        title = metadata.get("title") if isinstance(metadata, dict) else None
        return str(title or (node.get("class_type") if isinstance(node, dict) else None)
                   or event_data.get("display_node") or f"Node {node_id}")

    def _read_comfy_progress(self, progress_socket: Any, prompt_id: str,
                             workflow: dict[str, Any],
                             callback: Callable[[dict[str, Any]], None] | None,
                             completed_nodes: set[str],
                             active_node: str | None) -> tuple[bool, str | None]:
        if progress_socket is None:
            return False, active_node
        total_nodes = len(workflow)
        sampler_node_id = next((node_id for node_id, node in workflow.items()
                                 if isinstance(node, dict) and node.get("class_type") == "KSampler"), None)
        sampler_node = workflow.get(sampler_node_id) if sampler_node_id is not None else None
        sampler_inputs = sampler_node.get("inputs", {}) if isinstance(sampler_node, dict) else {}
        configured_steps = sampler_inputs.get("steps") if isinstance(sampler_inputs, dict) else None
        sampler_steps = (configured_steps if isinstance(configured_steps, int)
                         and not isinstance(configured_steps, bool) and configured_steps > 0
                         else 1 if sampler_node_id is not None else 0)
        total_units = total_nodes - (1 if sampler_node_id is not None else 0) + sampler_steps

        def completed_work_units() -> int:
            sampler_completed = sampler_node_id is not None and sampler_node_id in completed_nodes
            return len(completed_nodes) - int(sampler_completed) + (sampler_steps if sampler_completed else 0)

        def overall_percent(work_units: float) -> float:
            return min(99.0, work_units / total_units * 100) if total_units else 0.0

        while True:
            try:
                raw = progress_socket.recv(timeout=0)
            except TimeoutError:
                return True, active_node
            except Exception:
                self._report_generation_event(callback, {
                    "type": "phase",
                    "message": "Live progress connection lost; waiting for ComfyUI completion",
                })
                return False, active_node
            if not isinstance(raw, str):
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            data = event.get("data")
            if not isinstance(data, dict) or data.get("prompt_id") != prompt_id:
                continue
            node_id = data.get("node")
            if not isinstance(node_id, str):
                continue
            label = self._generation_node_label(workflow, node_id, data)
            if event_type == "progress":
                value, maximum = data.get("value"), data.get("max")
                if (isinstance(value, (int, float)) and not isinstance(value, bool)
                        and isinstance(maximum, (int, float)) and not isinstance(maximum, bool)
                        and maximum > 0):
                    step_progress = min(1.0, max(0.0, value / maximum))
                    active_units = 0.0
                    if node_id not in completed_nodes:
                        active_units = step_progress * (sampler_steps if node_id == sampler_node_id else 1)
                    completed_units = completed_work_units()
                    self._report_generation_event(callback, {
                        "type": "progress", "node": node_id, "label": label,
                        "value": value, "max": maximum,
                        "completed_nodes": len(completed_nodes), "total_nodes": total_nodes,
                        "completed_units": completed_units, "total_units": total_units,
                        "overall_percent": overall_percent(completed_units + active_units),
                        "message": f"{label}: {value:g}/{maximum:g}",
                    })
            elif event_type == "executing":
                if active_node is not None and active_node != node_id and active_node in workflow:
                    completed_nodes.add(active_node)
                active_node = node_id
                completed_units = completed_work_units()
                self._report_generation_event(callback, {
                    "type": "node", "state": "started", "node": node_id,
                    "label": label, "completed_nodes": len(completed_nodes),
                    "total_nodes": total_nodes,
                    "completed_units": completed_units, "total_units": total_units,
                    "overall_percent": overall_percent(completed_units),
                    "message": f"Running {label}",
                })
            elif event_type == "executed":
                if node_id in workflow:
                    completed_nodes.add(node_id)
                if active_node == node_id:
                    active_node = None
                completed_units = completed_work_units()
                self._report_generation_event(callback, {
                    "type": "node", "state": "complete", "node": node_id,
                    "label": label, "completed_nodes": len(completed_nodes),
                    "total_nodes": total_nodes,
                    "completed_units": completed_units, "total_units": total_units,
                    "overall_percent": overall_percent(completed_units),
                    "message": f"Completed {label}",
                })
        
    def _wait_for_generation(self, prompt_id: str, workflow: dict[str, Any],
                             progress_socket: Any = None,
                             progress_callback: Callable[[dict[str, Any]], None] | None = None) -> dict[str, str]:
        coordination = self.settings.section("coordination")
        deadline = self._clock() + coordination["generation_timeout_s"]
        completed_nodes: set[str] = set()
        active_node: str | None = None
        while self._clock() < deadline:
            if progress_socket is not None:
                connected, active_node = self._read_comfy_progress(
                    progress_socket, prompt_id, workflow, progress_callback,
                    completed_nodes, active_node,
                )
                if not connected:
                    progress_socket = None
            response = self.comfy.history(prompt_id)
            entry = response.get(prompt_id)
            if isinstance(entry, dict):
                status = entry.get("status", {})
                if isinstance(status, dict) and status.get("status_str") == "error":
                    raise CoordinationError("ComfyUI reported a generation error")
                outputs = entry.get("outputs", {})
                for output in outputs.values() if isinstance(outputs, dict) else ():
                    images = output.get("images", []) if isinstance(output, dict) else []
                    if images:
                        image = images[0]
                        if all(isinstance(image.get(key), str) for key in ("filename", "subfolder", "type")):
                            return {key: image[key] for key in ("filename", "subfolder", "type")}
                if isinstance(status, dict) and status.get("status_str") == "success":
                    raise CoordinationError("ComfyUI finished without a saved image")
            self._sleep(coordination["poll_interval_s"])
        raise CoordinationError("ComfyUI generation timed out")

    def generate(self, conversation_id: str, prompt: str, seed: int,
                 aspect_ratio: str | None = None, megapixels: float | None = None,
                 sampler_name: str | None = None, scheduler: str | None = None,
                 progress_callback: Callable[[dict[str, Any]], None] | None = None,
                 image_mode: str = DEFAULT_MODE_ID,
                 source_image_ids: list[str] | None = None) -> GenerationOutcome:
        conversation_id = self.store.validate_id(conversation_id)
        try:
            mode = get_mode(image_mode)
        except ValueError as error:
            raise CoordinationError(str(error)) from error
        self._enter()
        generation_id: str | None = None
        prompt_id: str | None = None
        completed_image: dict[str, str] | None = None
        failure: Exception | None = None
        progress_socket = None
        selected_source_images: list[dict[str, Any]] = []
        width = 0
        height = 0
        try:
            conversation = self.store.get_conversation(conversation_id)
            if conversation is None:
                raise CoordinationError("conversation does not exist")
            if conversation.get("image_mode", DEFAULT_MODE_ID) != image_mode:
                raise CoordinationError("image mode changed; reload the conversation before generating")
            selected_source_ids = source_image_ids or []
            try:
                validate_source_image_count(mode, len(selected_source_ids))
                selected_source_images = self.store.source_images_by_ids(
                    conversation_id, selected_source_ids,
                )
                width, height = calculate_mode_dimensions(
                    mode,
                    self.settings.section("generation"),
                    aspect_ratio,
                    megapixels,
                    selected_source_images,
                )
            except (StorageError, ValueError) as error:
                raise CoordinationError(str(error)) from error
            selected_ratio = (
                mode.stored_aspect_ratio_label
                or aspect_ratio
                or self.settings.section("generation")["default_aspect_ratio"]
            )
            try:
                workflow = build_workflow(
                    self.settings, prompt, seed, aspect_ratio, megapixels,
                    sampler_name, scheduler, image_mode, selected_source_images or None,
                )
            except WorkflowError as error:
                raise CoordinationError(str(error)) from error
            save_node = next((node for node in workflow.values()
                              if isinstance(node, dict) and node.get("class_type") == "SaveImage"), None)
            if save_node is None or not isinstance(save_node.get("inputs"), dict):
                raise CoordinationError("workflow must include a valid SaveImage node")
            save_node["inputs"]["filename_prefix"] = f"{conversation_id}/image"

            self._report_generation_event(progress_callback, {
                "type": "phase", "message": "Preparing GPU resources for ComfyUI",
            })
            self._require_idle_queue()
            backend = get_chat_backend(self.settings.chat_backend)
            backend_was_loaded = self._chat_backend_loaded()
            if backend.generation_handoff == "native-session" and backend_was_loaded:
                active_id = self.store.active_conversation_id()
                if active_id is None:
                    raise CoordinationError(
                        f"{backend.label} is loaded but its active conversation is unknown"
                    )
                self._save_session(active_id)
                self._native_session_backend().unload(wait_s=300)
                if self._chat_backend_loaded():
                    raise CoordinationError(
                        f"{backend.label} did not unload; image generation was not started"
                    )
            elif backend.generation_handoff == "stop-service" and backend_was_loaded:
                self._stop_backend_service()
            if (
                backend.context_strategy == "native-session"
                and self.store.messages(conversation_id)
                and not any(
                    self.store.context_is_current(conversation_id, backend_id)
                    for backend_id in CHAT_BACKENDS
                )
            ):
                raise CoordinationError("this conversation has no current saved chat context")
            if backend_was_loaded or not self._comfy_models_retained:
                self._release_comfy()
            generation = self.settings.section("generation")
            selected_megapixels = (
                min(generation["default_megapixels"], mode.max_megapixels)
                if megapixels is None else megapixels
            )
            selected_sampler = sampler_name or generation["default_sampler"]
            selected_scheduler = scheduler or generation["default_scheduler"]
            generation_id = self.store.create_generation(
                conversation_id, prompt.strip(), seed, selected_ratio, selected_megapixels,
                width, height, selected_sampler, selected_scheduler,
                image_mode, [image["id"] for image in selected_source_images],
            )
            client_id = uuid.uuid4().hex
            try:
                progress_socket = self.comfy.open_progress_socket(client_id)
                self._report_generation_event(progress_callback, {
                    "type": "phase", "message": "Connected to ComfyUI live progress",
                })
            except Exception:
                self._report_generation_event(progress_callback, {
                    "type": "phase", "message": "Live progress unavailable; tracking completion status",
                })
            self._report_generation_event(progress_callback, {
                "type": "phase", "message": "Submitting workflow to ComfyUI",
            })
            response = self.comfy.submit_prompt(workflow, client_id=client_id)
            prompt_id = response.get("prompt_id")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise CoordinationError("ComfyUI did not return a prompt id")
            self._comfy_models_retained = True
            self.store.update_generation(generation_id, "running", prompt_id=prompt_id)
            self._report_generation_event(progress_callback, {
                "type": "phase", "message": "ComfyUI is executing the workflow",
            })
            comfy_image = self._wait_for_generation(prompt_id, workflow, progress_socket,
                                                    progress_callback)
            self._report_generation_event(progress_callback, {
                "type": "phase", "message": "Saving generated image to the session and output archive",
            })
            completed_image = self._relocate_generated_image(
                conversation_id, comfy_image,
            )
            self.store.update_generation(generation_id, "complete", image=completed_image)
        except Exception as error:
            failure = error
            if generation_id is not None:
                try:
                    self.store.update_generation(generation_id, "failed", prompt_id=prompt_id,
                                                 error=str(error)[:1000])
                except StorageError:
                    pass
        self._report_generation_event(progress_callback, {
            "type": "phase", "message": "Keeping ComfyUI models loaded for the next generation",
        })
        if progress_socket is not None:
            try:
                progress_socket.close()
            except Exception:
                pass
        self._leave()
        if failure is not None:
            raise CoordinationError(str(failure)) from failure
        if generation_id is None or prompt_id is None or completed_image is None:
            raise CoordinationError("generation ended without a complete result")
        self._report_generation_event(progress_callback, {
            "type": "phase", "message": "Image ready",
        })
        return GenerationOutcome(generation_id, prompt_id, completed_image)