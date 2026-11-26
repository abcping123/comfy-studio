from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
import secrets
import shutil
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator

from .chat_backends import CHAT_BACKENDS, get_chat_backend
from .chat_runner import build_chat_messages, finish_chat
from .clients import BackendError, ChatBackend, ComfyUIClient, NativeSessionBackend
from .coordination import ChatOutcome, CoordinationError, GenerationOutcome
from .generation_runner import report_generation_event, wait_for_generation
from .modes import (
    DEFAULT_MODE_ID,
    MAX_SOURCE_IMAGES,
    calculate_mode_dimensions,
    get_mode,
    mode_default_megapixels,
    validate_source_image_count,
)
from .permissions import set_private_permissions
from .session_assets import SessionAssets
from .settings import Settings
from .services import ServiceError, service_running, start_service_tracked, stop_service
from .storage import ConversationStore, StorageError
from .workflow import WorkflowError, build_workflow


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
        self._assets = SessionAssets(settings, store, comfy, sleeper=sleeper)

    def _enter(self) -> None:
        if not self._operation_lock.acquire(blocking=False):
            raise CoordinationError("another model operation is already in progress")

    def _leave(self) -> None:
        self._operation_lock.release()

    @contextmanager
    def _operation_scope(self) -> Iterator[None]:
        self._enter()
        try:
            yield
        finally:
            self._leave()

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

    def _conversation_input_directory(self, conversation_id: str) -> Path:
        return self._assets.session_asset_directory(conversation_id)

    def generated_image_path(self, generation: dict[str, Any]) -> Path:
        return self._assets.generated_image_path(generation)

    def source_image_path(self, conversation_id: str,
                          source_image: dict[str, Any]) -> Path:
        return self._assets.source_image_path(conversation_id, source_image)

    def upload_source_image(self, conversation_id: str, content: bytes, content_type: str,
                            extension: str, width: int, height: int) -> dict[str, Any]:
        conversation_id = self.store.validate_id(conversation_id)
        if self.store.get_conversation(conversation_id) is None:
            raise StorageError("conversation does not exist")
        if extension not in {".png", ".jpg", ".webp"}:
            raise CoordinationError("unsupported source image extension")
        content_hash = hashlib.sha256(content).hexdigest()
        with self._operation_scope():
            existing = self.store.source_image_by_hash(conversation_id, content_hash)
            if existing is not None:
                if (existing["removed_at"] is not None
                        and len(self.store.source_images(conversation_id)) >= MAX_SOURCE_IMAGES):
                    raise CoordinationError(
                        f"a conversation can have at most {MAX_SOURCE_IMAGES} source images"
                    )
                self._assets.ensure_source_image_file(
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
            image_path, uploaded = self._assets.ensure_source_image_file(
                conversation_id, content, content_type, extension, content_hash,
            )
            try:
                return self.store.create_source_image(
                    source_image_id, conversation_id, image_path.name, conversation_id,
                    content_type, content_hash, len(content), width, height,
                )
            except (StorageError, sqlite3.Error):
                if uploaded:
                    try:
                        image_path.unlink(missing_ok=True)
                    except OSError as cleanup_error:
                        raise CoordinationError(
                            f"source image could not be recorded or removed: {cleanup_error}"
                        ) from cleanup_error
                raise

    def reorder_source_images(self, conversation_id: str,
                              source_image_ids: list[str]) -> list[dict[str, Any]]:
        with self._operation_scope():
            return self.store.reorder_source_images(conversation_id, source_image_ids)

    def set_conversation_mode(self, conversation_id: str, image_mode: str) -> None:
        with self._operation_scope():
            self.store.set_conversation_mode(conversation_id, image_mode)

    def delete_source_image(self, conversation_id: str, source_image_id: str) -> None:
        conversation_id = self.store.validate_id(conversation_id)
        with self._operation_scope():
            if self.store.soft_delete_source_image(conversation_id, source_image_id) is None:
                raise StorageError("source image does not exist in this conversation")

    def delete_conversation(self, conversation_id: str) -> str | None:
        conversation_id = self.store.validate_id(conversation_id)
        if self.store.get_conversation(conversation_id) is None:
            raise StorageError("conversation does not exist")
        with self._operation_scope():
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

    def _build_chat_messages(self, conversation_id: str, user_text: str,
                             current_image_prompt: str | None,
                             current_aspect_ratio: str | None
                             ) -> tuple[str, str, list[dict[str, Any]], str | None]:
        return build_chat_messages(
            self.settings,
            self.store,
            self._assets,
            conversation_id,
            user_text,
            current_image_prompt,
            current_aspect_ratio,
        )

    def _finish_chat(self, conversation_id: str, user_text: str,
                     response: dict[str, Any], preserve_aspect_ratio: str | None = None) -> ChatOutcome:
        return finish_chat(
            self.settings,
            self.store,
            self._save_context,
            conversation_id,
            user_text,
            response,
            preserve_aspect_ratio,
        )

    def chat(self, conversation_id: str, user_text: str, max_tokens: int = 8192, *,
             current_image_prompt: str | None = None,
             current_aspect_ratio: str | None = None,
             reasoning_effort: str | None = None) -> ChatOutcome:
        conversation_id, user_text, messages, preserve_aspect_ratio = self._build_chat_messages(
            conversation_id, user_text, current_image_prompt, current_aspect_ratio
        )
        with self._operation_scope():
            self._prepare_chat_backend(conversation_id)
            response = self.chat_backend.chat(messages, max_tokens=max_tokens,
                                              reasoning_effort=reasoning_effort)
            return self._finish_chat(conversation_id, user_text, response, preserve_aspect_ratio)

    def chat_stream(self, conversation_id: str, user_text: str, max_tokens: int = 8192, *,
                    current_image_prompt: str | None = None,
                    current_aspect_ratio: str | None = None,
                    reasoning_effort: str | None = None) -> Iterator[dict[str, Any]]:
        conversation_id, user_text, messages, preserve_aspect_ratio = self._build_chat_messages(
            conversation_id, user_text, current_image_prompt, current_aspect_ratio
        )
        with self._operation_scope():
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
        with self._operation_scope():
            generation_id: str | None = None
            prompt_id: str | None = None
            completed_image: dict[str, str] | None = None
            progress_socket = None
            selected_source_images: list[dict[str, Any]] = []
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

                report_generation_event(progress_callback, {
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
                    mode_default_megapixels(mode, generation)
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
                    report_generation_event(progress_callback, {
                        "type": "phase", "message": "Connected to ComfyUI live progress",
                    })
                except Exception:
                    report_generation_event(progress_callback, {
                        "type": "phase", "message": "Live progress unavailable; tracking completion status",
                    })
                report_generation_event(progress_callback, {
                    "type": "phase", "message": "Submitting workflow to ComfyUI",
                })
                response = self.comfy.submit_prompt(workflow, client_id=client_id)
                prompt_id = response.get("prompt_id")
                if not isinstance(prompt_id, str) or not prompt_id:
                    raise CoordinationError("ComfyUI did not return a prompt id")
                self._comfy_models_retained = True
                self.store.update_generation(generation_id, "running", prompt_id=prompt_id)
                report_generation_event(progress_callback, {
                    "type": "phase", "message": "ComfyUI is executing the workflow",
                })
                comfy_image = wait_for_generation(
                    self.comfy,
                    self.settings.section("coordination"),
                    prompt_id,
                    workflow,
                    self._clock,
                    self._sleep,
                    progress_socket,
                    progress_callback,
                )
                report_generation_event(progress_callback, {
                    "type": "phase", "message": "Saving generated image to the session and output archive",
                })
                completed_image = self._assets.relocate_generated_image(
                    conversation_id, comfy_image,
                )
                self.store.update_generation(generation_id, "complete", image=completed_image)
            except Exception as error:
                if generation_id is not None:
                    try:
                        self.store.update_generation(
                            generation_id, "failed", prompt_id=prompt_id,
                            error=str(error)[:1000],
                        )
                    except StorageError:
                        pass
                if isinstance(error, CoordinationError):
                    raise
                raise CoordinationError(str(error)) from error
            finally:
                if progress_socket is not None:
                    try:
                        progress_socket.close()
                    except Exception:
                        pass
            if generation_id is None or prompt_id is None or completed_image is None:
                raise CoordinationError("generation ended without a complete result")
            report_generation_event(progress_callback, {
                "type": "phase", "message": "Keeping ComfyUI models loaded for the next generation",
            })
            report_generation_event(progress_callback, {
                "type": "phase", "message": "Image ready",
            })
            return GenerationOutcome(generation_id, prompt_id, completed_image)