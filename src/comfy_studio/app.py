from __future__ import annotations

import json
import logging
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from .chat_backends import create_chat_backend, get_chat_backend
from .clients import BackendError, ChatBackend, ComfyUIClient
from .image_assets import MAX_SOURCE_IMAGE_BYTES, SourceImageError, inspect_source_image
from .modes import (
    DEFAULT_MODE_ID,
    IMAGE_MODES,
    MAX_SOURCE_IMAGES,
    get_mode,
    public_mode_options,
)
from .orchestrator import CoordinationError, Orchestrator
from .permissions import set_private_permissions
from .security import require_access
from .settings import CHAT_BACKENDS, ConfigError, Settings, default_config_path, load_config
from .services import ServiceError, service_running
from .storage import ConversationStore, StorageError
from .system_metrics import SystemMetrics


REASONING_EFFORT_LABELS = {
    "none": "Off",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "xhigh": "XHigh",
}
class ChatRequest(BaseModel):
    conversation_id: str
    message: str = Field(min_length=1, max_length=20000)
    current_image_prompt: str | None = Field(default=None, max_length=20000)
    current_aspect_ratio: str | None = Field(default=None, max_length=64)
    reasoning_effort: str | None = Field(default=None, max_length=32)


class GenerateRequest(BaseModel):
    conversation_id: str
    prompt: str = Field(min_length=1, max_length=20000)
    seed: int = Field(ge=0, le=2**64 - 1)
    aspect_ratio: str | None = None
    megapixels: float | None = Field(default=None, ge=0.5, le=20.0, multiple_of=0.01)
    sampler_name: str | None = None
    scheduler: str | None = None
    image_mode: str = DEFAULT_MODE_ID
    source_image_ids: list[str] = Field(default_factory=list, max_length=MAX_SOURCE_IMAGES)

    @field_validator("image_mode")
    @classmethod
    def validate_image_mode(cls, value: str) -> str:
        get_mode(value)
        return value


class ConversationModeRequest(BaseModel):
    image_mode: str

    @field_validator("image_mode")
    @classmethod
    def validate_image_mode(cls, value: str) -> str:
        get_mode(value)
        return value


class SourceImageOrderRequest(BaseModel):
    source_image_ids: list[str] = Field(max_length=MAX_SOURCE_IMAGES)


def _public_source_image(source_image: dict[str, Any]) -> dict[str, Any]:
    return {
        key: source_image[key]
        for key in (
            "id", "content_type", "content_hash", "size_bytes", "width", "height", "position", "created_at",
        )
    }


def _message_for_ui(message: dict[str, str]) -> dict[str, str]:
    display_message = {key: value for key, value in message.items() if key != "reasoning_content"}
    reasoning = message.get("reasoning_content")
    if reasoning:
        display_message["reasoning_content"] = reasoning
    if message["role"] != "assistant":
        return display_message
    try:
        parsed = json.loads(message["content"])
    except json.JSONDecodeError:
        return display_message
    if isinstance(parsed, dict) and isinstance(parsed.get("reply"), str):
        display_message["content"] = parsed["reply"]
    return display_message


def _resolve_reasoning_effort(settings: Settings, requested: str | None) -> str:
    backend = get_chat_backend(settings.chat_backend)
    allowed = backend.reasoning_efforts
    effort = requested if requested is not None else backend.default_reasoning_effort
    if effort not in allowed:
        choices = ", ".join(allowed)
        raise HTTPException(
            status_code=422,
            detail=f"reasoning_effort for {settings.chat_backend} must be one of: {choices}",
        )
    return effort


_STATIC_ASSETS = {
    "app.css": "text/css",
    "app.js": "text/javascript",
    "api.js": "text/javascript",
    "resolution-ui.js": "text/javascript",
    "source-images.js": "text/javascript",
    "gallery.js": "text/javascript",
}


class EventRun:
    def __init__(self, conversation_id: str):
        self.id = uuid.uuid4().hex
        self.conversation_id = conversation_id
        self._events: list[dict[str, Any]] = []
        self._condition = threading.Condition()
        self._complete = False

    @property
    def complete(self) -> bool:
        with self._condition:
            return self._complete

    def publish(self, event: dict[str, Any]) -> None:
        with self._condition:
            if self._complete:
                return
            self._events.append(event)
            self._condition.notify_all()

    def finish(self, event: dict[str, Any]) -> None:
        with self._condition:
            if self._complete:
                return
            self._events.append(event)
            self._complete = True
            self._condition.notify_all()

    def stream(self, after_sequence: int = 0) -> Iterator[str]:
        index = max(0, after_sequence)
        while True:
            with self._condition:
                if index < len(self._events):
                    event = self._events[index]
                    index += 1
                elif self._complete:
                    return
                else:
                    notified = self._condition.wait(timeout=15)
                    if notified:
                        continue
                    event = None
            if event is None:
                yield ": keep-alive\n\n"
                continue
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event.get("type") in {"done", "error"}:
                return


class ChatRun(EventRun):
    def __init__(self, conversation_id: str, message: str, base_message_count: int):
        super().__init__(conversation_id)
        self.message = message
        self.base_message_count = base_message_count

    def public_info(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "message": self.message,
            "base_message_count": self.base_message_count,
        }


class GenerationRun(EventRun):
    def public_info(self) -> dict[str, Any]:
        with self._condition:
            return {
                "id": self.id,
                "conversation_id": self.conversation_id,
                "sequence": len(self._events),
                "latest_event": self._events[-1] if self._events else None,
            }


class RunGate:
    def __init__(self):
        self.lock = threading.Lock()
        self._active: EventRun | None = None

    def claim(self, run: EventRun) -> None:
        if self._active is not None and not self._active.complete:
            raise CoordinationError("another model operation is already in progress")
        self._active = run

    def release(self, run: EventRun) -> None:
        if self._active is run:
            self._active = None


class _BaseRunManager:
    def __init__(self, gate: RunGate | None = None):
        self._gate = gate or RunGate()
        self._active: Any = None
        self._runs: dict[str, Any] = {}

    def _register_run(self, run: EventRun) -> None:
        with self._gate.lock:
            self._gate.claim(run)
            self._active = run
            self._runs[run.id] = run

    def _complete_run(self, run: EventRun, terminal_event: dict[str, Any]) -> None:
        with self._gate.lock:
            run.finish(terminal_event)
            if self._active is run:
                self._active = None
            self._gate.release(run)
            completed = [run_id for run_id, item in self._runs.items() if item.complete]
            for run_id in completed[:-8]:
                del self._runs[run_id]

    def active(self) -> Any:
        with self._gate.lock:
            return self._active if self._active and not self._active.complete else None

    def get(self, run_id: str) -> Any:
        with self._gate.lock:
            return self._runs.get(run_id)


class ChatRunManager(_BaseRunManager):
    def start(self, conversation_id: str, message: str, base_message_count: int,
              operation: Callable[[], Iterator[dict[str, Any]]]) -> ChatRun:
        run = ChatRun(conversation_id, message, base_message_count)
        self._register_run(run)
        thread = threading.Thread(target=self._execute, args=(run, operation),
                                  name="studio-chat", daemon=True)
        thread.start()
        return run

    def _execute(self, run: ChatRun, operation: Callable[[], Iterator[dict[str, Any]]]) -> None:
        stream: Iterator[dict[str, Any]] | None = None
        terminal_event: dict[str, Any] | None = None
        try:
            stream = operation()
            for event in stream:
                if event.get("type") in {"done", "error"}:
                    terminal_event = event
                else:
                    run.publish(event)
            if terminal_event is None:
                terminal_event = {"type": "error", "message": "Chat stream ended before the answer was complete."}
        except (BackendError, CoordinationError, StorageError) as error:
            terminal_event = {"type": "error", "message": str(error)}
        except Exception:
            logging.exception("Unexpected streamed chat failure")
            terminal_event = {"type": "error", "message": "Chat failed unexpectedly."}
        finally:
            close_stream = getattr(stream, "close", None)
            if close_stream is not None:
                try:
                    close_stream()
                except Exception:
                    logging.exception("Could not close the chat stream")
            if terminal_event is None:
                terminal_event = {"type": "error", "message": "Chat stream ended unexpectedly."}
            self._complete_run(run, terminal_event)


class GenerationRunManager(_BaseRunManager):
    def start(self, conversation_id: str) -> GenerationRun:
        run = GenerationRun(conversation_id)
        self._register_run(run)
        return run

    def finish(self, run: GenerationRun, event: dict[str, Any]) -> None:
        self._complete_run(run, event)


def create_app(settings: Settings | None = None, store: ConversationStore | None = None,
               chat_backend: ChatBackend | None = None,
               comfy: ComfyUIClient | None = None) -> FastAPI:
    settings = settings or load_config()
    store = store or ConversationStore(
        settings.path("paths.session_root"),
        settings.path("paths.runtime_root"),
    )
    if chat_backend is None:
        chat_backend = create_chat_backend(settings)
    comfy = comfy or ComfyUIClient(settings)
    orchestrator = Orchestrator(settings, store, chat_backend, comfy)
    run_gate = RunGate()
    chat_runs = ChatRunManager(run_gate)
    generation_runs = GenerationRunManager(run_gate)
    resource_monitor = SystemMetrics(settings.section("comfyui")["gpu_id"])
    static_dir = Path(__file__).resolve().parent / "static"

    app = FastAPI(title="Comfy Studio", docs_url=None, redoc_url=None,
                  telemetry={"auto_configure": False, "tracing": False, "metrics": False,
                             "logs": False, "operation_spans": False})
    app.state.settings = settings
    app.state.store = store
    app.state.chat_backend = chat_backend
    app.state.comfy = comfy
    app.state.orchestrator = orchestrator
    app.state.chat_runs = chat_runs
    app.state.generation_runs = generation_runs
    app.state.resource_monitor = resource_monitor

    def authenticated(request: Request) -> None:
        require_access(request, settings)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html", media_type="text/html")

    @app.get("/assets/{asset_name}", include_in_schema=False)
    async def assets(asset_name: str) -> FileResponse:
        media_type = _STATIC_ASSETS.get(asset_name)
        if media_type is None:
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(static_dir / asset_name, media_type=media_type)

    @app.get("/api/state", dependencies=[Depends(authenticated)])
    def state() -> dict[str, Any]:
        backend_definition = get_chat_backend(settings.chat_backend)
        chat_backend_status: dict[str, Any]
        comfy_status: dict[str, Any]
        try:
            if (backend_definition.loaded_state == "process"
                    and not service_running(settings, settings.chat_backend)):
                chat_backend_status = {"available": True, "loaded": False}
            else:
                response = chat_backend.status()
                loaded = response.get("loaded")
                chat_backend_status = {
                    "available": isinstance(loaded, bool),
                    "loaded": loaded,
                }
        except (BackendError, ServiceError) as error:
            chat_backend_status = {"available": False, "error": str(error)}
        try:
            queue = comfy.queue()
            stats = comfy.system_stats()
            devices = stats.get("devices", [])
            gpu_id = settings.section("comfyui")["gpu_id"]
            device = next((entry for entry in devices if isinstance(entry, dict) and entry.get("index") == gpu_id), None)
            comfy_status = {
                "available": True,
                "queue_running": len(queue.get("queue_running", [])),
                "queue_pending": len(queue.get("queue_pending", [])),
                "vram_free_mib": int(device["vram_free"] // (1024 * 1024)) if device else None,
            }
        except (BackendError, TypeError, KeyError) as error:
            comfy_status = {"available": False, "error": str(error)}
        active_chat = chat_runs.active()
        active_generation = generation_runs.active()
        return {
            "chat_backend": settings.chat_backend,
            "chat_backend_label": backend_definition.label,
            "services": {settings.chat_backend: chat_backend_status, "comfyui": comfy_status},
            "active_conversation_id": store.selected_conversation_id(),
            "conversations": store.list_conversations(),
            "active_chat": active_chat.public_info() if active_chat else None,
            "active_generation": active_generation.public_info() if active_generation else None,
        }

    @app.get("/api/system-metrics", dependencies=[Depends(authenticated)])
    def system_metrics() -> dict[str, float | None]:
        return resource_monitor.snapshot()

    @app.get("/api/options", dependencies=[Depends(authenticated)])
    def options() -> dict[str, Any]:
        generation = settings.section("generation")
        modes = public_mode_options(generation)
        backend_definition = get_chat_backend(settings.chat_backend)
        options: dict[str, Any] = {
            "default_mode": DEFAULT_MODE_ID,
            "modes": modes,
            "reasoning_efforts": [
                {
                    "value": effort,
                    "label": REASONING_EFFORT_LABELS.get(
                        effort, effort.replace("_", " ").title()
                    ),
                }
                for effort in backend_definition.reasoning_efforts
            ],
            "default_reasoning_effort": backend_definition.default_reasoning_effort,
            "chat_backend": {
                "id": backend_definition.id,
                "label": backend_definition.label,
            },
            "aspect_ratios": generation["aspect_ratios"],
            "default_aspect_ratio": generation["default_aspect_ratio"],
            "resolution_multiple": generation["resolution_multiple"],
            "prompt_enhancement": False,
            "source_images": {
                "max_count": MAX_SOURCE_IMAGES,
                "max_bytes": MAX_SOURCE_IMAGE_BYTES,
                "max_megapixels": 100,
                "allowed_types": ["image/png", "image/jpeg", "image/webp"],
            },
        }
        return options

    @app.get("/api/conversations", dependencies=[Depends(authenticated)])
    def list_conversations() -> dict[str, Any]:
        return {
            "active_conversation_id": store.selected_conversation_id(),
            "conversations": store.list_conversations(),
        }

    @app.post("/api/conversations", dependencies=[Depends(authenticated)])
    def create_conversation() -> dict[str, Any]:
        conversation = store.create_conversation()
        store.set_selected_conversation_id(conversation["id"])
        return conversation

    @app.get("/api/conversations/{conversation_id}", dependencies=[Depends(authenticated)])
    def get_conversation(conversation_id: str) -> dict[str, Any]:
        conversation = store.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        store.set_selected_conversation_id(conversation_id)
        return {
            "conversation": conversation,
            "messages": [_message_for_ui(message) for message in store.display_messages(conversation_id)],
            "generations": store.generations(conversation_id),
            "source_images": [
                _public_source_image(image) for image in store.source_images(conversation_id)
            ],
        }

    @app.put("/api/conversations/{conversation_id}/mode", dependencies=[Depends(authenticated)])
    def set_conversation_mode(conversation_id: str, body: ConversationModeRequest) -> dict[str, str]:
        try:
            orchestrator.set_conversation_mode(conversation_id, body.image_mode)
        except StorageError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except CoordinationError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"image_mode": body.image_mode}

    @app.post("/api/conversations/{conversation_id}/source-images",
              dependencies=[Depends(authenticated)])
    async def upload_source_image(conversation_id: str, request: Request) -> dict[str, Any]:
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type not in {"image/png", "image/jpeg", "image/webp"}:
            raise HTTPException(status_code=415, detail="Only PNG, JPEG, and WebP source images are supported.")
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError as error:
                raise HTTPException(status_code=400, detail="Invalid Content-Length header.") from error
            if declared_length < 0:
                raise HTTPException(status_code=400, detail="Invalid Content-Length header.")
            if declared_length > MAX_SOURCE_IMAGE_BYTES:
                raise HTTPException(status_code=413, detail="Source images may not exceed 25 MiB.")
        chunks: list[bytes] = []
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_SOURCE_IMAGE_BYTES:
                raise HTTPException(status_code=413, detail="Source images may not exceed 25 MiB.")
            chunks.append(chunk)
        content = b"".join(chunks)
        try:
            inspected = await run_in_threadpool(inspect_source_image, content, content_type)
        except SourceImageError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        try:
            source_image = await run_in_threadpool(
                orchestrator.upload_source_image,
                conversation_id, content, inspected.content_type, inspected.extension,
                inspected.width, inspected.height,
            )
        except StorageError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except CoordinationError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except BackendError as error:
            raise HTTPException(status_code=502, detail="ComfyUI source image upload failed.") from error
        return _public_source_image(source_image)

    @app.put("/api/conversations/{conversation_id}/source-images/order",
             dependencies=[Depends(authenticated)])
    def reorder_source_images(conversation_id: str,
                              body: SourceImageOrderRequest) -> dict[str, Any]:
        try:
            source_images = orchestrator.reorder_source_images(
                conversation_id, body.source_image_ids,
            )
        except StorageError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except CoordinationError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"source_images": [_public_source_image(image) for image in source_images]}

    @app.delete("/api/conversations/{conversation_id}/source-images/{source_image_id}",
                dependencies=[Depends(authenticated)])
    def delete_source_image(conversation_id: str, source_image_id: str) -> dict[str, bool]:
        try:
            orchestrator.delete_source_image(conversation_id, source_image_id)
        except StorageError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except CoordinationError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"deleted": True}

    @app.get("/api/conversations/{conversation_id}/source-images/{source_image_id}",
             dependencies=[Depends(authenticated)])
    def source_image(conversation_id: str, source_image_id: str) -> Response:
        try:
            image_record = store.source_image(conversation_id, source_image_id)
        except StorageError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if image_record is None:
            raise HTTPException(status_code=404, detail="Source image not found.")
        try:
            image_path = orchestrator.source_image_path(conversation_id, image_record)
        except CoordinationError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return FileResponse(
            image_path,
            media_type=image_record["content_type"],
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @app.delete("/api/conversations/{conversation_id}", dependencies=[Depends(authenticated)])
    def delete_conversation(conversation_id: str) -> dict[str, Any]:
        try:
            cleanup_warning = orchestrator.delete_conversation(conversation_id)
        except StorageError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except CoordinationError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"deleted": True, "cleanup_warning": cleanup_warning}

    @app.post("/api/chat", dependencies=[Depends(authenticated)])
    def chat(body: ChatRequest) -> dict[str, Any]:
        reasoning_effort = _resolve_reasoning_effort(settings, body.reasoning_effort)
        try:
            result = orchestrator.chat(
                body.conversation_id,
                body.message,
                max_tokens=settings.backend_config()["max_tokens"],
                current_image_prompt=body.current_image_prompt,
                current_aspect_ratio=body.current_aspect_ratio,
                reasoning_effort=reasoning_effort,
            )
        except CoordinationError as error:
            status = 409 if "busy" in str(error).lower() or "queue" in str(error).lower() else 503
            raise HTTPException(status_code=status, detail=str(error)) from error
        return {
            "answer": result.answer,
            "image_prompt": result.image_prompt,
            "aspect_ratio": result.aspect_ratio,
            "reasoning_content": result.reasoning_content,
            "context_saved": result.context_saved,
            "warning": result.warning,
        }

    @app.post("/api/chat/stream", dependencies=[Depends(authenticated)])
    def chat_stream(body: ChatRequest) -> StreamingResponse:
        reasoning_effort = _resolve_reasoning_effort(settings, body.reasoning_effort)
        try:
            base_message_count = len(store.messages(body.conversation_id))
            run = chat_runs.start(
                body.conversation_id,
                body.message,
                base_message_count,
                lambda: orchestrator.chat_stream(
                    body.conversation_id,
                    body.message,
                    max_tokens=settings.backend_config()["max_tokens"],
                    current_image_prompt=body.current_image_prompt,
                    current_aspect_ratio=body.current_aspect_ratio,
                    reasoning_effort=reasoning_effort,
                ),
            )
        except StorageError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except CoordinationError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

        return StreamingResponse(
            run.stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                     "X-Chat-Run-Id": run.id},
        )

    @app.get("/api/chat/runs/{run_id}/stream", dependencies=[Depends(authenticated)])
    def resume_chat_stream(run_id: str) -> StreamingResponse:
        run = chat_runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Chat run not found")
        return StreamingResponse(
            run.stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/generate/runs/{run_id}/stream", dependencies=[Depends(authenticated)])
    def resume_generation_stream(run_id: str, after: int = 0) -> StreamingResponse:
        run = generation_runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Generation run not found")
        return StreamingResponse(
            run.stream(after),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/generate", dependencies=[Depends(authenticated)])
    def generate(body: GenerateRequest) -> dict[str, Any]:
        try:
            result = orchestrator.generate(
                body.conversation_id, body.prompt, body.seed,
                body.aspect_ratio, body.megapixels, body.sampler_name, body.scheduler,
                image_mode=body.image_mode, source_image_ids=body.source_image_ids,
            )
        except CoordinationError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {
            "generation_id": result.generation_id,
            "prompt_id": result.prompt_id,
            "image": result.image,
            "cleanup_warning": result.cleanup_warning,
        }

    @app.post("/api/generate/stream", dependencies=[Depends(authenticated)])
    def generate_stream(body: GenerateRequest) -> StreamingResponse:
        try:
            run = generation_runs.start(body.conversation_id)
        except CoordinationError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

        def run_generation() -> None:
            try:
                result = orchestrator.generate(
                    body.conversation_id, body.prompt, body.seed,
                    body.aspect_ratio, body.megapixels,
                    body.sampler_name, body.scheduler,
                    progress_callback=run.publish,
                    image_mode=body.image_mode, source_image_ids=body.source_image_ids,
                )
                generation_runs.finish(run, {
                    "type": "done",
                    "generation_id": result.generation_id,
                    "prompt_id": result.prompt_id,
                    "image": result.image,
                    "cleanup_warning": result.cleanup_warning,
                })
            except (BackendError, CoordinationError, StorageError) as error:
                generation_runs.finish(run, {"type": "error", "message": str(error)})
            except Exception:
                logging.exception("Unexpected streamed image generation failure")
                generation_runs.finish(run, {
                    "type": "error", "message": "Image generation failed unexpectedly.",
                })

        threading.Thread(target=run_generation, name="studio-generation", daemon=True).start()

        return StreamingResponse(
            run.stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                     "X-Generation-Run-Id": run.id},
        )

    @app.get("/api/generations/{generation_id}/image", dependencies=[Depends(authenticated)])
    def generation_image(generation_id: str) -> Response:
        try:
            generation = store.generation(generation_id)
        except StorageError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        if generation is None or generation["status"] != "complete" or not generation["filename"]:
            raise HTTPException(status_code=404, detail="Generated image not found")
        try:
            image_path = orchestrator.generated_image_path(generation)
        except CoordinationError as error:
            raise HTTPException(status_code=404, detail="Generated image not found") from error
        return FileResponse(image_path, media_type="image/png",
                            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    return app


def run(settings: Settings | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if settings is None:
            arguments = sys.argv[1:]
            if len(arguments) > 1 or (arguments and arguments[0] not in CHAT_BACKENDS):
                raise SystemExit("usage: python -m comfy_studio.app [strata|bonsai]")
            chat_backend = arguments[0] if arguments else "strata"
            config_path = default_config_path()
            set_private_permissions(config_path)
            settings = load_config(config_path, chat_backend=chat_backend)
        app_config = settings.section("app")
        uvicorn.run(create_app(settings), host=app_config["listen_host"], port=app_config["port"],
                    log_level="info", access_log=False, workers=1,
                    forwarded_allow_ips="127.0.0.1,::1", proxy_headers=False, env_file=None)
    except (ConfigError, ValueError, OSError) as error:
        raise SystemExit(f"studio: {error}") from error


if __name__ == "__main__":
    run()