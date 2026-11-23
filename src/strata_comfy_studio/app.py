from __future__ import annotations

import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .clients import BackendError, ComfyUIClient, StrataClient
from .image_assets import MAX_SOURCE_IMAGE_BYTES, SourceImageError, inspect_source_image
from .orchestrator import CoordinationError, Orchestrator
from .permissions import set_private_permissions
from .security import require_access
from .settings import ConfigError, Settings, default_config_path, load_config
from .storage import ConversationStore, StorageConflictError, StorageError
from .system_metrics import SystemMetrics


class ChatRequest(BaseModel):
    conversation_id: str
    message: str = Field(min_length=1, max_length=20000)
    current_image_prompt: str | None = Field(default=None, max_length=20000)
    current_aspect_ratio: str | None = Field(default=None, max_length=64)
    reasoning_effort: Literal["none", "low", "medium", "high"] = "none"


class GenerateRequest(BaseModel):
    conversation_id: str
    prompt: str = Field(min_length=1, max_length=20000)
    seed: int = Field(ge=0, le=2**64 - 1)
    aspect_ratio: str | None = None
    megapixels: float | None = Field(default=None, ge=0.5, le=20.0, multiple_of=0.01)
    sampler_name: str | None = None
    scheduler: str | None = None
    image_mode: Literal["text-to-image", "edit"] = "text-to-image"
    source_image_ids: list[str] = Field(default_factory=list, max_length=5)


class ConversationModeRequest(BaseModel):
    image_mode: Literal["text-to-image", "edit"]


class SourceImageOrderRequest(BaseModel):
    source_image_ids: list[str] = Field(max_length=5)


def _public_source_image(source_image: dict[str, Any]) -> dict[str, Any]:
    return {
        key: source_image[key]
        for key in ("id", "content_type", "size_bytes", "width", "height", "position", "created_at", "in_use")
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


class ChatRun:
    def __init__(self, conversation_id: str, message: str, base_message_count: int):
        self.id = uuid.uuid4().hex
        self.conversation_id = conversation_id
        self.message = message
        self.base_message_count = base_message_count
        self._events: list[dict[str, Any]] = []
        self._condition = threading.Condition()
        self._complete = False

    @property
    def complete(self) -> bool:
        with self._condition:
            return self._complete

    def publish(self, event: dict[str, Any]) -> None:
        with self._condition:
            self._events.append(event)
            self._condition.notify_all()

    def finish(self, event: dict[str, Any]) -> None:
        with self._condition:
            self._events.append(event)
            self._complete = True
            self._condition.notify_all()

    def public_info(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "message": self.message,
            "base_message_count": self.base_message_count,
        }

    def stream(self) -> Iterator[str]:
        index = 0
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


class ChatRunManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._active: ChatRun | None = None
        self._runs: dict[str, ChatRun] = {}

    def start(self, conversation_id: str, message: str, base_message_count: int,
              operation: Callable[[], Iterator[dict[str, Any]]]) -> ChatRun:
        with self._lock:
            if self._active is not None and not self._active.complete:
                raise CoordinationError("another model operation is already in progress")
            run = ChatRun(conversation_id, message, base_message_count)
            self._active = run
            self._runs[run.id] = run
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
                    logging.exception("Could not close the Strata chat stream")
            if terminal_event is None:
                terminal_event = {"type": "error", "message": "Chat stream ended unexpectedly."}
            with self._lock:
                run.finish(terminal_event)
                if self._active is run:
                    self._active = None
                completed = [run_id for run_id, item in self._runs.items() if item.complete]
                for run_id in completed[:-8]:
                    del self._runs[run_id]

    def active(self) -> ChatRun | None:
        with self._lock:
            return self._active

    def get(self, run_id: str) -> ChatRun | None:
        with self._lock:
            return self._runs.get(run_id)


class GenerationRun:
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

    def public_info(self) -> dict[str, Any]:
        with self._condition:
            return {
                "id": self.id,
                "conversation_id": self.conversation_id,
                "sequence": len(self._events),
                "latest_event": self._events[-1] if self._events else None,
            }

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


class GenerationRunManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._active: GenerationRun | None = None
        self._runs: dict[str, GenerationRun] = {}

    def start(self, conversation_id: str) -> GenerationRun:
        with self._lock:
            if self._active is not None and not self._active.complete:
                raise CoordinationError("another model operation is already in progress")
            run = GenerationRun(conversation_id)
            self._active = run
            self._runs[run.id] = run
            return run

    def finish(self, run: GenerationRun, event: dict[str, Any]) -> None:
        run.finish(event)
        with self._lock:
            if self._active is run:
                self._active = None
            completed = [run_id for run_id, item in self._runs.items() if item.complete]
            for run_id in completed[:-8]:
                del self._runs[run_id]

    def active(self) -> GenerationRun | None:
        with self._lock:
            return self._active if self._active and not self._active.complete else None

    def get(self, run_id: str) -> GenerationRun | None:
        with self._lock:
            return self._runs.get(run_id)


def create_app(settings: Settings | None = None, store: ConversationStore | None = None,
               strata: StrataClient | None = None, comfy: ComfyUIClient | None = None) -> FastAPI:
    settings = settings or load_config()
    store = store or ConversationStore(settings.path("paths.database"))
    strata = strata or StrataClient(settings)
    comfy = comfy or ComfyUIClient(settings)
    orchestrator = Orchestrator(settings, store, strata, comfy)
    chat_runs = ChatRunManager()
    generation_runs = GenerationRunManager()
    resource_monitor = SystemMetrics(settings.section("comfyui")["gpu_id"])
    static_dir = Path(__file__).resolve().parent / "static"

    app = FastAPI(title="Strata + MiniMax Studio", docs_url=None, redoc_url=None,
                  telemetry={"auto_configure": False, "tracing": False, "metrics": False,
                             "logs": False, "operation_spans": False})
    app.state.settings = settings
    app.state.store = store
    app.state.strata = strata
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
        if asset_name not in {"app.css", "app.js"}:
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(static_dir / asset_name)

    @app.get("/api/state", dependencies=[Depends(authenticated)])
    async def state() -> dict[str, Any]:
        strata_status: dict[str, Any]
        comfy_status: dict[str, Any]
        try:
            response = strata.status()
            loaded = response.get("loaded")
            strata_status = {"available": isinstance(loaded, bool), "loaded": loaded}
        except BackendError as error:
            strata_status = {"available": False, "error": str(error)}
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
            "services": {"strata": strata_status, "comfyui": comfy_status},
            "active_conversation_id": store.selected_conversation_id(),
            "conversations": store.list_conversations(),
            "active_chat": active_chat.public_info() if active_chat else None,
            "active_generation": active_generation.public_info() if active_generation else None,
        }

    @app.get("/api/system-metrics", dependencies=[Depends(authenticated)])
    def system_metrics() -> dict[str, float | None]:
        return resource_monitor.snapshot()

    @app.get("/api/options", dependencies=[Depends(authenticated)])
    async def options() -> dict[str, Any]:
        generation = settings.section("generation")
        return {
            "aspect_ratios": generation["aspect_ratios"],
            "default_aspect_ratio": generation["default_aspect_ratio"],
            "megapixels": {"min": 0.5, "max": 20.0, "step": 0.01,
                           "default": generation["default_megapixels"]},
            "edit_megapixels": {"min": 0.5, "max": 16.0, "step": 0.01,
                                "default": min(generation["default_megapixels"], 16.0)},
            "resolution_multiple": generation["resolution_multiple"],
            "prompt_enhancement": False,
            "source_images": {
                "max_count": 5,
                "max_bytes": MAX_SOURCE_IMAGE_BYTES,
                "max_megapixels": 100,
                "allowed_types": ["image/png", "image/jpeg", "image/webp"],
            },
        }

    @app.post("/api/conversations", dependencies=[Depends(authenticated)])
    async def create_conversation() -> dict[str, Any]:
        conversation = store.create_conversation()
        store.set_selected_conversation_id(conversation["id"])
        return conversation

    @app.get("/api/conversations/{conversation_id}", dependencies=[Depends(authenticated)])
    async def get_conversation(conversation_id: str) -> dict[str, Any]:
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
    async def set_conversation_mode(conversation_id: str, body: ConversationModeRequest) -> dict[str, str]:
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
            source_image = orchestrator.upload_source_image(
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
    async def reorder_source_images(conversation_id: str,
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
    async def delete_source_image(conversation_id: str, source_image_id: str) -> dict[str, bool]:
        try:
            orchestrator.delete_source_image(conversation_id, source_image_id)
        except StorageConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except StorageError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except CoordinationError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"deleted": True}

    @app.get("/api/conversations/{conversation_id}/source-images/{source_image_id}",
             dependencies=[Depends(authenticated)])
    async def source_image(conversation_id: str, source_image_id: str) -> Response:
        try:
            image_record = store.source_image(conversation_id, source_image_id)
        except StorageError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if image_record is None:
            raise HTTPException(status_code=404, detail="Source image not found.")
        try:
            content, media_type = comfy.image(
                image_record["filename"], image_record["subfolder"], "input",
            )
        except BackendError as error:
            raise HTTPException(status_code=502, detail="ComfyUI source image retrieval failed.") from error
        return Response(content, media_type=media_type,
                        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    @app.delete("/api/conversations/{conversation_id}", dependencies=[Depends(authenticated)])
    async def delete_conversation(conversation_id: str) -> dict[str, Any]:
        try:
            cleanup_warning = orchestrator.delete_conversation(conversation_id)
        except StorageError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except CoordinationError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"deleted": True, "cleanup_warning": cleanup_warning}

    @app.post("/api/chat", dependencies=[Depends(authenticated)])
    async def chat(body: ChatRequest) -> dict[str, Any]:
        try:
            result = orchestrator.chat(
                body.conversation_id,
                body.message,
                current_image_prompt=body.current_image_prompt,
                current_aspect_ratio=body.current_aspect_ratio,
                reasoning_effort=body.reasoning_effort,
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
    async def chat_stream(body: ChatRequest) -> StreamingResponse:
        try:
            base_message_count = len(store.messages(body.conversation_id))
            run = chat_runs.start(
                body.conversation_id,
                body.message,
                base_message_count,
                lambda: orchestrator.chat_stream(
                    body.conversation_id,
                    body.message,
                    current_image_prompt=body.current_image_prompt,
                    current_aspect_ratio=body.current_aspect_ratio,
                    reasoning_effort=body.reasoning_effort,
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
    async def resume_chat_stream(run_id: str) -> StreamingResponse:
        run = chat_runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Chat run not found")
        return StreamingResponse(
            run.stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/generate/runs/{run_id}/stream", dependencies=[Depends(authenticated)])
    async def resume_generation_stream(run_id: str, after: int = 0) -> StreamingResponse:
        run = generation_runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Generation run not found")
        return StreamingResponse(
            run.stream(after),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/generate", dependencies=[Depends(authenticated)])
    async def generate(body: GenerateRequest) -> dict[str, Any]:
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
    async def generate_stream(body: GenerateRequest) -> StreamingResponse:
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
    async def generation_image(generation_id: str) -> Response:
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


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config_path = default_config_path()
        set_private_permissions(config_path)
        settings = load_config(config_path)
        app_config = settings.section("app")
        uvicorn.run(create_app(settings), host=app_config["listen_host"], port=app_config["port"],
                    log_level="info", access_log=False, workers=1,
                    forwarded_allow_ips="127.0.0.1,::1", proxy_headers=False, env_file=None)
    except (ConfigError, ValueError, OSError) as error:
        raise SystemExit(f"studio: {error}") from error


if __name__ == "__main__":
    run()