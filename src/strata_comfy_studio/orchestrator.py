from __future__ import annotations

import threading
import time
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from .clients import BackendError, ComfyUIClient, StrataClient
from .settings import Settings
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
                 strata: StrataClient, comfy: ComfyUIClient,
                 clock: Callable[[], float] = time.monotonic,
                 sleeper: Callable[[float], None] = time.sleep):
        self.settings = settings
        self.store = store
        self.strata = strata
        self.comfy = comfy
        self._clock = clock
        self._sleep = sleeper
        self._operation_lock = threading.Lock()

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

    def _free_vram_mib(self) -> int:
        devices = self.comfy.system_stats().get("devices")
        if not isinstance(devices, list):
            raise CoordinationError("ComfyUI did not report GPU devices")
        wanted = self.settings.section("comfyui")["gpu_id"]
        device = next((entry for entry in devices if isinstance(entry, dict) and entry.get("index") == wanted), None)
        if device is None:
            raise CoordinationError(f"ComfyUI did not report configured GPU index {wanted}")
        free_bytes = device.get("vram_free")
        if isinstance(free_bytes, bool) or not isinstance(free_bytes, (int, float)) or free_bytes < 0:
            raise CoordinationError("ComfyUI reported an invalid free-VRAM value")
        return int(free_bytes // (1024 * 1024))

    def _minimum_free_vram_mib(self) -> int:
        value = self.settings.section("strata")["min_free_vram_mib"]
        if value is None:
            raise CoordinationError("set strata.min_free_vram_mib in config.json before model switching")
        return value

    def _wait_for_free_vram(self) -> None:
        minimum = self._minimum_free_vram_mib()
        coordination = self.settings.section("coordination")
        deadline = self._clock() + coordination["handoff_timeout_s"]
        while True:
            self._require_idle_queue()
            free = self._free_vram_mib()
            if free >= minimum:
                return
            if self._clock() >= deadline:
                raise CoordinationError(
                    f"only {free} MiB VRAM is free; need {minimum} MiB before loading the next model"
                )
            self._sleep(coordination["poll_interval_s"])

    def _release_comfy(self) -> None:
        self._require_idle_queue()
        self.comfy.free_memory()
        self._wait_for_free_vram()

    def _strata_loaded(self) -> bool:
        response = self.strata.status()
        loaded = response.get("loaded")
        if not isinstance(loaded, bool):
            raise CoordinationError("Strata load state is unknown; refusing GPU handoff")
        return loaded

    def _session_path(self, conversation_id: str) -> Path:
        filename = self.store.session_filename(conversation_id)
        directory = self.settings.path("paths.strata_sessions")
        return directory / filename

    def delete_conversation(self, conversation_id: str) -> None:
        conversation_id = self.store.validate_id(conversation_id)
        if self.store.get_conversation(conversation_id) is None:
            raise StorageError("conversation does not exist")
        self._enter()
        try:
            if self.store.active_conversation_id() == conversation_id:
                if self._strata_loaded():
                    self.strata.unload(wait_s=300)
                    if self._strata_loaded():
                        raise CoordinationError("Strata did not unload; the conversation was not deleted")
                self.store.set_active_conversation_id(None)
            session_file = self._session_path(conversation_id)
            try:
                session_file.unlink(missing_ok=True)
            except OSError as error:
                raise CoordinationError(f"could not remove the saved Strata session; conversation kept: {error}") from error
            if not self.store.delete_conversation(conversation_id):
                raise StorageError("conversation does not exist")
        finally:
            self._leave()

    def _save_session(self, conversation_id: str) -> None:
        response = self.strata.save_session(self.store.session_filename(conversation_id))
        token_count = response.get("n_saved")
        byte_count = response.get("n_written")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (token_count, byte_count)):
            raise CoordinationError("Strata did not confirm the session save")
        self.store.mark_context_saved(conversation_id, token_count, byte_count)

    def _restore_session(self, conversation_id: str) -> None:
        conversation = self.store.get_conversation(conversation_id)
        if conversation is None:
            raise CoordinationError("conversation does not exist")
        if not conversation.get("context_saved_at"):
            if self.store.messages(conversation_id):
                raise CoordinationError("conversation has messages but no native Strata session file")
            return
        if not self.store.context_is_current(conversation_id):
            raise CoordinationError("saved Strata context does not match the latest transcript")
        if not self._session_path(conversation_id).is_file():
            raise CoordinationError("saved Strata session file is missing")
        response = self.strata.restore_session(self.store.session_filename(conversation_id))
        restored = response.get("n_restored")
        if isinstance(restored, bool) or not isinstance(restored, int) or restored <= 0:
            raise CoordinationError("Strata did not confirm the session restore")

    def _prepare_strata(self, conversation_id: str) -> None:
        self._require_idle_queue()
        loaded = self._strata_loaded()
        active_id = self.store.active_conversation_id()
        if loaded and active_id is None:
            raise CoordinationError("Strata is loaded but its active conversation is unknown")
        if loaded and active_id == conversation_id:
            return
        if loaded and active_id != conversation_id:
            if active_id is not None:
                self._save_session(active_id)
            self.strata.unload(wait_s=300)
            if self._strata_loaded():
                raise CoordinationError("Strata did not unload; refusing to replace its context")
            loaded = False
        if not loaded:
            self._release_comfy()
            self.strata.load()
            if not self._strata_loaded():
                raise CoordinationError("Strata load did not complete")
            self._restore_session(conversation_id)
            self.store.set_active_conversation_id(conversation_id)

    @staticmethod
    def _chat_content(response: dict[str, Any]) -> tuple[str, str]:
        try:
            message = response["choices"][0]["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise CoordinationError("Strata returned no assistant message") from error
        if isinstance(content, str):
            answer = content
        elif isinstance(content, list):
            answer = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        else:
            raise CoordinationError("Strata returned an unsupported message content type")
        if not answer.strip():
            raise CoordinationError("Strata returned an empty assistant message")
        reasoning = message.get("reasoning_content") or ""
        if not isinstance(reasoning, str):
            raise CoordinationError("Strata returned unsupported reasoning content")
        return answer, reasoning

    @classmethod
    def _parse_chat_response(cls, response: dict[str, Any]) -> tuple[str, str, str, str, str]:
        raw, reasoning = cls._chat_content(response)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            raise CoordinationError("Strata response was not the configured JSON object") from error
        if not isinstance(parsed, dict) or set(parsed) != {"reply", "image_prompt", "aspect_ratio"}:
            raise CoordinationError("Strata response must contain only reply, image_prompt, and aspect_ratio")
        reply, image_prompt, aspect_ratio = parsed["reply"], parsed["image_prompt"], parsed["aspect_ratio"]
        if (not isinstance(reply, str) or not reply.strip() or not isinstance(image_prompt, str)
                or not isinstance(aspect_ratio, str)):
            raise CoordinationError("Strata reply, image_prompt, and aspect_ratio must be strings")
        return reply.strip(), image_prompt.strip(), aspect_ratio.strip(), raw, reasoning

    def _build_chat_messages(self, conversation_id: str, user_text: str,
                             current_image_prompt: str | None,
                             current_aspect_ratio: str | None) -> tuple[str, str, list[dict[str, Any]]]:
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
        system_prompt_path = self.settings.path("generation.system_prompt_file")
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
        current_editor_state = json.dumps({
            "current_image_prompt": current_image_prompt,
            "current_aspect_ratio": current_aspect_ratio,
        }, ensure_ascii=False)
        current_user_message = (
            f"current_editor_state:\n{current_editor_state}\n\n"
            f"user_request:\n{user_text.strip()}"
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.store.messages(conversation_id))
        messages.append({"role": "user", "content": current_user_message})
        return conversation_id, user_text.strip(), messages

    def _finish_chat(self, conversation_id: str, user_text: str,
                     response: dict[str, Any]) -> ChatOutcome:
        answer, image_prompt, suggested_ratio, raw_answer, reasoning_content = self._parse_chat_response(response)
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
            self._save_session(conversation_id)
            return ChatOutcome(answer=display_answer, image_prompt=image_prompt, aspect_ratio=aspect_ratio,
                               context_saved=True, reasoning_content=reasoning_content)
        except (BackendError, StorageError, CoordinationError) as error:
            return ChatOutcome(answer=display_answer, image_prompt=image_prompt, aspect_ratio=aspect_ratio,
                               context_saved=False,
                               warning=f"Reply received, but native context was not saved: {error}",
                               reasoning_content=reasoning_content)

    def chat(self, conversation_id: str, user_text: str, max_tokens: int = 2048, *,
             current_image_prompt: str | None = None,
             current_aspect_ratio: str | None = None,
             reasoning_effort: str | None = None) -> ChatOutcome:
        conversation_id, user_text, messages = self._build_chat_messages(
            conversation_id, user_text, current_image_prompt, current_aspect_ratio
        )
        self._enter()
        try:
            self._prepare_strata(conversation_id)
            response = self.strata.chat(messages, max_tokens=max_tokens,
                                        reasoning_effort=reasoning_effort)
            return self._finish_chat(conversation_id, user_text, response)
        finally:
            self._leave()

    def chat_stream(self, conversation_id: str, user_text: str, max_tokens: int = 2048, *,
                    current_image_prompt: str | None = None,
                    current_aspect_ratio: str | None = None,
                    reasoning_effort: str | None = None) -> Iterator[dict[str, Any]]:
        conversation_id, user_text, messages = self._build_chat_messages(
            conversation_id, user_text, current_image_prompt, current_aspect_ratio
        )
        self._enter()
        stream = None
        try:
            self._prepare_strata(conversation_id)
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            stream = self.strata.chat_stream(messages, max_tokens=max_tokens,
                                             reasoning_effort=reasoning_effort)
            try:
                for delta in stream:
                    content = delta.get("content", "")
                    reasoning = delta.get("reasoning_content", "")
                    if not isinstance(content, str) or not isinstance(reasoning, str):
                        raise CoordinationError("Strata returned non-text streaming content")
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
            outcome = self._finish_chat(conversation_id, user_text, response)
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
                 progress_callback: Callable[[dict[str, Any]], None] | None = None) -> GenerationOutcome:
        conversation_id = self.store.validate_id(conversation_id)
        if self.store.get_conversation(conversation_id) is None:
            raise CoordinationError("conversation does not exist")
        try:
            workflow = build_workflow(self.settings, prompt, seed, aspect_ratio, megapixels,
                                      sampler_name, scheduler)
        except WorkflowError as error:
            raise CoordinationError(str(error)) from error

        self._enter()
        generation_id: str | None = None
        prompt_id: str | None = None
        completed_image: dict[str, str] | None = None
        failure: Exception | None = None
        progress_socket = None
        try:
            self._report_generation_event(progress_callback, {
                "type": "phase", "message": "Preparing GPU resources for ComfyUI",
            })
            self._require_idle_queue()
            if self._strata_loaded():
                active_id = self.store.active_conversation_id()
                if active_id is None:
                    raise CoordinationError("Strata is loaded but its active conversation is unknown")
                self._save_session(active_id)
                self.strata.unload(wait_s=300)
                if self._strata_loaded():
                    raise CoordinationError("Strata did not unload; image generation was not started")
            if self.store.messages(conversation_id) and not self.store.context_is_current(conversation_id):
                raise CoordinationError("this conversation has no current native Strata session")
            self._release_comfy()
            generation = self.settings.section("generation")
            selected_ratio = aspect_ratio or generation["default_aspect_ratio"]
            selected_megapixels = megapixels if megapixels is not None else generation["default_megapixels"]
            selected_sampler = sampler_name or generation["default_sampler"]
            selected_scheduler = scheduler or generation["default_scheduler"]
            width = workflow["4"]["inputs"]["width"]
            height = workflow["4"]["inputs"]["height"]
            generation_id = self.store.create_generation(
                conversation_id, prompt.strip(), seed, selected_ratio, selected_megapixels,
                width, height, selected_sampler, selected_scheduler,
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
            self.store.update_generation(generation_id, "running", prompt_id=prompt_id)
            self._report_generation_event(progress_callback, {
                "type": "phase", "message": "ComfyUI is executing the workflow",
            })
            completed_image = self._wait_for_generation(prompt_id, workflow, progress_socket,
                                                       progress_callback)
            self.store.update_generation(generation_id, "complete", image=completed_image)
        except Exception as error:
            failure = error
            if generation_id is not None:
                try:
                    self.store.update_generation(generation_id, "failed", prompt_id=prompt_id,
                                                 error=str(error)[:1000])
                except StorageError:
                    pass
        cleanup_warning = None
        self._report_generation_event(progress_callback, {
            "type": "phase", "message": "Releasing ComfyUI memory",
        })
        try:
            self._require_idle_queue()
            self.comfy.free_memory()
            self._wait_for_free_vram()
        except Exception as error:
            cleanup_warning = f"GPU cleanup is not confirmed; Strata remains blocked: {error}"
        finally:
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
        return GenerationOutcome(generation_id, prompt_id, completed_image, cleanup_warning)