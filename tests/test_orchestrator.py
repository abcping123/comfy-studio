from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from strata_comfy_studio.clients import BackendError
from strata_comfy_studio.orchestrator import CoordinationError, Orchestrator
from strata_comfy_studio.settings import Settings, load_config, project_root
from strata_comfy_studio.storage import ConversationStore, StorageError


class FakeStrata:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.loaded = False
        self.calls: list[tuple[str, Any]] = []
        self.fail_save = False
        self.aspect_ratio = "1:1 (Square)"
        self.reasoning_content = ""

    def status(self) -> dict:
        self.calls.append(("status", None))
        return {"loaded": self.loaded}

    def load(self) -> dict:
        self.calls.append(("load", None))
        self.loaded = True
        return {"status": "loaded", "loaded": True}

    def unload(self, wait_s: int = 300) -> dict:
        self.calls.append(("unload", wait_s))
        self.loaded = False
        return {"status": "unloaded", "loaded": False}

    def save_session(self, filename: str) -> dict:
        self.calls.append(("save", filename))
        if self.fail_save:
            raise BackendError("Strata", "save failed", status=507)
        path = self.settings.path("paths.strata_sessions") / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"native-context")
        return {"n_saved": 20, "n_written": 14}

    def restore_session(self, filename: str) -> dict:
        self.calls.append(("restore", filename))
        return {"n_restored": 20, "n_read": 14}

    def chat(self, messages: list[dict], max_tokens: int = 2048,
             reasoning_effort: str | None = None) -> dict:
        self.calls.append(("chat", messages))
        return {"choices": [{"message": {
            "content": json.dumps({
                "reply": "A carefully described image.",
                "image_prompt": "A red teapot in a soft studio light.",
                "aspect_ratio": self.aspect_ratio,
            }),
            "reasoning_content": self.reasoning_content,
        }}]}

    def chat_stream(self, messages: list[dict], max_tokens: int = 2048,
                    reasoning_effort: str | None = None):
        self.reasoning_effort = reasoning_effort
        self.calls.append(("chat_stream", messages))
        content = json.dumps({
            "reply": "A carefully described image.",
            "image_prompt": "A red teapot in a soft studio light.",
            "aspect_ratio": self.aspect_ratio,
        })
        yield {"content": "", "reasoning_content": self.reasoning_content[:12]}
        yield {"content": content[:10], "reasoning_content": self.reasoning_content[12:]}
        yield {"content": content[10:], "reasoning_content": ""}


class FakeProgressSocket:
    def __init__(self, messages: list[str]):
        self.messages = messages
        self.closed = False

    def recv(self, timeout: float = 0) -> str:
        if self.messages:
            return self.messages.pop(0)
        raise TimeoutError

    def close(self) -> None:
        self.closed = True


class FakeComfy:
    def __init__(self):
        self.running: list[Any] = []
        self.pending: list[Any] = []
        self.calls: list[str] = []
        self.submitted: dict | None = None
        self.client_id: str | None = None
        self.websocket_client_id: str | None = None
        self.progress_messages: list[str] = []
        self.progress_socket: FakeProgressSocket | None = None
        self.fail_progress_connection = False
        self.free_vram_mib = 15000
        self.saved_subfolder = ""
        self.input_directory: Path | None = None
        self.output_directory: Path | None = None

    def queue(self) -> dict:
        self.calls.append("queue")
        return {"queue_running": self.running, "queue_pending": self.pending}

    def free_memory(self) -> dict:
        self.calls.append("free")
        return {}

    def system_stats(self) -> dict:
        self.calls.append("stats")
        return {"devices": [{"index": 0, "vram_free": self.free_vram_mib * 1024 * 1024}]}

    def open_progress_socket(self, client_id: str) -> FakeProgressSocket:
        self.calls.append("websocket")
        if self.fail_progress_connection:
            raise OSError("websocket unavailable")
        self.websocket_client_id = client_id
        self.progress_socket = FakeProgressSocket(self.progress_messages.copy())
        return self.progress_socket

    def submit_prompt(self, workflow: dict, client_id: str | None = None) -> dict:
        self.calls.append("submit")
        self.submitted = workflow
        self.saved_subfolder = workflow["8"]["inputs"]["filename_prefix"].split("/", 1)[0]
        self.client_id = client_id
        return {"prompt_id": "prompt-123"}

    def history(self, prompt_id: str) -> dict:
        self.calls.append("history")
        if self.output_directory is None:
            raise RuntimeError("output directory is not configured")
        output_folder = self.output_directory / self.saved_subfolder
        output_folder.mkdir(parents=True, exist_ok=True)
        (output_folder / "generated.png").write_bytes(b"generated image")
        return {prompt_id: {
            "status": {"status_str": "success", "completed": True},
            "outputs": {"8": {"images": [{
                "filename": "generated.png", "subfolder": self.saved_subfolder, "type": "output",
            }]}},
        }}

    def upload_image(self, filename: str, content: bytes, content_type: str,
                     subfolder: str) -> dict:
        if self.input_directory is None:
            raise RuntimeError("input directory is not configured")
        directory = self.input_directory / subfolder
        directory.mkdir(parents=True, exist_ok=True)
        (directory / filename).write_bytes(content)
        self.calls.append("upload")
        return {"name": filename, "subfolder": subfolder, "type": "input"}


class OrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        data = copy.deepcopy(load_config(project_root() / "data" / "config.example.json").data)
        data["strata"]["min_free_vram_mib"] = 12000
        self.temp = tempfile.TemporaryDirectory()
        data["paths"]["strata_sessions"] = str(Path(self.temp.name) / "sessions")
        data["paths"]["temp_root"] = str(Path(self.temp.name) / "temp")
        data["comfyui"]["output_directory"] = str(Path(self.temp.name) / "ComfyUI" / "output")
        data["comfyui"]["input_directory"] = str(Path(self.temp.name) / "temp" / "comfyui" / "input")
        self.settings = Settings(project_root(), data)
        self.store = ConversationStore(Path(self.temp.name) / "studio.sqlite3")
        self.strata = FakeStrata(self.settings)
        self.comfy = FakeComfy()
        self.comfy.input_directory = self.settings.path("comfyui.input_directory")
        self.comfy.output_directory = self.settings.path("comfyui.output_directory")
        self.fake_time = [0.0]
        self.orchestrator = Orchestrator(self.settings, self.store, self.strata, self.comfy,
                                         clock=lambda: self.fake_time[0],
                                         sleeper=lambda seconds: self.fake_time.__setitem__(
                                             0, self.fake_time[0] + seconds))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_chat_generation_and_chat_restore_native_context(self) -> None:
        conversation_id = self.store.create_conversation()["id"]
        first = self.orchestrator.chat(conversation_id, "Design a red teapot")
        self.assertTrue(first.context_saved)
        self.assertEqual(first.image_prompt, "A red teapot in a soft studio light.")
        self.assertEqual(first.aspect_ratio, "1:1 (Square)")
        self.assertTrue(self.store.context_is_current(conversation_id))

        result = self.orchestrator.generate(conversation_id, "A red teapot on white", 42)
        self.assertEqual(result.image["filename"], f"{result.generation_id}.png")
        self.assertFalse(self.strata.loaded)

        second = self.orchestrator.chat(conversation_id, "Make the light warmer")
        self.assertTrue(second.context_saved)
        calls = [name for name, _ in self.strata.calls]
        restore_index = calls.index("restore")
        final_chat_index = len(calls) - 2
        self.assertLess(restore_index, final_chat_index)
        self.assertEqual(self.strata.calls[restore_index][1], self.store.session_filename(conversation_id))
        replayed_messages = self.strata.calls[final_chat_index][1]
        self.assertEqual(replayed_messages[0]["role"], "system")
        self.assertEqual(replayed_messages[-1]["role"], "user")
        self.assertIn("user_request:\nMake the light warmer", replayed_messages[-1]["content"])
        self.assertIn("submit", self.comfy.calls)

    def test_generation_reports_correlated_comfy_progress(self) -> None:
        conversation_id = self.store.create_conversation()["id"]
        self.comfy.progress_messages = [
            *[
                json.dumps({"type": "executing", "data": {
                    "prompt_id": "prompt-123", "node": node,
                }})
                for node in ("3", "2", "4", "5", "1", "6")
            ],
            json.dumps({"type": "progress", "data": {
                "prompt_id": "prompt-123", "node": "6", "value": 3, "max": 6,
            }}),
            json.dumps({"type": "progress", "data": {
                "prompt_id": "another-prompt", "node": "6", "value": 5, "max": 6,
            }}),
            json.dumps({"type": "executing", "data": {"prompt_id": "prompt-123", "node": "7"}}),
            json.dumps({"type": "executing", "data": {"prompt_id": "prompt-123", "node": "8"}}),
            json.dumps({"type": "executed", "data": {"prompt_id": "prompt-123", "node": "8"}}),
        ]
        events: list[dict] = []

        self.orchestrator.generate(conversation_id, "A red teapot", 42,
                                   progress_callback=events.append)

        sampler_progress = next(event for event in events if event["type"] == "progress")
        self.assertEqual((sampler_progress["value"], sampler_progress["max"]), (3, 6))
        self.assertEqual(sampler_progress["label"], "KSampler")
        self.assertEqual((sampler_progress["completed_nodes"], sampler_progress["total_nodes"]), (5, 8))
        self.assertEqual((sampler_progress["completed_units"], sampler_progress["total_units"]), (5, 13))
        self.assertAlmostEqual(sampler_progress["overall_percent"], 800 / 13)
        sampler_start = next(event for event in events
                     if event.get("type") == "node" and event.get("state") == "started"
                     and event.get("node") == "6")
        self.assertEqual(sampler_start["message"], "Running KSampler")
        decode_start = next(event for event in events
                    if event.get("type") == "node" and event.get("state") == "started"
                    and event.get("node") == "7")
        self.assertEqual(decode_start["completed_units"], 11)
        self.assertAlmostEqual(decode_start["overall_percent"], 1100 / 13)
        save_complete = next(event for event in events
                     if event.get("type") == "node" and event.get("state") == "complete"
                     and event.get("node") == "8")
        self.assertEqual(save_complete["completed_units"], 13)
        self.assertEqual(save_complete["overall_percent"], 99)
        self.assertTrue(self.comfy.client_id)
        self.assertEqual(self.comfy.client_id, self.comfy.websocket_client_id)
        self.assertTrue(self.comfy.progress_socket.closed)

    def test_generation_keeps_comfy_models_until_strata_handoff(self) -> None:
        conversation_id = self.store.create_conversation()["id"]

        result = self.orchestrator.generate(conversation_id, "A red teapot", 42)

        self.assertEqual(result.image["filename"], f"{result.generation_id}.png")
        self.assertEqual(self.comfy.calls.count("free"), 1)
        self.assertIsNone(result.cleanup_warning)

        self.orchestrator.generate(conversation_id, "A blue vase", 43)

        self.assertEqual(self.comfy.calls.count("free"), 1)

        self.orchestrator.chat(conversation_id, "Adjust the lighting")

        self.assertEqual(self.comfy.calls.count("free"), 2)

    def test_generation_falls_back_when_progress_socket_is_unavailable(self) -> None:
        conversation_id = self.store.create_conversation()["id"]
        self.comfy.fail_progress_connection = True
        events: list[dict] = []

        result = self.orchestrator.generate(conversation_id, "A red teapot", 42,
                                            progress_callback=events.append)

        self.assertEqual(result.image["filename"], f"{result.generation_id}.png")
        self.assertIn("submit", self.comfy.calls)
        self.assertTrue(any(event.get("message", "").startswith("Live progress unavailable")
                            for event in events))

    def test_generation_routes_image_to_conversation_directory(self) -> None:
        conversation_id = self.store.create_conversation()["id"]

        result = self.orchestrator.generate(conversation_id, "A red teapot", 42)

        self.assertEqual(
            self.comfy.submitted["8"]["inputs"]["filename_prefix"],
            f"{conversation_id}/image",
        )
        self.assertEqual(result.image["subfolder"], conversation_id)
        studio_image = (
            self.settings.path("paths.temp_root") / "comfyui" / conversation_id
            / f"{result.generation_id}.png"
        )
        comfy_image = self.settings.path("comfyui.output_directory") / conversation_id / "generated.png"
        self.assertTrue(studio_image.is_file())
        self.assertEqual(studio_image.read_bytes(), b"generated image")
        self.assertFalse(comfy_image.exists())

    def test_edit_generation_uploads_ordered_sources_and_persists_provenance(self) -> None:
        conversation_id = self.store.create_conversation()["id"]
        self.store.set_conversation_mode(conversation_id, "edit")
        base = self.orchestrator.upload_source_image(
            conversation_id, b"base-image", "image/png", ".png", 1920, 1080,
        )
        reference = self.orchestrator.upload_source_image(
            conversation_id, b"reference-image", "image/jpeg", ".jpg", 800, 1200,
        )

        result = self.orchestrator.generate(
            conversation_id, "Edit <Picture 1> using <Picture 2>.", 42,
            megapixels=4.0, image_mode="edit",
            source_image_ids=[base["id"], reference["id"]],
        )

        workflow = self.comfy.submitted
        self.assertEqual(workflow["9"]["inputs"]["image"], f"{conversation_id}/{base['filename']}")
        self.assertEqual(workflow["13"]["inputs"]["image"],
                         f"{conversation_id}/{reference['filename']}")
        self.assertEqual(workflow["4"]["inputs"]["images.image_2"], ["13", 0])
        generation = self.store.generation(result.generation_id)
        self.assertEqual(generation["image_mode"], "edit")
        self.assertEqual(generation["source_image_ids"], [base["id"], reference["id"]])
        self.assertEqual([name for name in self.comfy.calls if name == "upload"], ["upload", "upload"])
        with self.assertRaisesRegex(StorageError, "used by a generation"):
            self.orchestrator.delete_source_image(conversation_id, base["id"])
        self.assertTrue(
            (self.settings.path("comfyui.input_directory") / conversation_id / base["filename"]).is_file()
        )

    def test_removing_unused_source_image_deletes_its_uploaded_file(self) -> None:
        conversation_id = self.store.create_conversation()["id"]
        source = self.orchestrator.upload_source_image(
            conversation_id, b"base-image", "image/png", ".png", 100, 100,
        )
        image_path = self.settings.path("comfyui.input_directory") / conversation_id / source["filename"]

        self.orchestrator.delete_source_image(conversation_id, source["id"])

        self.assertFalse(image_path.exists())
        self.assertIsNone(self.store.source_image(conversation_id, source["id"]))

    def test_edit_chat_forwards_ordered_images_and_keeps_source_ratio(self) -> None:
        conversation_id = self.store.create_conversation()["id"]
        self.store.set_conversation_mode(conversation_id, "edit")
        base = self.orchestrator.upload_source_image(
            conversation_id, b"base-image", "image/png", ".png", 1920, 1080,
        )
        reference = self.orchestrator.upload_source_image(
            conversation_id, b"reference-image", "image/jpeg", ".jpg", 800, 600,
        )
        self.strata.aspect_ratio = "4:3 (Standard)"

        result = self.orchestrator.chat(
            conversation_id, "Keep the composition and change the colors.",
            current_aspect_ratio="16:9 (Widescreen)",
        )

        self.assertEqual(result.aspect_ratio, "16:9 (Widescreen)")
        messages = self.strata.calls[-2][1]
        message_content = messages[-1]["content"]
        self.assertIsInstance(message_content, list)
        self.assertIn('"source_image_count": 2', message_content[0]["text"])
        self.assertIn("<Picture 1> (base image)", message_content[0]["text"])
        self.assertIn("<Picture 2> (reference image)", message_content[0]["text"])
        image_urls = [
            part["image_url"]["url"] for part in message_content if part["type"] == "image_url"
        ]
        self.assertEqual(image_urls, [
            str(self.settings.path("comfyui.input_directory") / conversation_id / base["filename"]),
            str(self.settings.path("comfyui.input_directory") / conversation_id / reference["filename"]),
        ])
        self.assertNotIn("不会收到上传图片的像素内容", messages[0]["content"])

    def test_edit_chat_fails_when_an_uploaded_image_file_is_missing(self) -> None:
        conversation_id = self.store.create_conversation()["id"]
        self.store.set_conversation_mode(conversation_id, "edit")
        source = self.orchestrator.upload_source_image(
            conversation_id, b"base-image", "image/png", ".png", 100, 100,
        )
        image_path = self.settings.path("comfyui.input_directory") / conversation_id / source["filename"]
        image_path.unlink()

        with self.assertRaisesRegex(CoordinationError, "source image file is missing"):
            self.orchestrator.chat(conversation_id, "Describe and edit the image")

        self.assertNotIn("chat", [name for name, _ in self.strata.calls])

    def test_deleting_conversation_removes_sources_but_keeps_generated_output_folder(self) -> None:
        conversation_id = self.store.create_conversation()["id"]
        source = self.orchestrator.upload_source_image(
            conversation_id, b"base-image", "image/png", ".png", 100, 100,
        )
        source_path = self.settings.path("comfyui.input_directory") / conversation_id / source["filename"]
        output_folder = self.settings.path("paths.temp_root") / "comfyui" / conversation_id
        output_folder.mkdir(parents=True, exist_ok=True)
        output_image = output_folder / "image_00001_.png"
        output_image.write_bytes(b"generated")

        warning = self.orchestrator.delete_conversation(conversation_id)

        self.assertIsNone(warning)
        self.assertFalse(source_path.exists())
        self.assertEqual(output_image.read_bytes(), b"generated")

    def test_chat_includes_current_editor_prompt_and_aspect_ratio(self) -> None:
        conversation_id = self.store.create_conversation()["id"]
        self.strata.aspect_ratio = "16:9 (Widescreen)"

        result = self.orchestrator.chat(
            conversation_id,
            "Make the lighting warmer",
            current_image_prompt="A red teapot on a dark green table.",
            current_aspect_ratio="16:9 (Widescreen)",
        )

        self.assertEqual(result.aspect_ratio, "16:9 (Widescreen)")
        chat_messages = next(messages for name, messages in self.strata.calls if name == "chat")
        self.assertIn("current_editor_state", chat_messages[0]["content"])
        state_json = chat_messages[-1]["content"].split("current_editor_state:\n", 1)[1]
        state_json = state_json.split("\n\nuser_request:\n", 1)[0]
        editor_state = json.loads(state_json)
        self.assertEqual(editor_state["current_image_prompt"], "A red teapot on a dark green table.")
        self.assertEqual(editor_state["current_aspect_ratio"], "16:9 (Widescreen)")
        self.assertIn("Make the lighting warmer", chat_messages[-1]["content"])

    def test_chat_persists_and_replays_strata_reasoning(self) -> None:
        conversation_id = self.store.create_conversation()["id"]
        self.strata.reasoning_content = "First compare the requested layout and aspect ratio."

        result = self.orchestrator.chat(conversation_id, "Design a red teapot")

        self.assertEqual(result.reasoning_content, self.strata.reasoning_content)
        assistant_message = self.store.display_messages(conversation_id)[-1]
        self.assertEqual(assistant_message["reasoning_content"], self.strata.reasoning_content)

        self.strata.reasoning_content = "Now keep the same composition."
        self.orchestrator.chat(conversation_id, "Make the light warmer")
        chat_calls = [messages for name, messages in self.strata.calls if name == "chat"]
        self.assertEqual(chat_calls[1][-2]["reasoning_content"],
                         "First compare the requested layout and aspect ratio.")

    def test_chat_stream_emits_reasoning_and_persists_final_turn(self) -> None:
        conversation_id = self.store.create_conversation()["id"]
        self.strata.reasoning_content = "Compare the requested subject and composition."

        events = list(self.orchestrator.chat_stream(
            conversation_id,
            "Design a red teapot",
            current_image_prompt="A blue vase.",
            current_aspect_ratio="16:9 (Widescreen)",
            reasoning_effort="low",
        ))

        self.assertEqual("".join(event["text"] for event in events if event["type"] == "reasoning"),
                         self.strata.reasoning_content)
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["answer"], "A carefully described image.")
        self.assertEqual(self.strata.reasoning_effort, "low")
        self.assertEqual(done["reasoning_content"], self.strata.reasoning_content)
        saved = self.store.display_messages(conversation_id)[-1]
        self.assertEqual(saved["reasoning_content"], self.strata.reasoning_content)
        chat_messages = next(messages for name, messages in self.strata.calls if name == "chat_stream")
        self.assertIn("current_image_prompt", chat_messages[-1]["content"])

    def test_closing_chat_stream_releases_operation_lock(self) -> None:
        conversation_id = self.store.create_conversation()["id"]
        self.strata.reasoning_content = "Stream a partial thought."
        stream = self.orchestrator.chat_stream(conversation_id, "Design a red teapot")
        self.assertEqual(next(stream), {"type": "reasoning", "text": "Stream a par"})
        stream.close()

        result = self.orchestrator.chat(conversation_id, "Try again")
        self.assertTrue(result.context_saved)

    def test_external_queue_work_blocks_model_switch(self) -> None:
        conversation_id = self.store.create_conversation()["id"]
        self.comfy.pending = [[0, "someone-else", {}, {}, []]]
        with self.assertRaisesRegex(CoordinationError, "queue is busy"):
            self.orchestrator.chat(conversation_id, "Do not interrupt other work")
        self.assertEqual(self.strata.calls, [])
        self.assertNotIn("free", self.comfy.calls)

    def test_loaded_strata_session_does_not_repeat_load_floor_check(self) -> None:
        conversation_id = self.store.create_conversation()["id"]
        self.orchestrator.chat(conversation_id, "First turn")
        self.comfy.free_vram_mib = 1024
        second = self.orchestrator.chat(conversation_id, "Continue the same context")
        self.assertTrue(second.context_saved)
        self.assertEqual(sum(name == "load" for name, _ in self.strata.calls), 1)

    def test_unsupported_strata_ratio_is_normalized_and_explained(self) -> None:
        conversation_id = self.store.create_conversation()["id"]
        self.strata.aspect_ratio = "5:7"
        result = self.orchestrator.chat(conversation_id, "Design a 5:7 poster")
        self.assertEqual(result.aspect_ratio, "3:4 (Portrait Standard)")
        self.assertIn("5:7", result.answer)
        self.assertIn("3:4 (Portrait Standard)", result.answer)
        self.assertEqual(self.store.get_conversation(conversation_id)["aspect_ratio"],
                         "3:4 (Portrait Standard)")
        self.assertIn('"aspect_ratio": "5:7"', self.store.messages(conversation_id)[-1]["content"])

    def test_ui_selection_does_not_change_loaded_strata_context_identity(self) -> None:
        first_id = self.store.create_conversation("First") ["id"]
        second_id = self.store.create_conversation("Second") ["id"]
        self.orchestrator.chat(first_id, "Keep this as a separate context")
        self.store.set_selected_conversation_id(second_id)
        self.assertEqual(self.store.selected_conversation_id(), second_id)
        self.assertEqual(self.store.active_conversation_id(), first_id)

        self.orchestrator.chat(second_id, "Switch to the second conversation")

        calls = [name for name, _ in self.strata.calls]
        self.assertIn("unload", calls)
        self.assertEqual(self.store.active_conversation_id(), second_id)

    def test_deleting_loaded_conversation_unloads_and_removes_native_session(self) -> None:
        conversation_id = self.store.create_conversation()["id"]
        self.orchestrator.chat(conversation_id, "Create a native session")
        session_path = self.settings.path("paths.strata_sessions") / self.store.session_filename(conversation_id)
        self.assertTrue(session_path.is_file())
        self.orchestrator.delete_conversation(conversation_id)
        self.assertFalse(self.strata.loaded)
        self.assertFalse(session_path.exists())
        self.assertIsNone(self.store.get_conversation(conversation_id))
        self.assertIsNone(self.store.active_conversation_id())

    def test_failed_session_save_prevents_unload_and_generation(self) -> None:
        conversation_id = self.store.create_conversation()["id"]
        self.orchestrator.chat(conversation_id, "Keep the native context")
        self.strata.fail_save = True
        with self.assertRaisesRegex(CoordinationError, "save failed"):
            self.orchestrator.generate(conversation_id, "A red teapot", 1)
        self.assertTrue(self.strata.loaded)
        self.assertNotIn("submit", self.comfy.calls)

    def test_unknown_free_vram_threshold_fails_before_loading_strata(self) -> None:
        self.settings.data["strata"]["min_free_vram_mib"] = None
        conversation_id = self.store.create_conversation()["id"]
        with self.assertRaisesRegex(CoordinationError, "min_free_vram_mib"):
            self.orchestrator.chat(conversation_id, "Wait for a configured safety floor")
        self.assertFalse(self.strata.loaded)
        self.assertFalse(any(name == "load" for name, _ in self.strata.calls))


if __name__ == "__main__":
    unittest.main()