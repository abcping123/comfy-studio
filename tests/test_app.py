from __future__ import annotations

import json
import io
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from comfy_studio.chat_backends import CHAT_BACKEND_REGISTRY, ChatBackendDefinition
from comfy_studio.app import ChatRunManager, GenerationRunManager, create_app
from comfy_studio.modes import DEFAULT_MODE_ID, IMAGE_MODES
from comfy_studio.orchestrator import ChatOutcome, GenerationOutcome
from comfy_studio.settings import Settings
from comfy_studio.storage import ConversationStore
from test_settings import sample_config


class AppStrata:
    def status(self) -> dict:
        return {"loaded": False}


class AppComfy:
    def __init__(self, settings: Settings):
        self.settings = settings

    def queue(self) -> dict:
        return {"queue_running": [], "queue_pending": []}

    def system_stats(self) -> dict:
        return {"devices": [{"index": 0, "vram_free": 12 * 1024 * 1024 * 1024}]}

    def upload_image(self, filename: str, content: bytes, content_type: str,
                     subfolder: str) -> dict:
        directory = self.settings.path("comfyui.input_directory") / subfolder
        directory.mkdir(parents=True, exist_ok=True)
        (directory / filename).write_bytes(content)
        return {"name": filename, "subfolder": subfolder, "type": "input"}

    def image(self, filename: str, subfolder: str, image_type: str) -> tuple[bytes, str]:
        path = self.settings.path("comfyui.input_directory") / subfolder / filename
        return path.read_bytes(), "image/png"


class AppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = sample_config()
        self.data["app"]["access_token"] = "test-access-token"
        self.settings = Settings(self.root, self.data)
        self.store = ConversationStore(
            self.settings.path("paths.session_root"),
            self.settings.path("paths.runtime_root"),
        )
        self.app = create_app(self.settings, self.store, AppStrata(), AppComfy(self.settings))
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.client.close()
        self.temp.cleanup()

    def test_static_shell_is_public_but_api_requires_bearer_token(self) -> None:
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Fieldnotes", page.text)
        self.assertIn('id="delete-dialog"', page.text)
        self.assertIn('id="image-mode-switch"', page.text)
        self.assertNotIn('id="image-sampler"', page.text)
        self.assertNotIn('id="image-scheduler"', page.text)
        self.assertNotIn("DIRECT PROMPT", page.text)
        self.assertNotIn("Enhancement off", page.text)
        self.assertEqual(self.client.get("/api/state").status_code, 401)
        response = self.client.get("/api/state", headers={"Authorization": "Bearer test-access-token"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["services"]["comfyui"]["vram_free_mib"], 12288)

    def test_wrong_bearer_token_is_rejected(self) -> None:
        response = self.client.get("/api/state", headers={"Authorization": "Bearer wrong"})
        self.assertEqual(response.status_code, 401)

    def test_system_metrics_requires_auth_and_returns_utilization(self) -> None:
        path = "/api/system-metrics"
        self.assertEqual(self.client.get(path).status_code, 401)
        metrics = {
            "cpu_percent": 21.5,
            "memory_percent": 48.0,
            "gpu_percent": 12.0,
            "vram_percent": 36.0,
        }
        with patch("comfy_studio.app.SystemMetrics.snapshot", return_value=metrics):
            response = self.client.get(path, headers={"Authorization": "Bearer test-access-token"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), metrics)

    def test_options_endpoint_exposes_workflow_resolution_controls(self) -> None:
        response = self.client.get("/api/options", headers={"Authorization": "Bearer test-access-token"})
        self.assertEqual(response.status_code, 200)
        options = response.json()
        self.assertEqual(options["default_mode"], DEFAULT_MODE_ID)
        self.assertEqual(options["reasoning_efforts"], [
            {"value": "none", "label": "Off"},
            {"value": "low", "label": "Low"},
            {"value": "medium", "label": "Medium"},
            {"value": "high", "label": "High"},
        ])
        self.assertEqual(options["default_reasoning_effort"], "none")
        self.assertEqual([mode["id"] for mode in options["modes"]],
                         [mode.id for mode in IMAGE_MODES])
        edit_mode = next(mode for mode in options["modes"] if mode["id"] == "edit")
        text_to_image_mode = next(
            mode for mode in options["modes"] if mode["id"] == DEFAULT_MODE_ID
        )
        self.assertEqual(edit_mode["resolution"]["strategy"], "source-image")
        self.assertEqual(edit_mode["resolution"]["megapixels"]["max"], 16.0)
        self.assertEqual(edit_mode["source_images"]["min_count"], 1)
        self.assertEqual(edit_mode["source_images"]["max_count"], 5)
        self.assertEqual(options["default_aspect_ratio"], "1:1 (Square)")
        self.assertEqual(text_to_image_mode["resolution"]["megapixels"], {
            "min": 0.5,
            "max": 20.0,
            "step": 0.01,
            "default": 4.0,
        })
        self.assertNotIn("megapixels", options)
        self.assertNotIn("edit_megapixels", options)
        self.assertEqual(options["source_images"]["max_count"], 5)
        self.assertEqual(options["resolution_multiple"], 32)
        self.assertNotIn("samplers", options)
        self.assertNotIn("default_sampler", options)
        self.assertNotIn("schedulers", options)
        self.assertNotIn("default_scheduler", options)
        self.assertFalse(options["prompt_enhancement"])

    def test_bonsai_options_and_chat_use_supported_reasoning_efforts(self) -> None:
        settings = Settings(self.root, self.data, chat_backend="bonsai")
        app = create_app(settings, self.store, AppStrata(), AppComfy(settings))
        headers = {"Authorization": "Bearer test-access-token"}
        with TestClient(app) as client:
            options = client.get("/api/options", headers=headers).json()
            self.assertEqual(options["reasoning_efforts"], [
                {"value": "none", "label": "Off"},
                {"value": "medium", "label": "Medium"},
                {"value": "xhigh", "label": "XHigh"},
            ])
            self.assertEqual(options["default_reasoning_effort"], "none")

            conversation = client.post("/api/conversations", headers=headers).json()
            with patch(
                "comfy_studio.app.Orchestrator.chat",
                autospec=True,
                return_value=ChatOutcome("Reply", "Prompt", "1:1 (Square)", True),
            ) as chat:
                response = client.post("/api/chat", headers=headers, json={
                    "conversation_id": conversation["id"],
                    "message": "Hello",
                })
                self.assertEqual(response.status_code, 200)
                self.assertEqual(chat.call_args.kwargs["reasoning_effort"], "none")

                for effort in ("none", "medium", "xhigh"):
                    response = client.post("/api/chat", headers=headers, json={
                        "conversation_id": conversation["id"],
                        "message": "Hello",
                        "reasoning_effort": effort,
                    })
                    self.assertEqual(response.status_code, 200)

                for effort in ("low", "high"):
                    response = client.post("/api/chat", headers=headers, json={
                        "conversation_id": conversation["id"],
                        "message": "Hello",
                        "reasoning_effort": effort,
                    })
                    self.assertEqual(response.status_code, 422)

                self.assertEqual(chat.call_count, 4)

    def test_registered_backend_controls_reasoning_options_without_api_changes(self) -> None:
        definition = ChatBackendDefinition(
            id="future-chat",
            label="Future Chat",
            reasoning_efforts=("none", "ultra_low"),
            default_reasoning_effort="none",
            config_schema="strata",
            client_factory=lambda _settings: AppStrata(),
            managed_service=False,
            service_adapter=None,
            setup_adapter="preconfigured",
            runtime_preparer=None,
            loaded_state="stateless",
            endpoint_probe=None,
            chat_preparation="stateless",
            context_strategy="transcript",
            image_transport="none",
            generation_handoff="none",
        )
        settings = Settings(self.root, self.data, chat_backend=definition.id)
        app = create_app(settings, self.store, AppStrata(), AppComfy(settings))
        headers = {"Authorization": f"Bearer {self.data['app']['access_token']}"}

        with patch.dict(CHAT_BACKEND_REGISTRY, {definition.id: definition}), TestClient(app) as client:
            options_response = client.get("/api/options", headers=headers)
            self.assertEqual(options_response.status_code, 200, options_response.text)
            options = options_response.json()
            self.assertEqual(options["chat_backend"], {"id": "future-chat", "label": "Future Chat"})
            self.assertEqual(options["reasoning_efforts"], [
                {"value": "none", "label": "Off"},
                {"value": "ultra_low", "label": "Ultra Low"},
            ])

            conversation = client.post("/api/conversations", headers=headers).json()
            with patch(
                "comfy_studio.app.Orchestrator.chat",
                autospec=True,
                return_value=ChatOutcome("Reply", "Prompt", "1:1 (Square)", True),
            ) as chat:
                response = client.post("/api/chat", headers=headers, json={
                    "conversation_id": conversation["id"],
                    "message": "Hello",
                    "reasoning_effort": "ultra_low",
                })
            self.assertEqual(response.status_code, 200)
            self.assertEqual(chat.call_args.kwargs["reasoning_effort"], "ultra_low")

    def test_generate_stream_rejects_megapixels_outside_half_to_twenty(self) -> None:
        headers = {"Authorization": "Bearer test-access-token"}
        for megapixels in (0.49, 20.01):
            response = self.client.post("/api/generate/stream", headers=headers, json={
                "conversation_id": "invalid-for-this-request",
                "prompt": "A test image",
                "seed": 1,
                "megapixels": megapixels,
            })
            with self.subTest(megapixels=megapixels):
                self.assertEqual(response.status_code, 422)

    def test_chat_forwards_current_editor_state(self) -> None:
        headers = {"Authorization": "Bearer test-access-token"}
        conversation = self.client.post("/api/conversations", headers=headers).json()
        editor_state = {
            "current_image_prompt": "A red teapot on a dark green table.",
            "current_aspect_ratio": "16:9 (Widescreen)",
        }
        with patch("comfy_studio.app.Orchestrator.chat", autospec=True,
                   return_value=ChatOutcome("Reply", "Prompt", "16:9 (Widescreen)", True,
                                            reasoning_content="Consider a wide composition.")) as chat:
            response = self.client.post("/api/chat", headers=headers, json={
                "conversation_id": conversation["id"],
                "message": "Make the lighting warmer",
                **editor_state,
            })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(chat.call_args.args[1:3], (conversation["id"], "Make the lighting warmer"))
        self.assertEqual(chat.call_args.kwargs, {**editor_state, "reasoning_effort": "none"})
        self.assertEqual(response.json()["reasoning_content"], "Consider a wide composition.")

    def test_chat_stream_emits_server_sent_events(self) -> None:
        headers = {"Authorization": "Bearer test-access-token"}
        conversation = self.client.post("/api/conversations", headers=headers).json()
        editor_state = {
            "current_image_prompt": "A blue glass vase.",
            "current_aspect_ratio": "16:9 (Widescreen)",
            "reasoning_effort": "high",
        }
        stream_events = [
            {"type": "reasoning", "text": "Keep the subject centered. "},
            {"type": "done", "answer": "Done.", "image_prompt": "A blue glass vase, centered.",
             "aspect_ratio": "16:9 (Widescreen)", "reasoning_content": "Keep the subject centered. ",
             "context_saved": True, "warning": None},
        ]
        with patch("comfy_studio.app.Orchestrator.chat_stream", autospec=True,
                   return_value=iter(stream_events)) as chat:
            response = self.client.post("/api/chat/stream", headers=headers, json={
                "conversation_id": conversation["id"],
                "message": "Make the frame wider",
                **editor_state,
            })

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertEqual(events, stream_events)
        resumed = self.client.get(
            f"/api/chat/runs/{response.headers['x-chat-run-id']}/stream",
            headers=headers,
        )
        replayed_events = [json.loads(line[6:]) for line in resumed.text.splitlines()
                           if line.startswith("data: ")]
        self.assertEqual(replayed_events, stream_events)
        self.assertEqual(chat.call_args.args[1:3], (conversation["id"], "Make the frame wider"))
        self.assertEqual(chat.call_args.kwargs, editor_state)

    def test_chat_run_can_be_replayed_after_a_subscriber_disconnects(self) -> None:
        manager = ChatRunManager()
        release = threading.Event()
        waiting = threading.Event()

        def operation():
            def events():
                yield {"type": "reasoning", "text": "Still thinking."}
                waiting.set()
                release.wait(timeout=2)
                yield {"type": "done", "answer": "Finished."}
            return events()

        run = manager.start("conversation", "Request", 0, operation)
        subscriber = run.stream()
        try:
            self.assertIn('"text": "Still thinking."', next(subscriber))
            self.assertTrue(waiting.wait(timeout=1))
            subscriber.close()
            self.assertEqual(manager.active(), run)
        finally:
            release.set()

        replayed = list(run.stream())
        self.assertEqual(len(replayed), 2)
        self.assertIn('"type": "done"', replayed[-1])
        self.assertIsNone(manager.active())

    def test_state_exposes_active_chat_for_page_recovery(self) -> None:
        headers = {"Authorization": "Bearer test-access-token"}
        conversation = self.client.post("/api/conversations", headers=headers).json()
        release = threading.Event()
        waiting = threading.Event()

        def operation():
            def events():
                yield {"type": "reasoning", "text": "Thinking."}
                waiting.set()
                release.wait(timeout=2)
                yield {"type": "done", "answer": "Finished."}
            return events()

        run = self.app.state.chat_runs.start(conversation["id"], "Request", 0, operation)
        try:
            self.assertTrue(waiting.wait(timeout=1))
            response = self.client.get("/api/state", headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["active_chat"], run.public_info())
        finally:
            release.set()
        self.assertEqual(len(list(run.stream())), 2)

    def test_state_exposes_active_generation_for_page_recovery(self) -> None:
        headers = {"Authorization": "Bearer test-access-token"}
        conversation = self.client.post("/api/conversations", headers=headers).json()
        run = self.app.state.generation_runs.start(conversation["id"])
        event = {"type": "progress", "overall_percent": 42}
        run.publish(event)

        response = self.client.get("/api/state", headers=headers)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["active_generation"], {
            "id": run.id,
            "conversation_id": conversation["id"],
            "sequence": 1,
            "latest_event": event,
        })
        self.app.state.generation_runs.finish(run, {"type": "error", "message": "test complete"})

    def test_generation_stream_can_resume_after_last_seen_event(self) -> None:
        headers = {"Authorization": "Bearer test-access-token"}
        run = self.app.state.generation_runs.start("conversation-id")
        run.publish({"type": "phase", "message": "Preparing ComfyUI"})
        sequence = run.public_info()["sequence"]
        run.publish({"type": "progress", "value": 4, "max": 6})
        run.finish({"type": "done", "generation_id": "generation-1"})

        response = self.client.get(
            f"/api/generate/runs/{run.id}/stream?after={sequence}", headers=headers,
        )

        self.assertEqual(response.status_code, 200)
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertEqual([event["type"] for event in events], ["progress", "done"])

    def test_generate_stream_forwards_progress_and_completion(self) -> None:
        headers = {"Authorization": "Bearer test-access-token"}
        conversation = self.client.post("/api/conversations", headers=headers).json()
        result = GenerationOutcome("generation-1", "prompt-1", {
            "filename": "result.png", "subfolder": "", "type": "output",
        })

        def fake_generate(_orchestrator, _conversation_id, _prompt, _seed,
                          _aspect_ratio=None, _megapixels=None, _sampler_name=None,
                          _scheduler=None, progress_callback=None,
                          image_mode="text-to-image", source_image_ids=None):
            progress_callback({"type": "phase", "message": "Preparing ComfyUI"})
            progress_callback({"type": "progress", "value": 3, "max": 6})
            return result

        with patch("comfy_studio.app.Orchestrator.generate", autospec=True,
                   side_effect=fake_generate) as generate:
            response = self.client.post("/api/generate/stream", headers=headers, json={
                "conversation_id": conversation["id"],
                "prompt": "A red teapot",
                "seed": 42,
            })

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        run_id = response.headers["x-generation-run-id"]
        self.assertTrue(run_id)
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertEqual([event["type"] for event in events], ["phase", "progress", "done"])
        self.assertEqual(events[1]["value"], 3)
        self.assertEqual(events[-1]["generation_id"], "generation-1")
        self.assertIsNone(self.app.state.generation_runs.active())
        self.assertEqual(generate.call_args.kwargs["progress_callback"].__class__.__name__, "method")
        self.assertEqual(generate.call_args.kwargs["image_mode"], "text-to-image")
        self.assertEqual(generate.call_args.kwargs["source_image_ids"], [])
        self.assertEqual(generate.call_args.args[6:8], (None, None))

    def test_generated_image_is_served_from_studio_session_storage(self) -> None:
        headers = {"Authorization": "Bearer test-access-token"}
        conversation = self.client.post("/api/conversations", headers=headers).json()
        generation_id = self.store.create_generation(
            conversation["id"], "A red teapot", 42, "1:1 (Square)", 1.0,
            1024, 1024, "euler", "simple",
        )
        filename = f"generation_{generation_id}.png"
        self.store.update_generation(generation_id, "complete", image={
            "filename": filename, "subfolder": conversation["id"], "type": "output",
        })
        image_path = (
            self.settings.path("paths.session_root") / conversation["id"] / filename
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"generated PNG")

        response = self.client.get(f"/api/generations/{generation_id}/image", headers=headers)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(response.content, b"generated PNG")

    def test_source_upload_requires_auth_and_is_scoped_to_its_conversation(self) -> None:
        headers = {"Authorization": "Bearer test-access-token"}
        creation = self.client.post("/api/conversations", headers=headers)
        self.assertEqual(creation.status_code, 200, creation.text)
        conversation = creation.json()
        conversation_id = conversation["id"]
        image_buffer = io.BytesIO()
        Image.new("RGB", (4, 3), color=(40, 100, 160)).save(image_buffer, format="PNG")
        image_bytes = image_buffer.getvalue()
        path = f"/api/conversations/{conversation_id}/source-images"

        self.assertEqual(self.client.post(path, content=image_bytes,
                                          headers={"Content-Type": "image/png"}).status_code, 401)
        uploaded = self.client.post(
            path, content=image_bytes, headers={**headers, "Content-Type": "image/png"},
        )

        self.assertEqual(uploaded.status_code, 200)
        source_image = uploaded.json()
        self.assertEqual((source_image["width"], source_image["height"]), (4, 3))
        self.assertNotIn("filename", source_image)
        self.assertEqual(source_image["in_use"], 0)
        stored = self.client.get(
            f"/api/conversations/{conversation_id}", headers=headers,
        ).json()
        self.assertEqual([item["id"] for item in stored["source_images"]], [source_image["id"]])
        image_response = self.client.get(
            f"{path}/{source_image['id']}", headers=headers,
        )
        self.assertEqual(image_response.status_code, 200)
        self.assertEqual(image_response.content, image_bytes)

        other_response = self.client.post("/api/conversations", headers=headers)
        self.assertEqual(other_response.status_code, 200, other_response.text)
        other = other_response.json()
        self.assertEqual(
            self.client.get(
                f"/api/conversations/{other['id']}/source-images/{source_image['id']}",
                headers=headers,
            ).status_code,
            404,
        )

    def test_source_upload_rejects_bad_content_and_unknown_mime(self) -> None:
        headers = {"Authorization": "Bearer test-access-token"}
        creation = self.client.post("/api/conversations", headers=headers)
        self.assertEqual(creation.status_code, 200, creation.text)
        conversation = creation.json()
        path = f"/api/conversations/{conversation['id']}/source-images"
        unsupported = self.client.post(
            path, content=b"not an image", headers={**headers, "Content-Type": "image/gif"},
        )
        malformed = self.client.post(
            path, content=b"not an image", headers={**headers, "Content-Type": "image/png"},
        )
        self.assertEqual(unsupported.status_code, 415)
        self.assertEqual(malformed.status_code, 422)

    def test_conversation_mode_persists_and_source_order_is_owned(self) -> None:
        headers = {"Authorization": "Bearer test-access-token"}
        creation = self.client.post("/api/conversations", headers=headers)
        self.assertEqual(creation.status_code, 200, creation.text)
        conversation = creation.json()
        conversation_id = conversation["id"]
        mode = self.client.put(
            f"/api/conversations/{conversation_id}/mode",
            headers=headers, json={"image_mode": "edit"},
        )
        self.assertEqual(mode.json()["image_mode"], "edit")
        self.assertEqual(
            self.client.get(f"/api/conversations/{conversation_id}", headers=headers)
            .json()["conversation"]["image_mode"],
            "edit",
        )
        order = self.client.put(
            f"/api/conversations/{conversation_id}/source-images/order",
            headers=headers, json={"source_image_ids": []},
        )
        self.assertEqual(order.status_code, 200)

    def test_unknown_image_mode_is_rejected_by_api(self) -> None:
        headers = {"Authorization": f"Bearer {self.data['app']['access_token']}"}
        creation = self.client.post("/api/conversations", headers=headers)
        self.assertEqual(creation.status_code, 200, creation.text)
        conversation = creation.json()
        response = self.client.put(
            f"/api/conversations/{conversation['id']}/mode",
            headers=headers,
            json={"image_mode": "not-registered"},
        )
        self.assertEqual(response.status_code, 422)

    def test_bonsai_state_reports_selected_backend_as_idle_when_unloaded(self) -> None:
        settings = Settings(self.root, self.data, chat_backend="bonsai")
        app = create_app(settings, self.store, AppStrata(), AppComfy(settings))
        client = TestClient(app)
        try:
            with patch("comfy_studio.app.service_running", return_value=False):
                response = client.get(
                    "/api/state",
                    headers={"Authorization": "Bearer test-access-token"},
                )
        finally:
            client.close()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["chat_backend"], "bonsai")
        self.assertEqual(
            response.json()["services"]["bonsai"],
            {"available": True, "loaded": False},
        )

    def test_unknown_asset_is_not_served(self) -> None:
        response = self.client.get("/assets/../../config.json")
        self.assertEqual(response.status_code, 404)

    def test_conversation_delete_requires_auth_and_returns_empty_state(self) -> None:
        conversation = self.client.post("/api/conversations",
                                        headers={"Authorization": "Bearer test-access-token"})
        self.assertEqual(conversation.status_code, 200)
        conversation_id = conversation.json()["id"]
        self.assertEqual(self.client.delete(f"/api/conversations/{conversation_id}").status_code, 401)
        deleted = self.client.delete(
            f"/api/conversations/{conversation_id}",
            headers={"Authorization": "Bearer test-access-token"},
        )
        self.assertEqual(deleted.status_code, 200)
        self.assertTrue(deleted.json()["deleted"])
        self.assertIsNone(self.store.get_conversation(conversation_id))


if __name__ == "__main__":
    unittest.main()