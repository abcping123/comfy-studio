from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
from urllib.request import ProxyHandler

from strata_comfy_studio.clients import BackendError, ComfyUIClient, StrataClient
from strata_comfy_studio.settings import load_config, project_root
from test_settings import sample_config


class FakeHandler(BaseHTTPRequestHandler):
    calls: list[tuple[str, str, dict[str, str], dict | None]] = []

    def log_message(self, _format: str, *args: object) -> None:
        return

    def _reply(self, status: int, body: dict, content_type: str = "application/json") -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        self.calls.append(("GET", self.path, dict(self.headers), None))
        if self.path.startswith("/view?"):
            self._reply(200, {"image": True}, "image/png")
        elif self.path == "/v1/status":
            self._reply(200, {"loaded": False})
        else:
            self._reply(200, {"ok": True, "queue_running": [], "queue_pending": []})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        if self.path == "/upload/image":
            self.calls.append(("POST", self.path, dict(self.headers), {"raw": raw_body}))
            self._reply(200, {
                "name": "uploaded.png", "subfolder": "a" * 32, "type": "input",
            })
            return
        body = json.loads(raw_body or b"{}")
        self.calls.append(("POST", self.path, dict(self.headers), body))
        if self.path == "/v1/status":
            self._reply(200, {"loaded": False})
        elif self.path.startswith("/v1/unload"):
            self._reply(200, {"status": "unloaded", "loaded": False})
        elif self.path == "/slots/0?action=save":
            self._reply(200, {"n_saved": 20, "filename": body["filename"]})
        elif self.path == "/slots/0?action=restore":
            self._reply(200, {"n_restored": 20, "filename": body["filename"]})
        elif self.path == "/v1/chat/completions":
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for event in (
                    {"choices": [{"delta": {"reasoning_content": "Think first. "}}]},
                    {"choices": [{"delta": {"content": json.dumps({"reply": "ok"}, separators=(",", ":"))}}]},
                ):
                    self.wfile.write(b"data: " + json.dumps(event).encode("utf-8") + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                self._reply(200, {"choices": [{"message": {"content": "ok"}}]})
        elif self.path == "/prompt":
            self._reply(200, {"prompt_id": "test-prompt"})
        elif self.path == "/free":
            self._reply(200, {})
        else:
            self._reply(404, {"error": "not found"})


class ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        FakeHandler.calls = []
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.endpoint = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self) -> None:
        data = sample_config()
        data["strata"]["endpoint"] = self.endpoint
        data["comfyui"]["endpoint"] = self.endpoint
        self.config_path = project_root() / "tests" / "client-test-config.json"
        self.config_path.write_text(json.dumps(data), encoding="utf-8")
        self.settings = load_config(self.config_path)
        FakeHandler.calls.clear()

    def tearDown(self) -> None:
        self.config_path.unlink(missing_ok=True)

    def test_strata_requests_use_json_api_key_and_preserve_session_paths(self) -> None:
        client = StrataClient(self.settings)
        status = client.status()
        saved = client.save_session("conversation.bin")
        restored = client.restore_session("conversation.bin")
        unloaded = client.unload()
        self.assertFalse(status["loaded"])
        self.assertEqual(saved["n_saved"], 20)
        self.assertEqual(restored["n_restored"], 20)
        self.assertEqual(unloaded["status"], "unloaded")
        for _, _, headers, _ in FakeHandler.calls:
            self.assertEqual(headers.get("Authorization"), "Bearer test-api-key")

    def test_strata_chat_stream_yields_reasoning_and_content_deltas(self) -> None:
        client = StrataClient(self.settings)
        deltas = list(client.chat_stream([{"role": "user", "content": "Hi"}], max_tokens=32,
                         reasoning_effort="high"))

        self.assertEqual(deltas, [
            {"content": "", "reasoning_content": "Think first. "},
            {"content": json.dumps({"reply": "ok"}, separators=(",", ":")), "reasoning_content": ""},
        ])
        method, path, headers, body = FakeHandler.calls[-1]
        self.assertEqual((method, path), ("POST", "/v1/chat/completions"))
        self.assertEqual(headers.get("Accept"), "text/event-stream")
        self.assertTrue(body["stream"])
        self.assertEqual(body["reasoning_effort"], "high")

    def test_comfy_queue_submit_and_free_contract(self) -> None:
        client = ComfyUIClient(self.settings)
        client.queue()
        result = client.submit_prompt({"1": {"class_type": "SaveImage", "inputs": {}}},
                                      client_id="studio-client")
        client.free_memory()
        self.assertEqual(result["prompt_id"], "test-prompt")
        self.assertEqual(FakeHandler.calls[1][3], {
            "prompt": {"1": {"class_type": "SaveImage", "inputs": {}}},
            "client_id": "studio-client",
        })
        self.assertEqual(FakeHandler.calls[2][3], {"unload_models": True, "free_memory": True})

    def test_comfy_progress_websocket_url_uses_configured_endpoint(self) -> None:
        client = ComfyUIClient(self.settings)
        self.assertEqual(client.progress_websocket_url("studio-client"),
                         f"{self.endpoint.replace('http://', 'ws://')}/ws?clientId=studio-client")

    def test_image_proxy_encodes_query_and_rejects_traversal(self) -> None:
        client = ComfyUIClient(self.settings)
        body, content_type = client.image("sample.png", "job/1")
        self.assertEqual(content_type, "image/png")
        self.assertEqual(body, b'{"image": true}')
        method, path, _, _ = FakeHandler.calls[-1]
        self.assertEqual(method, "GET")
        self.assertEqual(parse_qs(urlsplit(path).query)["subfolder"], ["job/1"])
        with self.assertRaisesRegex(BackendError, "unsafe"):
            client.image("../secret.png")

    def test_comfy_upload_uses_input_subfolder_and_multipart_body(self) -> None:
        client = ComfyUIClient(self.settings)
        conversation_id = "a" * 32
        result = client.upload_image(
            "b" * 32 + ".png", b"sample-image-bytes", "image/png", conversation_id,
        )

        self.assertEqual(result, {
            "name": "uploaded.png", "subfolder": conversation_id, "type": "input",
        })
        _, path, headers, body = FakeHandler.calls[-1]
        self.assertEqual(path, "/upload/image")
        self.assertTrue(headers["Content-Type"].startswith("multipart/form-data; boundary="))
        self.assertIn(b'name="subfolder"', body["raw"])
        self.assertIn(conversation_id.encode(), body["raw"])
        self.assertIn(b'name="type"', body["raw"])
        self.assertIn(b"\r\n\r\nsample-image-bytes\r\n", body["raw"])
        with self.assertRaisesRegex(BackendError, "unsafe"):
            client.upload_image("../bad.png", b"x", "image/png", conversation_id)

    def test_http_errors_keep_status_without_exposing_credentials(self) -> None:
        client = ComfyUIClient(self.settings)
        with self.assertRaises(BackendError) as caught:
            client.request_json("POST", "/missing", {})
        self.assertEqual(caught.exception.status, 404)
        self.assertNotIn("test-api-key", str(caught.exception))

    def test_backend_clients_do_not_use_environment_proxy_settings(self) -> None:
        client = ComfyUIClient(self.settings)
        self.assertFalse(any(isinstance(handler, ProxyHandler) for handler in client._opener.handlers))


if __name__ == "__main__":
    unittest.main()