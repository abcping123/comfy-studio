from __future__ import annotations

import json
import secrets
from pathlib import PurePosixPath
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener

from websockets.sync.client import connect as websocket_connect

from .settings import Settings


class BackendError(RuntimeError):
    def __init__(self, service: str, message: str, status: int | None = None,
                 body: Any = None, retry_after: str | None = None):
        super().__init__(f"{service}: {message}")
        self.service = service
        self.status = status
        self.body = body
        self.retry_after = retry_after


class JsonHttpClient:
    def __init__(self, service: str, endpoint: str, timeout_s: float):
        self.service = service
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = timeout_s
        self._opener = build_opener(ProxyHandler({}))

    def request_json(self, method: str, path: str, body: dict[str, Any] | None = None,
                     headers: dict[str, str] | None = None) -> dict[str, Any]:
        request_headers = {"Accept": "application/json", **(headers or {})}
        payload = None
        if body is not None:
            request_headers["Content-Type"] = "application/json"
            payload = json.dumps(body).encode("utf-8")
        request = Request(f"{self.endpoint}{path}", data=payload, headers=request_headers, method=method)
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                raw = response.read()
                status = response.status
                retry_after = response.headers.get("Retry-After")
        except HTTPError as error:
            raw = error.read()
            status = error.code
            retry_after = error.headers.get("Retry-After")
        except (URLError, TimeoutError, OSError) as error:
            raise BackendError(self.service, f"request failed: {error}") from error

        try:
            parsed = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BackendError(self.service, "returned invalid JSON", status=status) from error
        if status < 200 or status >= 300:
            detail = parsed.get("error", parsed) if isinstance(parsed, dict) else parsed
            raise BackendError(self.service, str(detail), status=status, body=parsed,
                               retry_after=retry_after)
        if not isinstance(parsed, dict):
            raise BackendError(self.service, "returned a non-object JSON response", status=status,
                               body=parsed)
        return parsed


class StrataClient(JsonHttpClient):
    def __init__(self, settings: Settings):
        config = settings.section("strata")
        api_key = config["api_key"]
        if not isinstance(api_key, str) or not api_key.strip():
            raise BackendError("Strata", "strata.api_key is empty in config.json")
        self.api_key = api_key
        self.model = config["model"]
        super().__init__("Strata", config["endpoint"], config["request_timeout_s"])

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def status(self) -> dict[str, Any]:
        return self.request_json("GET", "/v1/status", headers=self._headers())

    def load(self) -> dict[str, Any]:
        return self.request_json("POST", "/v1/load", {}, self._headers())

    def unload(self, wait_s: int = 300) -> dict[str, Any]:
        return self.request_json("POST", f"/v1/unload?wait={wait_s}", {}, self._headers())

    def save_session(self, filename: str) -> dict[str, Any]:
        return self.request_json("POST", "/slots/0?action=save", {"filename": filename}, self._headers())

    def restore_session(self, filename: str) -> dict[str, Any]:
        return self.request_json("POST", "/slots/0?action=restore", {"filename": filename}, self._headers())

    def chat(self, messages: list[dict[str, Any]], max_tokens: int = 2048,
             reasoning_effort: str | None = None) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        return self.request_json("POST", "/v1/chat/completions", payload, self._headers())

    def chat_stream(self, messages: list[dict[str, Any]], max_tokens: int = 2048,
                    reasoning_effort: str | None = None) -> Iterator[dict[str, str]]:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        headers = {"Accept": "text/event-stream", "Content-Type": "application/json", **self._headers()}
        request = Request(f"{self.endpoint}/v1/chat/completions", data=json.dumps(payload).encode("utf-8"),
                          headers=headers, method="POST")
        try:
            response = self._opener.open(request, timeout=self.timeout_s)
        except HTTPError as error:
            raw = error.read()
            try:
                parsed = json.loads(raw.decode("utf-8")) if raw else {}
            except (UnicodeDecodeError, json.JSONDecodeError):
                parsed = raw.decode("utf-8", errors="replace")
            detail = parsed.get("error", parsed) if isinstance(parsed, dict) else parsed
            raise BackendError(self.service, str(detail), status=error.code, body=parsed,
                               retry_after=error.headers.get("Retry-After")) from error
        except (URLError, TimeoutError, OSError) as error:
            raise BackendError(self.service, f"stream request failed: {error}") from error

        try:
            with response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").rstrip("\r\n")
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError as error:
                        raise BackendError(self.service, "returned invalid streaming JSON") from error
                    if not isinstance(chunk, dict):
                        raise BackendError(self.service, "returned a non-object streaming event")
                    if chunk.get("error"):
                        detail = chunk["error"]
                        raise BackendError(self.service, str(detail.get("message", detail))
                                           if isinstance(detail, dict) else str(detail))
                    choices = chunk.get("choices")
                    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                        continue
                    delta = choices[0].get("delta")
                    if not isinstance(delta, dict):
                        continue
                    content = delta.get("content") or ""
                    reasoning = delta.get("reasoning_content") or ""
                    if not isinstance(content, str) or not isinstance(reasoning, str):
                        raise BackendError(self.service, "returned non-text streaming content")
                    if content or reasoning:
                        yield {"content": content, "reasoning_content": reasoning}
        except (URLError, TimeoutError, OSError) as error:
            raise BackendError(self.service, f"stream response failed: {error}") from error


class ComfyUIClient(JsonHttpClient):
    def __init__(self, settings: Settings):
        config = settings.section("comfyui")
        super().__init__("ComfyUI", config["endpoint"], config["request_timeout_s"])

    def queue(self) -> dict[str, Any]:
        return self.request_json("GET", "/queue")

    def system_stats(self) -> dict[str, Any]:
        return self.request_json("GET", "/system_stats")

    def free_memory(self) -> dict[str, Any]:
        return self.request_json("POST", "/free", {"unload_models": True, "free_memory": True})

    def progress_websocket_url(self, client_id: str) -> str:
        parsed = urlsplit(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise BackendError("ComfyUI", "endpoint must be an HTTP(S) URL for progress streaming")
        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = f"{parsed.path.rstrip('/')}/ws"
        return urlunsplit((scheme, parsed.netloc, path, urlencode({"clientId": client_id}), ""))

    def open_progress_socket(self, client_id: str) -> Any:
        return websocket_connect(
            self.progress_websocket_url(client_id),
            open_timeout=min(self.timeout_s, 10.0),
            close_timeout=2.0,
            max_size=8 * 1024 * 1024,
            proxy=None,
        )

    def submit_prompt(self, workflow: dict[str, Any], client_id: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"prompt": workflow}
        if client_id is not None:
            body["client_id"] = client_id
        return self.request_json("POST", "/prompt", body)

    def upload_image(self, filename: str, content: bytes, content_type: str,
                     subfolder: str) -> dict[str, Any]:
        if (filename in {"", ".", ".."} or PurePosixPath(filename).name != filename or "\\" in filename
                or any(character in filename for character in "\r\n\"")):
            raise BackendError("ComfyUI", "refusing an unsafe upload filename")
        if (not subfolder or PurePosixPath(subfolder).is_absolute()
                or any(part in {"", ".", ".."} for part in PurePosixPath(subfolder).parts)
                or "\\" in subfolder):
            raise BackendError("ComfyUI", "refusing an unsafe upload subfolder")
        if content_type not in {"image/png", "image/jpeg", "image/webp"}:
            raise BackendError("ComfyUI", "unsupported source image content type")

        boundary = f"----Studio{secrets.token_hex(16)}"
        parts = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {content_type}\r\n\r\n".encode("ascii"),
            content,
            b"\r\n",
        ]
        for name, value in (("subfolder", subfolder), ("type", "input"), ("overwrite", "false")):
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                f"{value}\r\n".encode("ascii")
            )
        parts.append(f"--{boundary}--\r\n".encode("ascii"))
        request = Request(
            f"{self.endpoint}/upload/image",
            data=b"".join(parts),
            headers={
                "Accept": "application/json",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                raw = response.read()
                status = response.status
        except HTTPError as error:
            raw = error.read()
            status = error.code
        except (URLError, TimeoutError, OSError) as error:
            raise BackendError("ComfyUI", f"image upload failed: {error}") from error

        try:
            parsed = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BackendError("ComfyUI", "returned invalid upload JSON", status=status) from error
        if status < 200 or status >= 300:
            detail = parsed.get("error", parsed) if isinstance(parsed, dict) else parsed
            raise BackendError("ComfyUI", f"image upload failed: {detail}", status=status, body=parsed)
        if not isinstance(parsed, dict):
            raise BackendError("ComfyUI", "returned a non-object upload response", status=status, body=parsed)
        return parsed

    def history(self, prompt_id: str) -> dict[str, Any]:
        return self.request_json("GET", f"/history/{prompt_id}")

    def image(self, filename: str, subfolder: str = "", image_type: str = "output") -> tuple[bytes, str]:
        if (filename in {"", ".", ".."} or PurePosixPath(filename).name != filename
                or "\\" in filename):
            raise BackendError("ComfyUI", "refusing an unsafe image filename")
        folder = PurePosixPath(subfolder)
        if folder.is_absolute() or ".." in folder.parts or "\\" in subfolder:
            raise BackendError("ComfyUI", "refusing an unsafe image subfolder")
        if image_type not in {"output", "temp", "input"}:
            raise BackendError("ComfyUI", "unsupported image type")
        query = urlencode({"filename": filename, "subfolder": subfolder, "type": image_type})
        request = Request(f"{self.endpoint}/view?{query}", headers={"Accept": "image/*"})
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                return response.read(), response.headers.get("Content-Type", "application/octet-stream")
        except HTTPError as error:
            raise BackendError("ComfyUI", f"image request failed with HTTP {error.code}",
                               status=error.code) from error
        except (URLError, TimeoutError, OSError) as error:
            raise BackendError("ComfyUI", f"image request failed: {error}") from error