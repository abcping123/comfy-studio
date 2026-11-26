export function createApiClient({ getToken, onUnauthorized }) {
  function authHeaders(extra = {}) {
    const token = getToken();
    return token ? { ...extra, Authorization: `Bearer ${token}` } : extra;
  }

  function handleUnauthorized() {
    onUnauthorized();
    return new Error("Access token is missing or invalid.");
  }

  async function api(path, options = {}) {
    const headers = authHeaders(options.headers || {});
    if (options.body && !headers["Content-Type"]) headers["Content-Type"] = "application/json";
    const response = await fetch(path, { ...options, headers, cache: "no-store" });
    if (response.status === 401) throw handleUnauthorized();
    const contentType = response.headers.get("content-type") || "";
    const body = contentType.includes("application/json") ? await response.json() : null;
    if (!response.ok) throw new Error(body?.detail || `Request failed (${response.status})`);
    return body;
  }

  async function consumeEventStream(response, onEvent, {
    missingBodyMessage,
    incompleteMessage,
    defaultErrorMessage,
    markGenerationFailed = false,
  }) {
    if (response.status === 401) throw handleUnauthorized();
    if (!response.ok) {
      const error = response.headers.get("content-type")?.includes("application/json")
        ? await response.json() : null;
      const failure = new Error(error?.detail || `Request failed (${response.status})`);
      failure.status = response.status;
      throw failure;
    }
    if (!response.body) throw new Error(missingBodyMessage);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let outcome = null;
    let streamError = null;
    const dispatch = (frame) => {
      const data = frame.split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).replace(/^ /, ""))
        .join("\n");
      if (!data || data === "[DONE]") return;
      const event = JSON.parse(data);
      if (event.type === "done") {
        outcome = event;
      } else if (event.type === "error") {
        streamError = new Error(event.message || defaultErrorMessage);
        if (markGenerationFailed) streamError.generationFailed = true;
      }
      onEvent(event);
    };
    try {
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done }).replace(/\r\n/g, "\n");
        let boundary = buffer.indexOf("\n\n");
        while (boundary >= 0) {
          dispatch(buffer.slice(0, boundary));
          buffer = buffer.slice(boundary + 2);
          boundary = buffer.indexOf("\n\n");
        }
        if (done) break;
      }
      if (buffer.trim()) dispatch(buffer);
    } catch (error) {
      await reader.cancel().catch(() => {});
      throw error;
    } finally {
      reader.releaseLock();
    }
    if (streamError) throw streamError;
    if (!outcome) throw new Error(incompleteMessage);
    return outcome;
  }

  async function streamChat(body, onEvent) {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json", Accept: "text/event-stream" }),
      body: JSON.stringify(body),
      cache: "no-store",
    });
    return consumeEventStream(response, onEvent, {
      missingBodyMessage: "Streaming response body is unavailable.",
      incompleteMessage: "Image assistant stream ended before the answer was complete.",
      defaultErrorMessage: "Image assistant stream failed.",
    });
  }

  async function resumeChatRun(runId, onEvent) {
    const response = await fetch(`/api/chat/runs/${encodeURIComponent(runId)}/stream`, {
      headers: authHeaders({ Accept: "text/event-stream" }),
      cache: "no-store",
    });
    return consumeEventStream(response, onEvent, {
      missingBodyMessage: "Streaming response body is unavailable.",
      incompleteMessage: "Image assistant stream ended before the answer was complete.",
      defaultErrorMessage: "Image assistant stream failed.",
    });
  }

  async function streamGeneration(body, onEvent, onRunId = () => {}) {
    const requestBody = JSON.stringify(body).replace(
      `"seed":${JSON.stringify(body.seed)}`,
      `"seed":${body.seed}`,
    );
    const response = await fetch("/api/generate/stream", {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json", Accept: "text/event-stream" }),
      body: requestBody,
      cache: "no-store",
    });
    if (response.ok) onRunId(response.headers.get("X-Generation-Run-Id"));
    return consumeEventStream(response, onEvent, {
      missingBodyMessage: "Generation stream is unavailable.",
      incompleteMessage: "Generation stream ended before the image was ready.",
      defaultErrorMessage: "ComfyUI generation failed.",
      markGenerationFailed: true,
    });
  }

  async function resumeGenerationStream(runId, afterSequence, onEvent) {
    const response = await fetch(
      `/api/generate/runs/${encodeURIComponent(runId)}/stream?after=${afterSequence}`,
      { headers: authHeaders({ Accept: "text/event-stream" }), cache: "no-store" },
    );
    return consumeEventStream(response, onEvent, {
      missingBodyMessage: "Generation stream is unavailable.",
      incompleteMessage: "Generation stream ended before the image was ready.",
      defaultErrorMessage: "ComfyUI generation failed.",
      markGenerationFailed: true,
    });
  }

  async function fetchBlobUrl(path, errorPrefix) {
    const response = await fetch(path, { headers: authHeaders(), cache: "no-store" });
    if (response.status === 401) {
      onUnauthorized();
      return null;
    }
    if (!response.ok) throw new Error(`${errorPrefix} (${response.status})`);
    return URL.createObjectURL(await response.blob());
  }

  return {
    api,
    authHeaders,
    fetchBlobUrl,
    resumeChatRun,
    resumeGenerationStream,
    streamChat,
    streamGeneration,
  };
}
