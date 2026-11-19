(() => {
  const tokenKey = "fieldnotes.access-token";
  const state = {
    token: sessionStorage.getItem(tokenKey) || "",
    conversationId: null,
    options: null,
    pendingDeleteId: null,
    busy: false,
    imageUrls: new Set(),
    imageViewerGenerationId: null,
  };

  const byId = (id) => document.getElementById(id);
  const messages = byId("messages");
  const chatInput = byId("chat-input");
  const imagePrompt = byId("image-prompt");
  const tokenDialog = byId("token-dialog");
  const deleteDialog = byId("delete-dialog");
  const imageLightbox = byId("image-lightbox");
  const emptyResult = byId("empty-result");

  function authHeaders(extra = {}) {
    return state.token ? { ...extra, Authorization: `Bearer ${state.token}` } : extra;
  }

  async function api(path, options = {}) {
    const headers = authHeaders(options.headers || {});
    if (options.body && !headers["Content-Type"]) headers["Content-Type"] = "application/json";
    const response = await fetch(path, { ...options, headers, cache: "no-store" });
    if (response.status === 401) {
      sessionStorage.removeItem(tokenKey);
      state.token = "";
      if (!tokenDialog.open) tokenDialog.showModal();
      throw new Error("Access token is missing or invalid.");
    }
    const contentType = response.headers.get("content-type") || "";
    const body = contentType.includes("application/json") ? await response.json() : null;
    if (!response.ok) throw new Error(body?.detail || `Request failed (${response.status})`);
    return body;
  }

  async function consumeChatStream(response, onEvent) {
    if (response.status === 401) {
      sessionStorage.removeItem(tokenKey);
      state.token = "";
      if (!tokenDialog.open) tokenDialog.showModal();
      throw new Error("Access token is missing or invalid.");
    }
    if (!response.ok) {
      const error = response.headers.get("content-type")?.includes("application/json")
        ? await response.json() : null;
      const failure = new Error(error?.detail || `Request failed (${response.status})`);
      failure.status = response.status;
      throw failure;
    }
    if (!response.body) throw new Error("Streaming response body is unavailable.");

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
      if (event.type === "done") outcome = event;
      else if (event.type === "error") streamError = new Error(event.message || "Strata stream failed.");
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
    if (!outcome) throw new Error("Strata stream ended before the answer was complete.");
    return outcome;
  }

  async function streamChat(body, onEvent) {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json", Accept: "text/event-stream" }),
      body: JSON.stringify(body),
      cache: "no-store",
    });
    return consumeChatStream(response, onEvent);
  }

  async function resumeChatRun(runId, onEvent) {
    const response = await fetch(`/api/chat/runs/${encodeURIComponent(runId)}/stream`, {
      headers: authHeaders({ Accept: "text/event-stream" }),
      cache: "no-store",
    });
    return consumeChatStream(response, onEvent);
  }

  async function streamGeneration(body, onEvent) {
    const response = await fetch("/api/generate/stream", {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json", Accept: "text/event-stream" }),
      body: JSON.stringify(body),
      cache: "no-store",
    });
    if (response.status === 401) {
      sessionStorage.removeItem(tokenKey);
      state.token = "";
      if (!tokenDialog.open) tokenDialog.showModal();
      throw new Error("Access token is missing or invalid.");
    }
    if (!response.ok) {
      const error = response.headers.get("content-type")?.includes("application/json")
        ? await response.json() : null;
      throw new Error(error?.detail || `Request failed (${response.status})`);
    }
    if (!response.body) throw new Error("Generation stream is unavailable.");

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
      if (!data) return;
      const event = JSON.parse(data);
      if (event.type === "done") outcome = event;
      else if (event.type === "error") streamError = new Error(event.message || "ComfyUI generation failed.");
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
    if (!outcome) throw new Error("Generation stream ended before the image was ready.");
    return outcome;
  }

  function resetGenerationMonitor() {
    const button = byId("generate-image");
    button.classList.remove("is-error");
    button.classList.add("is-progressing");
    button.setAttribute("aria-busy", "true");
    button.title = "";
    byId("generate-image-label").textContent = "Preparing GPU";
    byId("generate-image-percent").textContent = "";
    const progress = byId("generate-image-progress");
    progress.classList.add("is-indeterminate");
    progress.removeAttribute("aria-valuenow");
    progress.style.setProperty("--progress", "0%");
  }

  function clearGenerationMonitor() {
    const button = byId("generate-image");
    button.classList.remove("is-error", "is-progressing");
    button.removeAttribute("aria-busy");
    button.title = "";
    byId("generate-image-label").textContent = "Generate image";
    byId("generate-image-percent").textContent = "";
    const progress = byId("generate-image-progress");
    progress.classList.add("is-indeterminate");
    progress.removeAttribute("aria-valuenow");
    progress.style.removeProperty("--progress");
  }

  function setGenerationProgress(value) {
    const progress = byId("generate-image-progress");
    const percentLabel = byId("generate-image-percent");
    if (!Number.isFinite(value)) {
      progress.classList.add("is-indeterminate");
      progress.removeAttribute("aria-valuenow");
      percentLabel.textContent = "";
      return;
    }
    const percentage = Math.min(100, Math.max(0, Math.floor(value)));
    progress.classList.remove("is-indeterminate");
    progress.setAttribute("aria-valuenow", String(percentage));
    progress.style.setProperty("--progress", `${percentage}%`);
    percentLabel.textContent = `${percentage}%`;
  }

  function updateGenerationMonitor(event) {
    const button = byId("generate-image");
    const label = byId("generate-image-label");
    if (event.type === "phase") {
      const phaseLabels = {
        "Preparing GPU resources for ComfyUI": "Preparing GPU",
        "Connected to ComfyUI live progress": "Connected to ComfyUI",
        "Live progress unavailable; tracking completion status": "Waiting on ComfyUI",
        "Submitting workflow to ComfyUI": "Submitting workflow",
        "ComfyUI is executing the workflow": "Generating image",
        "Releasing ComfyUI memory": "Finishing image",
        "Image ready": "Image ready",
      };
      label.textContent = phaseLabels[event.message] || event.message || "Generating image";
      if (event.message?.startsWith("Live progress unavailable")) setGenerationProgress(Number.NaN);
    } else if (event.type === "node" || event.type === "progress") {
      const nodeLabels = {
        UNETLoader: "Loading diffusion model",
        CLIPLoader: "Loading text encoder",
        VAELoader: "Loading VAE",
        MiniMaxH3ImageConditioning: "Preparing image conditioning",
        ConditioningZeroOut: "Preparing negative conditioning",
        KSampler: "Sampling image",
        VAEDecode: "Decoding image",
        SaveImage: "Saving image",
      };
      const total = Number(event.total_nodes);
      const totalUnits = Number(event.total_units);
      const completedUnits = Number(event.completed_units);
      const nodeLabel = nodeLabels[event.label] || event.label || "Processing image";
      if (event.label === "KSampler") {
        label.textContent = event.type === "progress"
          ? `Sampling image (${event.value}/${event.max})`
          : event.state === "complete" ? "Sampling complete" : "Sampling image";
      } else {
        const step = event.state === "complete" ? completedUnits : completedUnits + 1;
        const stepLabel = Number.isFinite(totalUnits) && totalUnits > 0 && step > 0
          ? `Step ${step}/${totalUnits} - ${nodeLabel}`
          : nodeLabel;
        label.textContent = event.type === "progress"
          ? `${stepLabel} (${event.value}/${event.max})`
          : stepLabel;
      }
      const overall = Number(event.overall_percent);
      if (Number.isFinite(overall)) setGenerationProgress(overall);
    } else if (event.type === "error") {
      button.classList.add("is-error");
      button.title = event.message || "Image generation failed.";
      label.textContent = "Generation failed";
      setGenerationProgress(Number.NaN);
    } else if (event.type === "done") {
      button.classList.toggle("is-error", Boolean(event.cleanup_warning));
      button.title = event.cleanup_warning || "";
      label.textContent = event.cleanup_warning ? "Image ready - cleanup warning" : "Image ready";
      setGenerationProgress(100);
    }
  }

  function toast(message, isError = false) {
    const item = document.createElement("div");
    item.className = `toast${isError ? " is-error" : ""}`;
    item.textContent = message;
    byId("toast-region").append(item);
    window.setTimeout(() => item.remove(), 5500);
  }

  function setBusy(busy) {
    state.busy = busy;
    byId("send-message").disabled = busy;
    chatInput.disabled = busy;
    byId("thinking-effort").disabled = busy;
    byId("generate-image").disabled = busy || !imagePrompt.value.trim() || !calculateResolution();
    byId("new-conversation").disabled = busy;
    document.querySelectorAll(".conversation-item, .conversation-delete").forEach((button) => { button.disabled = busy; });
  }

  function appendMessage(role, text, label = "", reasoningContent = "", streaming = false) {
    const empty = messages.querySelector(".empty-chat");
    if (empty) empty.remove();
    const row = document.createElement("article");
    row.className = `message-row ${role}`;
    const avatar = document.createElement("span");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "Y" : "S";
    const stack = document.createElement("div");
    let details = null;
    let summary = null;
    let reasoning = null;
    if (role === "assistant" && (streaming || reasoningContent.trim())) {
      details = document.createElement("details");
      details.className = "reasoning-details";
      details.open = streaming;
      summary = document.createElement("summary");
      summary.textContent = streaming ? "Strata 正在思考…" : "思考过程";
      reasoning = document.createElement("div");
      reasoning.className = "reasoning-content";
      reasoning.textContent = reasoningContent;
      details.append(summary, reasoning);
      stack.append(details);
    }
    const bubble = document.createElement("div");
    bubble.className = "message-body";
    bubble.textContent = text;
    bubble.hidden = streaming && !text;
    stack.append(bubble);
    let meta = null;
    if (label) {
      meta = document.createElement("div");
      meta.className = "message-meta";
      meta.textContent = label;
      stack.append(meta);
    }
    row.append(avatar, stack);
    messages.append(row);
    messages.scrollTop = messages.scrollHeight;
    return { stack, bubble, details, summary, reasoning, meta };
  }

  function showEmptyChat() {
    messages.replaceChildren();
    const wrap = document.createElement("div");
    wrap.className = "empty-chat";
    const art = document.createElement("div");
    art.className = "empty-art";
    art.setAttribute("aria-hidden", "true");
    art.textContent = "S";
    const heading = document.createElement("h2");
    heading.textContent = "What are we making?";
    const copy = document.createElement("p");
    copy.textContent = "Start with a scene, object, mood, or any detail you want to explore.";
    const suggestions = document.createElement("div");
    suggestions.className = "suggestion-row";
    ["A quiet coastal bookstore", "A sculptural glass lamp", "A field guide illustration"].forEach((text) => {
      const button = document.createElement("button");
      button.className = "suggestion";
      button.type = "button";
      button.textContent = text;
      button.addEventListener("click", () => {
        chatInput.value = text;
        resizeChatInput();
        chatInput.focus();
      });
      suggestions.append(button);
    });
    wrap.append(art, heading, copy, suggestions);
    messages.append(wrap);
  }

  function updatePromptCount() {
    byId("prompt-count").textContent = `${imagePrompt.value.length.toLocaleString()} characters`;
    updateResolutionPreview();
  }

  function resizeChatInput() {
    chatInput.style.height = "auto";
    const maxHeight = Number.parseFloat(getComputedStyle(chatInput).maxHeight);
    const contentHeight = chatInput.scrollHeight;
    chatInput.style.height = `${Math.min(contentHeight, maxHeight)}px`;
    chatInput.style.overflowY = contentHeight > maxHeight ? "auto" : "hidden";
  }

  function roundToEven(value) {
    const lower = Math.floor(value);
    const fraction = value - lower;
    const tolerance = Number.EPSILON * Math.max(1, Math.abs(value)) * 2;
    if (Math.abs(fraction - 0.5) > tolerance) return Math.round(value);
    return lower % 2 === 0 ? lower : lower + 1;
  }

  function calculateResolution() {
    if (!state.options) return null;
    const ratio = state.options.aspect_ratios.find((item) => item.label === byId("image-aspect-ratio").value);
    const megapixels = Number(byId("image-megapixels").value);
    const multiple = state.options.resolution_multiple;
    if (!ratio || !Number.isFinite(megapixels) || megapixels < 0.1 || megapixels > 16) return null;
    const scale = Math.sqrt(megapixels * 1024 * 1024 / (ratio.width * ratio.height));
    return {
      width: roundToEven(ratio.width * scale / multiple) * multiple,
      height: roundToEven(ratio.height * scale / multiple) * multiple,
      megapixels,
      multiple,
    };
  }

  function updateResolutionPreview() {
    const result = calculateResolution();
    byId("generate-image").disabled = state.busy || !imagePrompt.value.trim() || !result;
    if (!result) {
      byId("resolution-value").textContent = "Check resolution settings";
      byId("resolution-details").textContent = "0.1–16 MP";
      return;
    }
    byId("resolution-value").textContent = `${result.width} x ${result.height} px`;
    byId("resolution-details").textContent = `${result.megapixels.toFixed(2)} MP target / ${result.multiple} px aligned`;
  }

  function fillSelect(select, options, valueOf, labelOf, selected) {
    select.replaceChildren();
    options.forEach((item) => {
      const option = document.createElement("option");
      option.value = valueOf(item);
      option.textContent = labelOf(item);
      select.append(option);
    });
    select.value = selected;
    select.disabled = false;
  }

  async function loadOptions() {
    const options = await api("/api/options");
    state.options = options;
    fillSelect(byId("image-aspect-ratio"), options.aspect_ratios,
      (item) => item.label, (item) => item.label, options.default_aspect_ratio);
    fillSelect(byId("image-sampler"), options.samplers, (item) => item, (item) => item,
      options.default_sampler);
    fillSelect(byId("image-scheduler"), options.schedulers, (item) => item, (item) => item,
      options.default_scheduler);
    const megapixels = byId("image-megapixels");
    megapixels.min = String(options.megapixels.min);
    megapixels.max = String(options.megapixels.max);
    megapixels.step = String(options.megapixels.step);
    megapixels.value = Number(options.megapixels.default).toFixed(2);
    updateResolutionPreview();
  }

  function setServiceStatus(name, service, text) {
    byId(`${name}-status`).textContent = text;
    const dot = byId(`${name}-dot`);
    dot.className = "status-dot";
    if (!service?.available) dot.classList.add("is-error");
    else if (service?.loaded || service?.queue_running || service?.queue_pending) dot.classList.add("is-busy");
    else dot.classList.add("is-live");
  }

  async function refreshState() {
    const data = await api("/api/state");
    const strata = data.services.strata;
    const comfy = data.services.comfyui;
    setServiceStatus("strata", strata, strata?.available ? (strata.loaded ? "Strata loaded" : "Strata idle") : "Strata offline");
    const comfyText = !comfy?.available ? "MiniMax offline" :
      (comfy.queue_running || comfy.queue_pending ? "MiniMax busy" : "MiniMax ready");
    setServiceStatus("comfy", comfy, comfyText);
    renderConversationList(data.conversations || []);
    if (!state.conversationId) {
      const activeChat = data.active_chat;
      const selected = activeChat?.conversation_id || data.active_conversation_id || data.conversations?.[0]?.id;
      if (activeChat) setBusy(true);
      if (selected) {
        const conversation = await selectConversation(selected, null, Boolean(activeChat));
        if (activeChat && conversation) void resumeActiveChat(activeChat, conversation);
        else if (activeChat) setBusy(false);
      }
      else await createConversation();
    }
  }

  function renderSystemMetrics(metrics) {
    const fields = [
      ["cpu", "cpu_percent"],
      ["memory", "memory_percent"],
      ["gpu", "gpu_percent"],
      ["vram", "vram_percent"],
    ];
    let available = false;
    fields.forEach(([name, key]) => {
      const value = metrics[key];
      const label = byId(`system-${name}-value`);
      const meter = byId(`system-${name}-meter`);
      const fill = meter.firstElementChild;
      if (typeof value !== "number" || !Number.isFinite(value)) {
        label.textContent = "--";
        meter.removeAttribute("aria-valuenow");
        meter.setAttribute("aria-valuetext", "Unavailable");
        fill.style.width = "0%";
        return;
      }
      available = true;
      const percent = Math.max(0, Math.min(100, value));
      const display = Math.round(percent);
      label.textContent = `${display}%`;
      meter.setAttribute("aria-valuenow", percent.toFixed(1));
      meter.setAttribute("aria-valuetext", `${display}%`);
      fill.style.width = `${percent}%`;
    });
    byId("system-status-label").textContent = available ? "Live" : "Unavailable";
  }

  async function refreshSystemMetrics() {
    renderSystemMetrics(await api("/api/system-metrics"));
  }

  let systemMetricsTimer = null;
  function startSystemMetricsPolling() {
    if (systemMetricsTimer !== null) return;
    const update = () => {
      if (!state.token) return;
      refreshSystemMetrics().catch(() => renderSystemMetrics({}));
    };
    update();
    systemMetricsTimer = window.setInterval(update, 3000);
  }

  function renderConversationList(conversations) {
    const list = byId("conversation-list");
    list.replaceChildren();
    conversations.forEach((conversation) => {
      const entry = document.createElement("div");
      entry.className = "conversation-entry";
      const select = document.createElement("button");
      select.type = "button";
      select.className = `conversation-item${conversation.id === state.conversationId ? " is-active" : ""}`;
      const title = conversation.title || "New conversation";
      select.title = title;
      select.setAttribute("aria-label", title);
      const label = document.createElement("span");
      label.className = "conversation-item-label";
      label.textContent = title;
      const initial = document.createElement("span");
      initial.className = "conversation-item-initial";
      initial.setAttribute("aria-hidden", "true");
      initial.textContent = Array.from(title.trim())[0] || "N";
      select.append(label, initial);
      select.addEventListener("click", () => selectConversation(conversation.id));
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "conversation-delete";
      remove.title = "Delete conversation";
      remove.setAttribute("aria-label", `Delete conversation ${conversation.title || "New conversation"}`);
      const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      icon.setAttribute("viewBox", "0 0 16 16");
      icon.setAttribute("aria-hidden", "true");
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", "M3 4.5h10M6 4.5V3h4v1.5m2 0-.6 8.5H4.6L4 4.5m2.5 2v4m3-4v4");
      icon.append(path);
      remove.append(icon);
      remove.addEventListener("click", () => requestDelete(conversation.id, conversation.title || "New conversation"));
      entry.append(select, remove);
      list.append(entry);
    });
  }

  function requestDelete(conversationId, title) {
    if (state.busy) return;
    state.pendingDeleteId = conversationId;
    byId("delete-title").textContent = `Delete “${title}”?`;
    deleteDialog.showModal();
  }

  async function deleteConversation() {
    const conversationId = state.pendingDeleteId;
    if (!conversationId || state.busy) return;
    setBusy(true);
    byId("confirm-delete").disabled = true;
    try {
      await api(`/api/conversations/${encodeURIComponent(conversationId)}`, { method: "DELETE" });
      deleteDialog.close();
      state.pendingDeleteId = null;
      if (state.conversationId === conversationId) {
        state.conversationId = null;
        clearImages();
        clearGenerationMonitor();
        imagePrompt.value = "";
        updatePromptCount();
        showEmptyChat();
        byId("conversation-title").textContent = "New conversation";
      }
      const data = await api("/api/state");
      renderConversationList(data.conversations || []);
      if (!state.conversationId) {
        const nextId = data.active_conversation_id || data.conversations?.[0]?.id;
        if (nextId) await selectConversation(nextId);
        else await createConversation();
      }
      toast("Conversation deleted. Saved PNG files remain in ComfyUI output.");
    } catch (error) {
      toast(error.message, true);
    } finally {
      byId("confirm-delete").disabled = false;
      setBusy(false);
    }
  }

  function clearImages() {
    closeImageViewer();
    state.imageUrls.forEach((url) => URL.revokeObjectURL(url));
    state.imageUrls.clear();
    byId("image-gallery").replaceChildren(emptyResult);
    showGeneration();
  }

  async function addGenerationImage(generation) {
    if (generation.status !== "complete" || !generation.filename) return;
    const response = await fetch(`/api/generations/${generation.id}/image`, {
      headers: authHeaders(), cache: "no-store",
    });
    if (response.status === 401) {
      sessionStorage.removeItem(tokenKey);
      state.token = "";
      if (!tokenDialog.open) tokenDialog.showModal();
      return;
    }
    if (!response.ok) throw new Error(`Image fetch failed (${response.status})`);
    const url = URL.createObjectURL(await response.blob());
    state.imageUrls.add(url);
    const card = document.createElement("article");
    card.className = "gallery-item";
    card.dataset.generationId = generation.id;
    const image = document.createElement("img");
    image.src = url;
    image.alt = generation.prompt;
    const openImage = document.createElement("button");
    openImage.type = "button";
    openImage.className = "gallery-image-button";
    openImage.title = "View full-size image";
    openImage.setAttribute("aria-label", "View full-size image");
    openImage.addEventListener("click", () => openImageViewer(generation.id));
    openImage.append(image);
    const caption = document.createElement("div");
    caption.className = "gallery-caption";
    const text = document.createElement("span");
    text.textContent = `${generation.aspect_ratio} · ${Number(generation.megapixels).toFixed(2)} MP · ` +
      `${generation.width} x ${generation.height} · ${generation.sampler_name} / ${generation.scheduler} · seed ${generation.seed}`;
    const link = document.createElement("a");
    link.href = url;
    link.download = generation.filename;
    link.textContent = "Download";
    caption.append(text, link);
    card.append(openImage, caption);
    byId("empty-result")?.remove();
    byId("image-gallery").prepend(card);
  }

  function showGeneration(generationId = null) {
    const cards = [...byId("image-gallery").querySelectorAll(".gallery-item")];
    let activeIndex = generationId
      ? cards.findIndex((card) => card.dataset.generationId === generationId)
      : cards.findIndex((card) => card.classList.contains("is-active"));
    if (activeIndex < 0) activeIndex = 0;
    cards.forEach((card, index) => {
      const active = index === activeIndex;
      card.classList.toggle("is-active", active);
      card.setAttribute("aria-hidden", String(!active));
    });
    byId("result-viewer").classList.toggle("has-results", cards.length > 0);
    byId("older-result").disabled = cards.length < 2 || activeIndex >= cards.length - 1;
    byId("newer-result").disabled = cards.length < 2 || activeIndex <= 0;
    byId("result-count").textContent = cards.length ? `${activeIndex + 1} / ${cards.length}` : "0";
  }

  function moveGeneration(direction) {
    const cards = [...byId("image-gallery").querySelectorAll(".gallery-item")];
    if (cards.length < 2) return;
    const currentIndex = cards.findIndex((card) => card.classList.contains("is-active"));
    const nextIndex = Math.max(0, Math.min(cards.length - 1, (currentIndex < 0 ? 0 : currentIndex) + direction));
    showGeneration(cards[nextIndex].dataset.generationId);
  }

  function updateImageViewer() {
    const cards = [...byId("image-gallery").querySelectorAll(".gallery-item")];
    const index = cards.findIndex((card) => card.dataset.generationId === state.imageViewerGenerationId);
    if (index < 0) {
      closeImageViewer();
      return;
    }
    const source = cards[index].querySelector("img");
    const image = byId("lightbox-image");
    image.src = source.src;
    image.alt = source.alt;
    byId("lightbox-count").textContent = `${index + 1} / ${cards.length}`;
    byId("lightbox-older").disabled = index >= cards.length - 1;
    byId("lightbox-newer").disabled = index <= 0;
  }

  function openImageViewer(generationId) {
    state.imageViewerGenerationId = generationId;
    updateImageViewer();
    if (!imageLightbox.open) imageLightbox.showModal();
  }

  function moveImageViewer(direction) {
    const cards = [...byId("image-gallery").querySelectorAll(".gallery-item")];
    const index = cards.findIndex((card) => card.dataset.generationId === state.imageViewerGenerationId);
    const nextIndex = index + direction;
    if (index < 0 || nextIndex < 0 || nextIndex >= cards.length) return;
    state.imageViewerGenerationId = cards[nextIndex].dataset.generationId;
    updateImageViewer();
  }

  function closeImageViewer() {
    if (imageLightbox.open) imageLightbox.close();
    state.imageViewerGenerationId = null;
    byId("lightbox-image").removeAttribute("src");
  }

  async function selectConversation(conversationId, revealGenerationId = null, allowBusy = false) {
    if (state.busy && !revealGenerationId && !allowBusy) return null;
    try {
      const data = await api(`/api/conversations/${encodeURIComponent(conversationId)}`);
      state.conversationId = conversationId;
      clearImages();
      if (!revealGenerationId) clearGenerationMonitor();
      renderConversationList((await api("/api/state")).conversations || []);
      byId("conversation-title").textContent = data.conversation.title || "New conversation";
      imagePrompt.value = data.conversation.image_prompt || "";
      const conversationRatio = data.conversation.aspect_ratio;
      if (state.options?.aspect_ratios.some((item) => item.label === conversationRatio)) {
        byId("image-aspect-ratio").value = conversationRatio;
      }
      const latestGeneration = data.generations[0];
      if (latestGeneration) {
        if (state.options?.aspect_ratios.some((item) => item.label === latestGeneration.aspect_ratio)) {
          byId("image-aspect-ratio").value = latestGeneration.aspect_ratio;
        }
        byId("image-megapixels").value = Number(latestGeneration.megapixels).toFixed(2);
        byId("image-sampler").value = latestGeneration.sampler_name;
        byId("image-scheduler").value = latestGeneration.scheduler;
      }
      updatePromptCount();
      const badge = byId("context-badge");
      badge.classList.toggle("is-warning", !data.conversation.context_saved_at);
      badge.replaceChildren();
      const dot = document.createElement("i");
      const label = document.createElement("span");
      dot.setAttribute("aria-hidden", "true");
      label.textContent = data.conversation.context_saved_at ? "Native context saved" : "New context";
      badge.append(dot, label);
      if (data.messages.length) {
        messages.replaceChildren();
        data.messages.forEach((message) => appendMessage(
          message.role === "user" ? "user" : "assistant",
          message.content,
          "",
          message.reasoning_content || "",
        ));
      } else {
        showEmptyChat();
      }
      for (const generation of data.generations.slice(0, 8).reverse()) await addGenerationImage(generation);
      showGeneration(revealGenerationId);
      return data;
    } catch (error) {
      toast(error.message, true);
      return null;
    }
  }

  async function createConversation() {
    try {
      const conversation = await api("/api/conversations", { method: "POST", body: "{}" });
      state.conversationId = conversation.id;
      imagePrompt.value = "";
      if (state.options) {
        byId("image-aspect-ratio").value = state.options.default_aspect_ratio;
        byId("image-megapixels").value = Number(state.options.megapixels.default).toFixed(2);
        byId("image-sampler").value = state.options.default_sampler;
        byId("image-scheduler").value = state.options.default_scheduler;
      }
      updatePromptCount();
      clearImages();
      clearGenerationMonitor();
      byId("conversation-title").textContent = "New conversation";
      byId("context-badge").classList.remove("is-warning");
      byId("context-badge").textContent = "New context";
      showEmptyChat();
      await refreshState();
    } catch (error) {
      toast(error.message, true);
    }
  }

  async function sendMessage(event) {
    event.preventDefault();
    const message = chatInput.value.trim();
    if (!message || !state.conversationId || state.busy) return;
    const reasoningEffort = byId("thinking-effort").value;
    appendMessage("user", message);
    chatInput.value = "";
    resizeChatInput();
    setBusy(true);
    const assistant = appendMessage("assistant", "", "", "", true);
    try {
      const result = await streamChat({
          conversation_id: state.conversationId,
          message,
          current_image_prompt: imagePrompt.value,
          current_aspect_ratio: byId("image-aspect-ratio").value,
          reasoning_effort: reasoningEffort,
        }, (event) => {
          updateAssistantTurn(assistant, event);
        });
      completeAssistantTurn(assistant, result);
      await refreshState();
    } catch (error) {
      failAssistantTurn(assistant, error);
    } finally {
      setBusy(false);
    }
  }

  function updateAssistantTurn(assistant, event) {
    if (event.type === "reasoning" && assistant?.reasoning) {
      assistant.reasoning.textContent += event.text;
      assistant.reasoning.scrollTop = assistant.reasoning.scrollHeight;
      messages.scrollTop = messages.scrollHeight;
    }
  }

  function completeAssistantTurn(assistant, result) {
    assistant.bubble.textContent = result.answer;
    assistant.bubble.hidden = false;
    assistant.stack.append(assistant.bubble);
    if (assistant.reasoning) {
      assistant.reasoning.textContent = result.reasoning_content || assistant.reasoning.textContent;
      if (assistant.reasoning.textContent.trim()) {
        assistant.summary.textContent = "思考过程";
        assistant.details.open = false;
      } else {
        assistant.details.remove();
      }
    }
    const contextLabel = result.context_saved ? "Native context saved" : "Context save needs attention";
    if (result.warning) toast(result.warning, true);
    if (result.image_prompt) imagePrompt.value = result.image_prompt;
    if (state.options?.aspect_ratios.some((item) => item.label === result.aspect_ratio)) {
      byId("image-aspect-ratio").value = result.aspect_ratio;
    }
    updatePromptCount();
    assistant.meta = document.createElement("div");
    assistant.meta.className = "message-meta";
    assistant.meta.textContent = contextLabel;
    assistant.stack.append(assistant.meta);
  }

  function failAssistantTurn(assistant, error) {
    if (assistant?.details && assistant.reasoning?.textContent.trim()) {
      assistant.summary.textContent = "思考未完成";
    } else {
      assistant?.details?.remove();
    }
    if (assistant) {
      assistant.bubble.hidden = false;
      assistant.stack.append(assistant.bubble);
      assistant.bubble.textContent = `The turn could not be completed: ${error.message}`;
    }
    toast(error.message, true);
  }

  async function resumeActiveChat(activeChat, conversation) {
    const alreadyCommitted = conversation.messages.length > activeChat.base_message_count;
    let assistant = null;
    if (!alreadyCommitted) {
      appendMessage("user", activeChat.message);
      assistant = appendMessage("assistant", "", "", "", true);
    }
    try {
      const result = await resumeChatRun(activeChat.id, (event) => updateAssistantTurn(assistant, event));
      if (assistant) completeAssistantTurn(assistant, result);
      await refreshState();
    } catch (error) {
      if (error.status === 404) {
        setBusy(false);
        await selectConversation(activeChat.conversation_id);
      } else {
        failAssistantTurn(assistant, error);
      }
    } finally {
      setBusy(false);
    }
  }

  async function generateImage() {
    if (state.busy || !state.conversationId || !imagePrompt.value.trim()) return;
    const resolution = calculateResolution();
    if (!resolution) return;
    const seed = Number(byId("image-seed").value);
    setBusy(true);
    resetGenerationMonitor();
    try {
      const result = await streamGeneration({
          conversation_id: state.conversationId,
          prompt: imagePrompt.value.trim(),
          seed,
          aspect_ratio: byId("image-aspect-ratio").value,
          megapixels: resolution.megapixels,
          sampler_name: byId("image-sampler").value,
          scheduler: byId("image-scheduler").value,
        }, updateGenerationMonitor);
      await selectConversation(state.conversationId, result.generation_id);
      if (result.cleanup_warning) toast(result.cleanup_warning, true);
      else toast("Image generated.");
      await refreshState();
    } catch (error) {
      byId("generate-image").classList.add("is-error");
      byId("generate-image").title = error.message;
      byId("generate-image-label").textContent = "Generation failed";
      setGenerationProgress(Number.NaN);
      toast(error.message, true);
    } finally {
      setBusy(false);
      clearGenerationMonitor();
    }
  }

  byId("new-conversation").addEventListener("click", createConversation);
  byId("refresh-state").addEventListener("click", () => refreshState().catch((error) => toast(error.message, true)));
  byId("chat-form").addEventListener("submit", sendMessage);
  chatInput.addEventListener("input", resizeChatInput);
  chatInput.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey || event.isComposing || event.keyCode === 229) return;
    event.preventDefault();
    byId("chat-form").requestSubmit();
  });
  byId("generate-image").addEventListener("click", generateImage);
  byId("older-result").addEventListener("click", () => moveGeneration(1));
  byId("newer-result").addEventListener("click", () => moveGeneration(-1));
  byId("lightbox-close").addEventListener("click", closeImageViewer);
  byId("lightbox-older").addEventListener("click", () => moveImageViewer(1));
  byId("lightbox-newer").addEventListener("click", () => moveImageViewer(-1));
  imageLightbox.addEventListener("click", (event) => {
    if (event.target === imageLightbox) closeImageViewer();
  });
  imageLightbox.addEventListener("close", () => {
    state.imageViewerGenerationId = null;
    byId("lightbox-image").removeAttribute("src");
  });
  document.addEventListener("keydown", (event) => {
    if (!imageLightbox.open) return;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      moveImageViewer(-1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      moveImageViewer(1);
    } else if (event.key === "Escape") {
      event.preventDefault();
      closeImageViewer();
    }
  });
  byId("cancel-delete").addEventListener("click", () => {
    state.pendingDeleteId = null;
    deleteDialog.close();
  });
  byId("confirm-delete").addEventListener("click", deleteConversation);
  imagePrompt.addEventListener("input", updatePromptCount);
  window.addEventListener("resize", resizeChatInput);
  byId("image-aspect-ratio").addEventListener("change", updateResolutionPreview);
  byId("image-megapixels").addEventListener("input", updateResolutionPreview);
  byId("image-sampler").addEventListener("change", updateResolutionPreview);
  byId("image-scheduler").addEventListener("change", updateResolutionPreview);
  byId("random-seed").addEventListener("click", () => {
    const value = new Uint32Array(1);
    crypto.getRandomValues(value);
    byId("image-seed").value = String(value[0]);
  });
  byId("token-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    state.token = byId("token-input").value.trim();
    sessionStorage.setItem(tokenKey, state.token);
    byId("token-error").textContent = "";
    try {
      await loadOptions();
      await refreshState();
      tokenDialog.close();
      startSystemMetricsPolling();
      byId("token-input").value = "";
    } catch (error) {
      state.token = "";
      sessionStorage.removeItem(tokenKey);
      byId("token-error").textContent = error.message;
    }
  });

  updatePromptCount();
  resizeChatInput();
  if (state.token) {
    loadOptions().then(refreshState).catch((error) => toast(error.message, true));
    startSystemMetricsPolling();
  } else tokenDialog.showModal();
})();