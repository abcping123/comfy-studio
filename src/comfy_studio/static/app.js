(() => {
  const tokenKey = "fieldnotes.access-token";
  const conversationSettingsPrefix = "fieldnotes.conversation-settings.v1.";
  const state = {
    token: sessionStorage.getItem(tokenKey) || "",
    conversationId: null,
    options: null,
    pendingDeleteId: null,
    busy: false,
    imageMode: null,
    sourceImages: [],
    sourceImageUrls: new Map(),
    sourceRenderToken: 0,
    draggingSourceImageId: null,
    generationRunId: null,
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

  function getMode(modeId = state.imageMode) {
    const mode = state.options?.modes.find((item) => item.id === modeId);
    if (!mode) throw new Error(`Image mode '${modeId}' is not supported.`);
    return mode;
  }

  function getSavedMegapixels(saved, mode) {
    const modeSettings = saved.mode_settings;
    const modeValue = modeSettings && typeof modeSettings === "object" && !Array.isArray(modeSettings)
      ? modeSettings[mode.id]?.megapixels : undefined;
    if (modeValue !== undefined) return Number(modeValue);
    return mode.legacy_local_storage_key
      ? Number(saved[mode.legacy_local_storage_key]) : Number.NaN;
  }

  function sourceImageRole(mode, index) {
    const roles = mode.source_images.roles;
    return roles.length ? roles[Math.min(index, roles.length - 1)] : "source image";
  }

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

  async function consumeGenerationStream(response, onEvent) {
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
      else if (event.type === "error") {
        streamError = new Error(event.message || "ComfyUI generation failed.");
        streamError.generationFailed = true;
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
    if (!outcome) throw new Error("Generation stream ended before the image was ready.");
    return outcome;
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
    return consumeGenerationStream(response, onEvent);
  }

  async function resumeGenerationStream(runId, afterSequence, onEvent) {
    const response = await fetch(
      `/api/generate/runs/${encodeURIComponent(runId)}/stream?after=${afterSequence}`,
      { headers: authHeaders({ Accept: "text/event-stream" }), cache: "no-store" },
    );
    return consumeGenerationStream(response, onEvent);
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
        LoadImage: "Loading source image",
        ResizeImageMaskNode: "Resizing source image",
        GetImageSize: "Reading source image size",
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
    byId("new-conversation").disabled = busy;
    updateResolutionPreview();
    updateModeControls();
    document.querySelectorAll(".conversation-item, .conversation-delete").forEach((button) => { button.disabled = busy; });
    document.querySelectorAll(".source-image-action").forEach((button) => { button.disabled = busy || button.dataset.locked === "true"; });
    document.querySelectorAll(".source-image-card").forEach((card) => { card.draggable = !busy; });
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
    const prompts = [
      "A quiet coastal bookstore",
      "A sculptural glass lamp",
      "A field guide illustration of desert flowers",
      "A tiny greenhouse on a rainy rooftop",
      "A night market just after the rain",
      "A mountain observatory at blue hour",
      "A teacup filled with a miniature garden",
      "An old tram winding through a hillside town",
      "A white fox beneath the northern lights",
      "A botanical print of unusual mushrooms",
      "A floating library among misty cliffs",
      "A warm bakery before sunrise",
      "A coral reef seen through a glass submarine",
      "A ceramic vase shaped by ocean waves",
      "A lantern-lit path through a bamboo forest",
      "A vintage radio in a sunlit attic",
      "A tiny sailboat crossing a pond in a teacup",
      "A modern cabin tucked into autumn woods",
      "A still life of citrus and blue ceramics",
      "A moonlit greenhouse full of tropical leaves",
      "An illustrated map of an imaginary island",
      "A red bicycle beside a flower shop",
      "A snow-covered train station at dusk",
      "A cloud-shaped reading nook for children",
    ];
    for (let index = prompts.length - 1; index > 0; index--) {
      const swapIndex = Math.floor(Math.random() * (index + 1));
      [prompts[index], prompts[swapIndex]] = [prompts[swapIndex], prompts[index]];
    }
    prompts.slice(0, 3).forEach((text) => {
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
    const mode = getMode();
    const megapixels = Number(byId("image-megapixels").value);
    const megapixelOptions = mode.resolution.megapixels;
    if (!Number.isFinite(megapixels) || megapixels < megapixelOptions.min ||
        megapixels > megapixelOptions.max) return null;
    const multiple = state.options.resolution_multiple;
    if (mode.resolution.strategy === "source-image") {
      const base = state.sourceImages[0];
      if (!base || !base.width || !base.height) return null;
      const scale = Math.sqrt(Math.floor(megapixels * 1024 * 1024) / (base.width * base.height));
      const resizedWidth = roundToEven(base.width * scale);
      const resizedHeight = roundToEven(base.height * scale);
      const alignedWidth = Math.floor(resizedWidth / multiple) * multiple;
      const alignedHeight = Math.floor(resizedHeight / multiple) * multiple;
      return {
        width: alignedWidth && alignedHeight ? alignedWidth : resizedWidth,
        height: alignedWidth && alignedHeight ? alignedHeight : resizedHeight,
        megapixels,
        multiple,
      };
    }
    if (mode.resolution.strategy !== "aspect-ratio") return null;
    const ratio = state.options.aspect_ratios.find((item) => item.label === byId("image-aspect-ratio").value);
    if (!ratio) return null;
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
    const mode = state.options ? getMode() : null;
    const sourceCountValid = !mode || mode.source_images.max_count === 0 ||
      (state.sourceImages.length >= mode.source_images.min_count &&
       state.sourceImages.length <= mode.source_images.max_count);
    byId("generate-image").disabled = state.busy || !imagePrompt.value.trim() || !result || !sourceCountValid;
    if (!result) {
      byId("resolution-value").textContent = mode &&
        state.sourceImages.length < mode.source_images.min_count
        ? mode.ui.missing_sources_hint || "Add the required source images"
        : "Check resolution settings";
      return;
    }
    byId("resolution-value").textContent = `${result.width} x ${result.height} px`;
  }

  function renderModeButtons() {
    const container = byId("image-mode-switch");
    container.replaceChildren(...state.options.modes.map((mode) => {
      const button = document.createElement("button");
      button.className = "mode-button";
      button.type = "button";
      button.dataset.modeId = mode.id;
      button.textContent = mode.label;
      button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", () => changeImageMode(mode.id));
      return button;
    }));
  }

  function updateModeControls() {
    if (!state.options) return;
    const mode = getMode();
    const sourcePolicy = mode.source_images;
    document.querySelectorAll(".mode-button").forEach((button) => {
      const active = button.dataset.modeId === mode.id;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
      button.disabled = state.busy;
    });
    const sourcePanel = byId("source-image-panel");
    sourcePanel.hidden = sourcePolicy.max_count === 0;
    sourcePanel.setAttribute("aria-label", mode.ui.source_images_label);
    byId("source-image-help").textContent = mode.ui.source_images_help;
    byId("source-image-input").disabled = state.busy || sourcePolicy.max_count === 0;
    const maxCount = sourcePolicy.max_count;
    const addButton = byId("upload-source-images");
    const atLimit = state.sourceImages.length >= maxCount;
    addButton.disabled = state.busy || maxCount === 0 || atLimit;
    const firstRole = sourceImageRole(mode, 0);
    const addLabel = state.sourceImages.length ? "Add source images" : `Add a ${firstRole}`;
    addButton.querySelector(".source-image-add-label").textContent =
      state.sourceImages.length ? "Add source images" :
        `${firstRole[0].toUpperCase()}${firstRole.slice(1)}`;
    addButton.title = atLimit ? `Maximum of ${maxCount} source images reached` : addLabel;
    addButton.setAttribute("aria-label", addButton.title);
    byId("source-image-count").textContent =
      `${state.sourceImages.length} / ${sourcePolicy.max_count}`;
  }

  function applyImageMode(imageMode) {
    const mode = getMode(imageMode);
    state.imageMode = mode.id;
    const megapixels = byId("image-megapixels");
    const megapixelOptions = mode.resolution.megapixels;
    byId("image-aspect-ratio").disabled = mode.resolution.strategy !== "aspect-ratio";
    megapixels.min = String(megapixelOptions.min);
    megapixels.max = String(megapixelOptions.max);
    megapixels.step = String(megapixelOptions.step);
    byId("image-prompt-label").textContent = mode.ui.prompt_label;
    byId("megapixels-label").textContent = mode.ui.megapixels_label;
    byId("composer-hint").textContent = mode.ui.composer_hint;
    byId("generate-image-label").textContent = mode.ui.generate_label;
    updateModeControls();
    updateResolutionPreview();
  }

  async function changeImageMode(imageMode) {
    if (state.busy || !state.options || !state.conversationId || imageMode === state.imageMode) return;
    const mode = getMode(imageMode);
    const previousMode = state.imageMode;
    const previousMegapixels = byId("image-megapixels").value;
    saveConversationSettings();
    const saved = readConversationSettings();
    applyImageMode(imageMode);
    const megapixelOptions = mode.resolution.megapixels;
    const savedMegapixels = getSavedMegapixels(saved, mode);
    const previousValue = Number(previousMegapixels);
    const selectedMegapixels = Number.isFinite(savedMegapixels) &&
      savedMegapixels >= megapixelOptions.min && savedMegapixels <= megapixelOptions.max
      ? savedMegapixels
      : Number.isFinite(previousValue) && previousValue >= megapixelOptions.min &&
        previousValue <= megapixelOptions.max ? previousValue : megapixelOptions.default;
    byId("image-megapixels").value = selectedMegapixels.toFixed(2);
    updateResolutionPreview();
    try {
      await api(`/api/conversations/${encodeURIComponent(state.conversationId)}/mode`, {
        method: "PUT",
        body: JSON.stringify({ image_mode: imageMode }),
      });
      saveConversationSettings();
    } catch (error) {
      applyImageMode(previousMode);
      byId("image-megapixels").value = previousMegapixels;
      updateResolutionPreview();
      toast(error.message, true);
    }
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
    renderModeButtons();
    fillSelect(byId("image-aspect-ratio"), options.aspect_ratios,
      (item) => item.label, (item) => item.label, options.default_aspect_ratio);
    const megapixels = byId("image-megapixels");
    megapixels.value = Number(getMode(options.default_mode).resolution.megapixels.default).toFixed(2);
    const selectedMode = options.modes.some((mode) => mode.id === state.imageMode)
      ? state.imageMode : options.default_mode;
    applyImageMode(selectedMode);
    updateResolutionPreview();
  }

  function readConversationSettings(conversationId = state.conversationId) {
    if (!conversationId) return {};
    try {
      const saved = JSON.parse(localStorage.getItem(`${conversationSettingsPrefix}${conversationId}`) || "{}");
      return saved && typeof saved === "object" && !Array.isArray(saved) ? saved : {};
    } catch {
      return {};
    }
  }

  function saveConversationSettings() {
    if (!state.conversationId || !state.options) return;
    const saved = readConversationSettings();
    const mode = getMode();
    const aspectRatio = byId("image-aspect-ratio").value;
    const megapixels = Number(byId("image-megapixels").value);
    const thinkingEffort = byId("thinking-effort").value;
    const megapixelOptions = mode.resolution.megapixels;

    if (state.options.aspect_ratios.some((item) => item.label === aspectRatio)) {
      saved.aspect_ratio = aspectRatio;
    }
    if (Number.isFinite(megapixels) && megapixels >= megapixelOptions.min &&
        megapixels <= megapixelOptions.max) {
      const currentSettings = saved.mode_settings;
      const modeSettings = currentSettings && typeof currentSettings === "object" &&
        !Array.isArray(currentSettings) ? currentSettings : {};
      const currentModeSettings = modeSettings[mode.id];
      saved.mode_settings = {
        ...modeSettings,
        [mode.id]: {
          ...(currentModeSettings && typeof currentModeSettings === "object" &&
            !Array.isArray(currentModeSettings) ? currentModeSettings : {}),
          megapixels,
        },
      };
    }
    if (["none", "low", "medium", "high"].includes(thinkingEffort)) {
      saved.thinking_effort = thinkingEffort;
    }

    try {
      localStorage.setItem(`${conversationSettingsPrefix}${state.conversationId}`, JSON.stringify(saved));
    } catch {
      // Browser storage may be unavailable; preferences remain usable for this page load.
    }
  }

  function restoreConversationSettings() {
    const saved = readConversationSettings();
    if (!state.options) return;

    if (state.options.aspect_ratios.some((item) => item.label === saved.aspect_ratio)) {
      byId("image-aspect-ratio").value = saved.aspect_ratio;
    }
    const mode = getMode();
    const megapixelOptions = mode.resolution.megapixels;
    const megapixels = getSavedMegapixels(saved, mode);
    byId("image-megapixels").value = Number.isFinite(megapixels) &&
      megapixels >= megapixelOptions.min && megapixels <= megapixelOptions.max
      ? megapixels.toFixed(2) : Number(megapixelOptions.default).toFixed(2);
    if (["none", "low", "medium", "high"].includes(saved.thinking_effort)) {
      byId("thinking-effort").value = saved.thinking_effort;
    }
  }

  function clearConversationSettings(conversationId) {
    try {
      localStorage.removeItem(`${conversationSettingsPrefix}${conversationId}`);
    } catch {
      // Ignore unavailable browser storage.
    }
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
      const activeGeneration = data.active_generation;
      const activeOperation = activeChat || activeGeneration;
      const selected = activeChat?.conversation_id || activeGeneration?.conversation_id ||
        data.active_conversation_id || data.conversations?.[0]?.id;
      if (activeOperation) setBusy(true);
      if (selected) {
        const conversation = await selectConversation(selected, null, true);
        if (activeChat && conversation) void resumeActiveChat(activeChat, conversation);
        else if (activeGeneration && conversation) void resumeActiveGeneration(activeGeneration);
        else setBusy(false);
      }
      else {
        await createConversation();
        if (!state.generationRunId) setBusy(false);
      }
    } else if (data.active_generation && !state.generationRunId) {
      const activeGeneration = data.active_generation;
      let conversation = true;
      if (state.conversationId !== activeGeneration.conversation_id) {
        setBusy(true);
        conversation = await selectConversation(activeGeneration.conversation_id, null, true);
      }
      if (conversation) void resumeActiveGeneration(activeGeneration);
      else setBusy(false);
    }
  }

  function renderSystemMetrics(metrics) {
    const fields = [
      ["cpu", "cpu_percent"],
      ["memory", "memory_percent"],
      ["gpu", "gpu_percent"],
      ["vram", "vram_percent"],
    ];
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
      const percent = Math.max(0, Math.min(100, value));
      const display = Math.round(percent);
      label.textContent = `${display}%`;
      meter.setAttribute("aria-valuenow", percent.toFixed(1));
      meter.setAttribute("aria-valuetext", `${display}%`);
      fill.style.width = `${percent}%`;
    });
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
      const deletion = await api(`/api/conversations/${encodeURIComponent(conversationId)}`, { method: "DELETE" });
      clearConversationSettings(conversationId);
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
      toast(deletion.cleanup_warning ||
        "Conversation and uploaded source images deleted. Generated PNG files remain in temp/comfyui.");
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
    clearSourceImages();
    byId("image-gallery").replaceChildren(emptyResult);
    showGeneration();
  }

  function clearSourceImages() {
    state.sourceRenderToken += 1;
    state.draggingSourceImageId = null;
    state.sourceImageUrls.forEach((url) => URL.revokeObjectURL(url));
    state.sourceImageUrls.clear();
    state.sourceImages = [];
    const addButton = byId("upload-source-images");
    addButton.className = "source-image-add";
    byId("source-image-list").replaceChildren(addButton);
    if (state.options) {
      byId("source-image-count").textContent = `0 / ${getMode().source_images.max_count}`;
    }
    updateModeControls();
  }

  async function renderSourceImages(sourceImages) {
    const renderToken = ++state.sourceRenderToken;
    const conversationId = state.conversationId;
    const ids = new Set(sourceImages.map((sourceImage) => sourceImage.id));
    state.sourceImageUrls.forEach((url, id) => {
      if (!ids.has(id)) {
        URL.revokeObjectURL(url);
        state.sourceImageUrls.delete(id);
      }
    });
    state.sourceImages = sourceImages;
    const list = byId("source-image-list");
    const addButton = byId("upload-source-images");
    const mode = getMode();
    addButton.className = "source-image-add";
    list.replaceChildren(addButton);
    updateModeControls();

    for (const [index, sourceImage] of sourceImages.entries()) {
      const role = sourceImageRole(mode, index);
      const formattedRole = `${role[0].toUpperCase()}${role.slice(1)}`;
      const card = document.createElement("article");
      card.className = "source-image-card";
      card.dataset.sourceImageId = sourceImage.id;
      card.draggable = !state.busy;
      card.tabIndex = 0;
      card.setAttribute("aria-keyshortcuts", "Alt+Shift+ArrowLeft Alt+Shift+ArrowRight");
      card.setAttribute(
        "aria-label",
        `Picture ${index + 1}, ${role}. Drag to reorder; press Alt+Shift+Left or Alt+Shift+Right to move.`,
      );
      const preview = document.createElement("div");
      preview.className = "source-image-preview";
      const image = document.createElement("img");
      image.draggable = false;
      image.alt = `Picture ${index + 1} ${role}`;
      image.title = `${image.alt} (${sourceImage.width} x ${sourceImage.height})`;
      const url = state.sourceImageUrls.get(sourceImage.id);
      if (url) {
        image.src = url;
      } else {
        const response = await fetch(
          `/api/conversations/${encodeURIComponent(conversationId)}/source-images/${encodeURIComponent(sourceImage.id)}`,
          { headers: authHeaders(), cache: "no-store" },
        );
        if (response.status === 401) {
          sessionStorage.removeItem(tokenKey);
          state.token = "";
          if (!tokenDialog.open) tokenDialog.showModal();
          return;
        }
        if (!response.ok) throw new Error(`Source image fetch failed (${response.status})`);
        const objectUrl = URL.createObjectURL(await response.blob());
        if (renderToken !== state.sourceRenderToken || conversationId !== state.conversationId) {
          URL.revokeObjectURL(objectUrl);
          return;
        }
        state.sourceImageUrls.set(sourceImage.id, objectUrl);
        image.src = objectUrl;
      }
      if (renderToken !== state.sourceRenderToken || conversationId !== state.conversationId) return;

      const details = document.createElement("div");
      details.className = "source-image-details";
      const label = document.createElement("span");
      label.className = "source-image-label";
      label.textContent = `Picture ${index + 1} - ${formattedRole}`;
      details.append(label);

      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "icon-button source-image-action source-image-remove";
      remove.dataset.locked = String(Boolean(sourceImage.in_use));
      remove.textContent = "×";
      remove.title = sourceImage.in_use ? "This image is linked to a generation" : "Remove source image";
      remove.setAttribute("aria-label", `Remove Picture ${index + 1}`);
      remove.disabled = state.busy || Boolean(sourceImage.in_use);
      remove.addEventListener("click", () => removeSourceImage(sourceImage.id));

      preview.append(image, remove);
      card.append(preview, details);

      card.addEventListener("dragstart", (event) => {
        if (state.busy || !event.dataTransfer || (event.target instanceof Element && event.target.closest("button"))) {
          event.preventDefault();
          return;
        }
        state.draggingSourceImageId = sourceImage.id;
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData("text/plain", sourceImage.id);
        card.classList.add("is-dragging");
      });
      card.addEventListener("dragover", (event) => {
        if (state.busy || !state.draggingSourceImageId || state.draggingSourceImageId === sourceImage.id) return;
        event.preventDefault();
        if (event.dataTransfer) event.dataTransfer.dropEffect = "move";
        card.classList.add("is-drop-target");
      });
      card.addEventListener("dragleave", (event) => {
        if (!(event.relatedTarget instanceof Node) || !card.contains(event.relatedTarget)) {
          card.classList.remove("is-drop-target");
        }
      });
      card.addEventListener("drop", async (event) => {
        const draggedId = state.draggingSourceImageId;
        if (!draggedId || state.busy) return;
        event.preventDefault();
        const fromIndex = state.sourceImages.findIndex((imageItem) => imageItem.id === draggedId);
        const targetIndex = state.sourceImages.findIndex((imageItem) => imageItem.id === sourceImage.id);
        const targetRect = card.getBoundingClientRect();
        const afterTarget = event.clientX >= targetRect.left + targetRect.width / 2;
        const destination = targetIndex + Number(afterTarget) - Number(fromIndex < targetIndex);
        state.draggingSourceImageId = null;
        if (fromIndex < 0 || targetIndex < 0 || fromIndex === targetIndex) return;
        await moveSourceImage(fromIndex, destination);
      });
      card.addEventListener("dragend", () => {
        state.draggingSourceImageId = null;
        document.querySelectorAll(".source-image-card").forEach((sourceCard) => {
          sourceCard.classList.remove("is-dragging", "is-drop-target");
        });
      });
      card.addEventListener("keydown", (event) => {
        if (event.target !== card || !event.altKey || !event.shiftKey) return;
        const delta = event.key === "ArrowLeft" ? -1 : event.key === "ArrowRight" ? 1 : 0;
        if (!delta || index + delta < 0 || index + delta >= state.sourceImages.length) return;
        event.preventDefault();
        void moveSourceImage(index, index + delta);
      });
      list.insertBefore(card, addButton);
    }
    updateModeControls();
    updateResolutionPreview();
  }

  async function moveSourceImage(fromIndex, toIndex) {
    if (state.busy || !state.conversationId) return;
    const sourceImageIds = state.sourceImages.map((sourceImage) => sourceImage.id);
    if (fromIndex < 0 || fromIndex >= sourceImageIds.length || toIndex < 0 || toIndex >= sourceImageIds.length) return;
    if (fromIndex === toIndex) return;
    const [sourceImageId] = sourceImageIds.splice(fromIndex, 1);
    sourceImageIds.splice(toIndex, 0, sourceImageId);
    state.draggingSourceImageId = null;
    setBusy(true);
    try {
      const response = await api(
        `/api/conversations/${encodeURIComponent(state.conversationId)}/source-images/order`,
        { method: "PUT", body: JSON.stringify({ source_image_ids: sourceImageIds }) },
      );
      await renderSourceImages(response.source_images);
      const movedCard = [...byId("source-image-list").querySelectorAll(".source-image-card")]
        .find((card) => card.dataset.sourceImageId === sourceImageId);
      movedCard?.focus();
    } catch (error) {
      toast(error.message, true);
    } finally {
      setBusy(false);
    }
  }

  async function removeSourceImage(sourceImageId) {
    if (state.busy || !state.conversationId) return;
    setBusy(true);
    try {
      await api(
        `/api/conversations/${encodeURIComponent(state.conversationId)}/source-images/${encodeURIComponent(sourceImageId)}`,
        { method: "DELETE" },
      );
      await renderSourceImages(state.sourceImages.filter((image) => image.id !== sourceImageId));
    } catch (error) {
      toast(error.message, true);
    } finally {
      setBusy(false);
    }
  }

  async function uploadSourceImages(files) {
    if (state.busy || !state.conversationId) return;
    const mode = getMode();
    if (mode.source_images.max_count === 0) {
      toast(`${mode.label} does not accept source images.`, true);
      return;
    }
    const selectedFiles = [...files];
    const limits = state.options?.source_images;
    if (!limits) return;
    const maximum = mode.source_images.max_count;
    if (state.sourceImages.length + selectedFiles.length > maximum) {
      toast(`This mode supports at most ${maximum} source images.`, true);
      return;
    }
    setBusy(true);
    try {
      for (const file of selectedFiles) {
        if (file.size > limits.max_bytes) throw new Error(`${file.name} exceeds the 25 MiB upload limit.`);
        if (!limits.allowed_types.includes(file.type)) {
          throw new Error(`${file.name} is not a PNG, JPEG, or WebP image.`);
        }
        const uploaded = await api(
          `/api/conversations/${encodeURIComponent(state.conversationId)}/source-images`,
          { method: "POST", body: file, headers: { "Content-Type": file.type } },
        );
        await renderSourceImages([...state.sourceImages, uploaded]);
      }
    } catch (error) {
      toast(error.message, true);
    } finally {
      byId("source-image-input").value = "";
      setBusy(false);
    }
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
    const indexBadge = document.createElement("span");
    indexBadge.className = "gallery-index";
    indexBadge.setAttribute("aria-hidden", "true");
    openImage.append(indexBadge);
    const caption = document.createElement("div");
    caption.className = "gallery-caption";
    const text = document.createElement("span");
    const generationMode = state.options?.modes.find((mode) => mode.id === generation.image_mode);
    const details = [generationMode?.label || generation.image_mode || "Image generation"];
    if (generation.aspect_ratio && generation.aspect_ratio !== "Source image") {
      details.push(generation.aspect_ratio);
    }
    details.push(
      `${Number(generation.megapixels).toFixed(2)} MP`,
      `${generation.width} x ${generation.height}`,
    );
    const sourceCount = generation.source_image_ids?.length || 0;
    if (sourceCount) details.push(`${sourceCount} source images`);
    if (generation.sampler_name && generation.scheduler) {
      details.push(`${generation.sampler_name} / ${generation.scheduler}`);
    }
    details.push(`seed ${generation.seed}`);
    text.textContent = details.join(" · ");
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
      card.querySelector(".gallery-index").textContent = `${index + 1} / ${cards.length}`;
    });
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
      applyImageMode(data.conversation.image_mode || state.options.default_mode);
      const conversationRatio = data.conversation.aspect_ratio;
      if (state.options?.aspect_ratios.some((item) => item.label === conversationRatio)) {
        byId("image-aspect-ratio").value = conversationRatio;
      }
      const latestGeneration = data.generations[0];
      if (latestGeneration && (latestGeneration.image_mode || state.options.default_mode) === state.imageMode) {
        if (state.options?.aspect_ratios.some((item) => item.label === latestGeneration.aspect_ratio)) {
          byId("image-aspect-ratio").value = latestGeneration.aspect_ratio;
        }
        byId("image-megapixels").value = Number(latestGeneration.megapixels).toFixed(2);
      }
      restoreConversationSettings();
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
      await renderSourceImages(data.source_images || []);
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
      applyImageMode(state.options.default_mode);
      imagePrompt.value = "";
      if (state.options) {
        byId("image-aspect-ratio").value = state.options.default_aspect_ratio;
        byId("image-megapixels").value = Number(
          getMode(state.options.default_mode).resolution.megapixels.default,
        ).toFixed(2);
        saveConversationSettings();
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
    saveConversationSettings();
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
    saveConversationSettings();
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

  async function resumeActiveGeneration(activeGeneration) {
    if (state.generationRunId === activeGeneration.id) return;
    state.generationRunId = activeGeneration.id;
    setBusy(true);
    resetGenerationMonitor();
    if (activeGeneration.latest_event) updateGenerationMonitor(activeGeneration.latest_event);
    let afterSequence = Math.max(0, Number(activeGeneration.sequence) || 0);
    try {
      let result;
      while (!result) {
        try {
          result = await resumeGenerationStream(activeGeneration.id, afterSequence, (event) => {
            afterSequence += 1;
            updateGenerationMonitor(event);
          });
        } catch (error) {
          if (error.status === 401 || error.status === 404 || error.generationFailed) throw error;
          byId("generate-image-label").textContent = "Reconnecting to generation";
          setGenerationProgress(Number.NaN);
          await new Promise((resolve) => window.setTimeout(resolve, 1000));
        }
      }
      await selectConversation(activeGeneration.conversation_id, result.generation_id);
      if (result.cleanup_warning) toast(result.cleanup_warning, true);
      else toast("Image generated.");
      await refreshState();
    } catch (error) {
      const button = byId("generate-image");
      button.classList.add("is-error");
      button.title = error.message;
      byId("generate-image-label").textContent = error.generationFailed
        ? "Generation failed" : "Generation status unavailable";
      setGenerationProgress(Number.NaN);
      toast(error.message, true);
    } finally {
      if (state.generationRunId === activeGeneration.id) {
        state.generationRunId = null;
        setBusy(false);
        clearGenerationMonitor();
      }
    }
  }

  function normalizeSeedInput(value) {
    const seedText = value.replace(/^0+(?=\d)/, "");
    if (!/^[0-9]+$/.test(seedText) || seedText.length > 20 ||
        (seedText.length === 20 && seedText > "18446744073709551615")) {
      return null;
    }
    return seedText;
  }

  function updateSeedValidity() {
    const seedInput = byId("image-seed");
    seedInput.setCustomValidity("");
    const seedText = normalizeSeedInput(seedInput.value);
    const valid = seedText !== null && seedInput.validity.valid;
    seedInput.classList.toggle("is-invalid", !valid);
    if (valid) seedInput.removeAttribute("aria-invalid");
    else seedInput.setAttribute("aria-invalid", "true");
    return { seedInput, seedText, valid };
  }

  async function generateImage() {
    if (state.busy || !state.conversationId || !imagePrompt.value.trim()) return;
    const mode = getMode();
    const resolution = calculateResolution();
    if (!resolution) return;
    const { seedInput, seedText, valid } = updateSeedValidity();
    if (!valid) {
      seedInput.setCustomValidity("Seed must be an integer from 0 to 18446744073709551615.");
      seedInput.reportValidity();
      return;
    }
    const seed = seedText;
    saveConversationSettings();
    state.generationRunId = "starting";
    setBusy(true);
    resetGenerationMonitor();
    try {
      const result = await streamGeneration({
          conversation_id: state.conversationId,
          prompt: imagePrompt.value.trim(),
          seed,
          aspect_ratio: byId("image-aspect-ratio").value,
          megapixels: resolution.megapixels,
          image_mode: state.imageMode,
          source_image_ids: mode.source_images.max_count > 0
            ? state.sourceImages.map((sourceImage) => sourceImage.id) : [],
        }, updateGenerationMonitor, (runId) => {
          if (runId) state.generationRunId = runId;
        });
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
      state.generationRunId = null;
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
  byId("upload-source-images").addEventListener("click", () => byId("source-image-input").click());
  byId("source-image-input").addEventListener("change", (event) => {
    uploadSourceImages(event.currentTarget.files || []);
  });
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
  byId("thinking-effort").addEventListener("change", saveConversationSettings);
  byId("image-aspect-ratio").addEventListener("change", () => {
    updateResolutionPreview();
    saveConversationSettings();
  });
  byId("image-megapixels").addEventListener("input", () => {
    updateResolutionPreview();
    saveConversationSettings();
  });
  byId("image-seed").addEventListener("input", updateSeedValidity);
  byId("random-seed").addEventListener("click", () => {
    const value = new Uint32Array(1);
    crypto.getRandomValues(value);
    byId("image-seed").value = String(value[0]);
    updateSeedValidity();
  });
  byId("token-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    state.token = byId("token-input").value.trim();
    sessionStorage.setItem(tokenKey, state.token);
    byId("token-error").textContent = "";
    setBusy(true);
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
  updateModeControls();
  if (state.token) {
    setBusy(true);
    loadOptions().then(refreshState).catch((error) => toast(error.message, true));
    startSystemMetricsPolling();
  } else tokenDialog.showModal();
})();