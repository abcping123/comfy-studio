import { createApiClient } from "./api.js?v=14";
import {
  calculateModeResolution,
  getSavedMegapixels,
  normalizeSeedInput,
  sourceImageRole,
} from "./resolution-ui.js?v=14";
import { createSourceImagesController } from "./source-images.js?v=14";
import { createGalleryController } from "./gallery.js?v=14";

(() => {
  const tokenKey = "fieldnotes.access-token";
  const conversationSettingsPrefix = "fieldnotes.conversation-settings.v1.";
  const state = {
    token: sessionStorage.getItem(tokenKey) || "",
    conversationId: null,
    options: null,
    ready: false,
    pendingDeleteId: null,
    busy: false,
    imageMode: null,
    sourceImages: [],
    sourceImageUrls: new Map(),
    sourceRenderToken: 0,
    galleryRenderToken: 0,
    draggingSourceImageId: null,
    activeChatRunId: null,
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

  function currentGenerateLabel() {
    if (state.options && state.imageMode) {
      const mode = state.options.modes.find((item) => item.id === state.imageMode);
      if (mode?.ui?.generate_label) return mode.ui.generate_label;
    }
    return "Generate image";
  }

  const {
    api,
    fetchBlobUrl,
    resumeChatRun,
    resumeGenerationStream,
    streamChat,
    streamGeneration,
  } = createApiClient({
    getToken: () => state.token,
    onUnauthorized: () => {
      sessionStorage.removeItem(tokenKey);
      state.token = "";
      if (!tokenDialog.open) tokenDialog.showModal();
    },
  });

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
    byId("generate-image-label").textContent = currentGenerateLabel();
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
        "Live progress connection lost; waiting for ComfyUI completion": "Waiting on ComfyUI",
        "Submitting workflow to ComfyUI": "Submitting workflow",
        "ComfyUI is executing the workflow": "Generating image",
        "Saving generated image to the session and output archive": "Saving image",
        "Keeping ComfyUI models loaded for the next generation": "Finishing image",
        "Image ready": "Image ready",
      };
      label.textContent = phaseLabels[event.message] || event.message || "Generating image";
      if (event.message?.startsWith("Live progress unavailable") ||
          event.message?.startsWith("Live progress connection lost")) {
        setGenerationProgress(Number.NaN);
      }
    } else if (event.type === "node" || event.type === "progress") {
      const nodeLabels = {
        LoadImage: "Loading source image",
        ResizeImageMaskNode: "Resizing source image",
        GetImageSize: "Reading source image size",
        UNETLoader: "Loading diffusion model",
        LoadLoraModel: "Loading character swap LoRA",
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
    document.querySelectorAll(".source-image-action").forEach((button) => { button.disabled = busy; });
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
      summary.textContent = streaming ? "助手正在思考…" : "思考过程";
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

  function calculateResolution() {
    if (!state.options) return null;
    return calculateModeResolution({
      options: state.options,
      mode: getMode(),
      megapixels: Number(byId("image-megapixels").value),
      aspectRatioLabel: byId("image-aspect-ratio").value,
      sourceImages: state.sourceImages,
    });
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
    const selectedMegapixels = Number.isFinite(savedMegapixels) &&
      savedMegapixels >= megapixelOptions.min && savedMegapixels <= megapixelOptions.max
      ? savedMegapixels
      : megapixelOptions.default;
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

  function isSupportedReasoningEffort(value) {
    return state.options?.reasoning_efforts?.some((item) => item.value === value) ?? false;
  }

  async function loadOptions() {
    const options = await api("/api/options");
    state.options = options;
    renderModeButtons();
    fillSelect(byId("image-aspect-ratio"), options.aspect_ratios,
      (item) => item.label, (item) => item.label, options.default_aspect_ratio);
    fillSelect(byId("thinking-effort"), options.reasoning_efforts,
      (item) => item.value, (item) => item.label, options.default_reasoning_effort);
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
    if (isSupportedReasoningEffort(thinkingEffort)) {
      saved.thinking_effort = thinkingEffort;
    }

    try {
      localStorage.setItem(`${conversationSettingsPrefix}${state.conversationId}`, JSON.stringify(saved));
    } catch {
      // Browser storage may be unavailable; preferences remain usable for this page load.
    }
  }

  function restoreConversationSettings(fallbackGeneration = null) {
    const saved = readConversationSettings();
    if (!state.options) return;

    const candidateRatio = saved.aspect_ratio || fallbackGeneration?.aspect_ratio;
    if (state.options.aspect_ratios.some((item) => item.label === candidateRatio)) {
      byId("image-aspect-ratio").value = candidateRatio;
    }
    const mode = getMode();
    const megapixelOptions = mode.resolution.megapixels;
    const savedMegapixels = getSavedMegapixels(saved, mode);
    const generationMegapixels = fallbackGeneration &&
      (fallbackGeneration.image_mode || state.options.default_mode) === mode.id
      ? Number(fallbackGeneration.megapixels)
      : Number.NaN;
    const selectedMegapixels = Number.isFinite(savedMegapixels) &&
      savedMegapixels >= megapixelOptions.min && savedMegapixels <= megapixelOptions.max
      ? savedMegapixels
      : Number.isFinite(generationMegapixels) &&
        generationMegapixels >= megapixelOptions.min && generationMegapixels <= megapixelOptions.max
        ? generationMegapixels
        : Number(megapixelOptions.default);
    byId("image-megapixels").value = selectedMegapixels.toFixed(2);
    byId("thinking-effort").value = isSupportedReasoningEffort(saved.thinking_effort)
      ? saved.thinking_effort : state.options.default_reasoning_effort;
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
    const backendId = data.chat_backend || "strata";
    if (state.options?.chat_backend?.id !== backendId) {
      await loadOptions();
      restoreConversationSettings();
      updateResolutionPreview();
    }
    const backendName = data.chat_backend_label || backendId;
    const backend = data.services[backendId];
    const comfy = data.services.comfyui;
    setServiceStatus("chat-backend", backend, backend?.available
      ? (backend.loaded ? `${backendName} loaded` : `${backendName} idle`)
      : `${backendName} offline`);
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
    state.ready = true;
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
      select.dataset.conversationId = conversation.id;
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

  function updateSelectedConversationHighlight(conversationId = state.conversationId) {
    document.querySelectorAll(".conversation-item").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.conversationId === conversationId);
    });
  }

  const gallery = createGalleryController({
    state,
    byId,
    imageLightbox,
    emptyResult,
    fetchBlobUrl,
  });

  const sourceImages = createSourceImagesController({
    state,
    byId,
    getMode,
    api,
    fetchBlobUrl,
    setBusy,
    toast,
    updateModeControls,
    updateResolutionPreview,
  });

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
      const data = await api("/api/conversations");
      renderConversationList(data.conversations || []);
      if (!state.conversationId) {
        const nextId = data.active_conversation_id || data.conversations?.[0]?.id;
        if (nextId) await selectConversation(nextId, null, true);
        else await createConversation();
      }
      toast(deletion.cleanup_warning ||
        "Conversation session deleted. Archived PNG copies in output/ remain.");
    } catch (error) {
      toast(error.message, true);
    } finally {
      byId("confirm-delete").disabled = false;
      setBusy(false);
    }
  }

  function clearImages() {
    gallery.clearGalleryImages();
    sourceImages.clearSourceImages();
  }

  async function selectConversation(conversationId, revealGenerationId = null, allowBusy = false) {
    if (state.busy && !revealGenerationId && !allowBusy) return null;
    try {
      const data = await api(`/api/conversations/${encodeURIComponent(conversationId)}`);
      state.conversationId = conversationId;
      clearImages();
      if (!revealGenerationId) clearGenerationMonitor();
      updateSelectedConversationHighlight(conversationId);
      byId("conversation-title").textContent = data.conversation.title || "New conversation";
      imagePrompt.value = data.conversation.image_prompt || "";
      applyImageMode(data.conversation.image_mode || state.options.default_mode);
      const conversationRatio = data.conversation.aspect_ratio;
      if (state.options?.aspect_ratios.some((item) => item.label === conversationRatio)) {
        byId("image-aspect-ratio").value = conversationRatio;
      } else if (state.options) {
        byId("image-aspect-ratio").value = state.options.default_aspect_ratio;
      }
      const latestGeneration = data.generations[0];
      const matchingGeneration = latestGeneration &&
        (latestGeneration.image_mode || state.options.default_mode) === state.imageMode
        ? latestGeneration
        : null;
      restoreConversationSettings(matchingGeneration);
      updatePromptCount();
      const badge = byId("context-badge");
      badge.classList.toggle("is-warning", !data.conversation.context_saved_at);
      badge.replaceChildren();
      const dot = document.createElement("i");
      const label = document.createElement("span");
      dot.setAttribute("aria-hidden", "true");
      label.textContent = data.conversation.context_saved_at ? "Conversation context saved" : "New context";
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
      await Promise.all([
        sourceImages.renderSourceImages(data.source_images || []),
        gallery.renderGenerations(data.generations.slice(0, 8), conversationId),
      ]);
      gallery.showGeneration(revealGenerationId);
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
        byId("thinking-effort").value = state.options.default_reasoning_effort;
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
    const contextLabel = result.context_saved ? "Conversation context saved" : "Context save needs attention";
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
    if (state.activeChatRunId === activeChat.id) return;
    state.activeChatRunId = activeChat.id;
    setBusy(true);
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
      if (state.activeChatRunId === activeChat.id) {
        state.activeChatRunId = null;
      }
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
    sourceImages.uploadSourceImages(event.currentTarget.files || []);
  });
  byId("older-result").addEventListener("click", () => gallery.moveGeneration(1));
  byId("newer-result").addEventListener("click", () => gallery.moveGeneration(-1));
  byId("lightbox-close").addEventListener("click", gallery.closeImageViewer);
  byId("lightbox-older").addEventListener("click", () => gallery.moveImageViewer(1));
  byId("lightbox-newer").addEventListener("click", () => gallery.moveImageViewer(-1));
  imageLightbox.addEventListener("click", (event) => {
    if (event.target === imageLightbox) gallery.closeImageViewer();
  });
  imageLightbox.addEventListener("close", () => {
    state.imageViewerGenerationId = null;
    byId("lightbox-image").removeAttribute("src");
  });
  document.addEventListener("keydown", (event) => {
    if (!imageLightbox.open) return;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      gallery.moveImageViewer(-1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      gallery.moveImageViewer(1);
    } else if (event.key === "Escape") {
      event.preventDefault();
      gallery.closeImageViewer();
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

  async function bootAuthenticatedSession() {
    setBusy(true);
    try {
      await loadOptions();
      await refreshState();
      startSystemMetricsPolling();
    } catch (error) {
      setBusy(false);
      throw error;
    } finally {
      if (!state.activeChatRunId && !state.generationRunId) {
        setBusy(false);
      }
    }
  }

  byId("token-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    state.token = byId("token-input").value.trim();
    sessionStorage.setItem(tokenKey, state.token);
    byId("token-error").textContent = "";
    try {
      await bootAuthenticatedSession();
      tokenDialog.close();
      byId("token-input").value = "";
    } catch (error) {
      state.token = "";
      sessionStorage.removeItem(tokenKey);
      byId("token-error").textContent = error.message;
    }
  });

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState !== "visible" || !state.token || !state.ready) return;
    refreshState().catch((error) => toast(error.message, true));
  });

  updatePromptCount();
  resizeChatInput();
  updateModeControls();
  if (state.token) {
    bootAuthenticatedSession().catch((error) => toast(error.message, true));
  } else tokenDialog.showModal();
})();