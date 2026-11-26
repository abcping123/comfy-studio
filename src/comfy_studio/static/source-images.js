import { sourceImageRole } from "./resolution-ui.js?v=14";

export async function hashSourceFile(file) {
  const subtleCrypto = globalThis.crypto?.subtle;
  if (!subtleCrypto || typeof subtleCrypto.digest !== "function") return null;
  const bytes = await file.arrayBuffer();
  const digest = await subtleCrypto.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

export function createSourceImagesController({
  state,
  byId,
  getMode,
  api,
  fetchBlobUrl,
  setBusy,
  toast,
  updateModeControls,
  updateResolutionPreview,
}) {
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

    const resolvedUrls = await Promise.all(sourceImages.map(async (sourceImage) => {
      const existingUrl = state.sourceImageUrls.get(sourceImage.id);
      if (existingUrl) return { id: sourceImage.id, url: existingUrl, fresh: false };
      const objectUrl = await fetchBlobUrl(
        `/api/conversations/${encodeURIComponent(conversationId)}/source-images/${encodeURIComponent(sourceImage.id)}`,
        "Source image fetch failed",
      );
      return { id: sourceImage.id, url: objectUrl, fresh: Boolean(objectUrl) };
    }));

    if (renderToken !== state.sourceRenderToken || conversationId !== state.conversationId) {
      resolvedUrls.forEach((entry) => {
        if (entry.fresh && entry.url) URL.revokeObjectURL(entry.url);
      });
      return;
    }

    for (const [index, sourceImage] of sourceImages.entries()) {
      const resolved = resolvedUrls[index];
      if (!resolved?.url) {
        resolvedUrls.slice(index + 1).forEach((entry) => {
          if (entry.fresh && entry.url) URL.revokeObjectURL(entry.url);
        });
        return;
      }
      if (resolved.fresh) state.sourceImageUrls.set(sourceImage.id, resolved.url);

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
      image.src = resolved.url;

      const details = document.createElement("div");
      details.className = "source-image-details";
      const label = document.createElement("span");
      label.className = "source-image-label";
      label.textContent = `Picture ${index + 1} - ${formattedRole}`;
      details.append(label);

      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "icon-button source-image-action source-image-remove";
      remove.textContent = "×";
      remove.title = "Remove source image";
      remove.setAttribute("aria-label", `Remove Picture ${index + 1}`);
      remove.disabled = state.busy;
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
    setBusy(true);
    try {
      const activeHashes = new Set(state.sourceImages.map((sourceImage) => sourceImage.content_hash));
      const selectedHashes = new Map();
      const newHashes = new Set();
      let clientHashingAvailable = true;
      for (const file of selectedFiles) {
        if (file.size > limits.max_bytes) throw new Error(`${file.name} exceeds the 25 MiB upload limit.`);
        if (!limits.allowed_types.includes(file.type)) {
          throw new Error(`${file.name} is not a PNG, JPEG, or WebP image.`);
        }
        const digest = await hashSourceFile(file);
        selectedHashes.set(file, digest);
        if (digest === null) {
          clientHashingAvailable = false;
        } else if (!activeHashes.has(digest)) {
          newHashes.add(digest);
        }
      }
      if (clientHashingAvailable && state.sourceImages.length + newHashes.size > maximum) {
        throw new Error(`This mode supports at most ${maximum} source images.`);
      }
      for (const file of selectedFiles) {
        const uploaded = await api(
          `/api/conversations/${encodeURIComponent(state.conversationId)}/source-images`,
          { method: "POST", body: file, headers: { "Content-Type": file.type } },
        );
        if (state.sourceImages.some((sourceImage) => sourceImage.id === uploaded.id)) {
          toast("This image is already in the source list.");
          continue;
        }
        const expectedHash = selectedHashes.get(file);
        if (expectedHash !== null && uploaded.content_hash !== expectedHash) {
          throw new Error("Uploaded source image hash did not match the selected file.");
        }
        await renderSourceImages([...state.sourceImages, uploaded]);
      }
    } catch (error) {
      toast(error.message, true);
    } finally {
      byId("source-image-input").value = "";
      setBusy(false);
    }
  }

  return {
    clearSourceImages,
    moveSourceImage,
    removeSourceImage,
    renderSourceImages,
    uploadSourceImages,
  };
}
