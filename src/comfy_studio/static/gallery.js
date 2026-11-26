export function createGalleryController({
  state,
  byId,
  imageLightbox,
  emptyResult,
  fetchBlobUrl,
}) {
  function buildCard(generation, url) {
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
    return card;
  }

  async function renderGenerations(generations, expectedConversationId) {
    const renderToken = ++state.galleryRenderToken;
    const completed = generations.filter(
      (generation) => generation.status === "complete" && generation.filename,
    );
    if (!completed.length) return;
    const fetched = await Promise.all(completed.map(async (generation) => ({
      generation,
      url: await fetchBlobUrl(`/api/generations/${generation.id}/image`, "Image fetch failed"),
    })));
    if (renderToken !== state.galleryRenderToken || state.conversationId !== expectedConversationId) {
      fetched.forEach(({ url }) => {
        if (url) URL.revokeObjectURL(url);
      });
      return;
    }
    const gallery = byId("image-gallery");
    for (const { generation, url } of fetched) {
      if (!url) continue;
      state.imageUrls.add(url);
      byId("empty-result")?.remove();
      gallery.append(buildCard(generation, url));
    }
  }

  function showGeneration(generationId = null) {
    const cards = [...byId("image-gallery").querySelectorAll(".gallery-item")];
    let activeIndex = generationId
      ? cards.findIndex((card) => card.dataset.generationId === generationId)
      : cards.findIndex((card) => card.classList.contains("is-active"));
    if (activeIndex < 0) activeIndex = 0;
    cards.forEach((card, index) => {
      card.querySelector(".gallery-index").textContent = `${index + 1} / ${cards.length}`;
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

  function clearGalleryImages() {
    state.galleryRenderToken += 1;
    closeImageViewer();
    state.imageUrls.forEach((url) => URL.revokeObjectURL(url));
    state.imageUrls.clear();
    byId("image-gallery").replaceChildren(emptyResult);
    showGeneration();
  }

  return {
    clearGalleryImages,
    closeImageViewer,
    moveGeneration,
    moveImageViewer,
    openImageViewer,
    renderGenerations,
    showGeneration,
  };
}
