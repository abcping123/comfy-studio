export function roundToEven(value) {
  const lower = Math.floor(value);
  const fraction = value - lower;
  const tolerance = Number.EPSILON * Math.max(1, Math.abs(value)) * 2;
  if (Math.abs(fraction - 0.5) > tolerance) return Math.round(value);
  return lower % 2 === 0 ? lower : lower + 1;
}

export function sourceImageRole(mode, index) {
  const roles = mode.source_images.roles;
  return roles.length ? roles[Math.min(index, roles.length - 1)] : "source image";
}

export function getSavedMegapixels(saved, mode) {
  const modeSettings = saved.mode_settings;
  const modeValue = modeSettings && typeof modeSettings === "object" && !Array.isArray(modeSettings)
    ? modeSettings[mode.id]?.megapixels : undefined;
  return modeValue === undefined ? Number.NaN : Number(modeValue);
}

export function calculateModeResolution({ options, mode, megapixels, aspectRatioLabel, sourceImages }) {
  if (!options || !mode) return null;
  const megapixelOptions = mode.resolution.megapixels;
  if (!Number.isFinite(megapixels) || megapixels < megapixelOptions.min ||
      megapixels > megapixelOptions.max) return null;
  const multiple = options.resolution_multiple;
  if (mode.resolution.strategy === "source-image") {
    const base = sourceImages[0];
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
  const ratio = options.aspect_ratios.find((item) => item.label === aspectRatioLabel);
  if (!ratio) return null;
  const scale = Math.sqrt(megapixels * 1024 * 1024 / (ratio.width * ratio.height));
  return {
    width: roundToEven(ratio.width * scale / multiple) * multiple,
    height: roundToEven(ratio.height * scale / multiple) * multiple,
    megapixels,
    multiple,
  };
}

export function normalizeSeedInput(value) {
  const seedText = value.replace(/^0+(?=\d)/, "");
  if (!/^[0-9]+$/.test(seedText) || seedText.length > 20 ||
      (seedText.length === 20 && seedText > "18446744073709551615")) {
    return null;
  }
  return seedText;
}
