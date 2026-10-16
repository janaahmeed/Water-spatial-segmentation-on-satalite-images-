
const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file");
const statusText = document.getElementById("status");
const errorText = document.getElementById("error");
const button = document.getElementById("submit-button");

const resultSection = document.getElementById("result-section");
const originalPreview = document.getElementById("original-preview");
const maskPreview = document.getElementById("mask-preview");
const overlayPreview = document.getElementById("overlay-preview");
const summary = document.getElementById("summary");
const downloadMask = document.getElementById("download-mask");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = fileInput.files[0];

  if (!file) {
    errorText.hidden = false;
    errorText.textContent = "Please select a GeoTIFF.";
    return;
  }

  errorText.hidden = true;
  resultSection.hidden = true;
  button.disabled = true;
  statusText.textContent = "Validating raster and running inference...";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    originalPreview.src = data.original_rgb;
    maskPreview.src = data.binary_mask;
    overlayPreview.src = data.water_overlay;

    downloadMask.href = data.binary_mask;

    const m = data.metadata;
    summary.textContent =
      `${m.width} × ${m.height} pixels · ` +
      `${m.bands} bands · CRS: ${m.crs} · ` +
      `${m.water_percentage}% predicted water`;

    resultSection.hidden = false;
    statusText.textContent = "Prediction completed successfully.";

  } catch (error) {
    errorText.hidden = false;
    errorText.textContent = error.message;
    statusText.textContent = "Prediction could not be completed.";
  } finally {
    button.disabled = false;
  }
});
