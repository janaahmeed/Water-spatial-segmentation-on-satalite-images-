from pathlib import Path
import time
import uuid
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from model_service import WaterSegmentationService
from logging_config import setup_logger
import rasterio
import base64
from io import BytesIO
from PIL import Image
from preprocessing import (
    FEATURE_COLS,
    band_min, 
    band_max,
    create_inference_patches,
    create_rgb_composite,
    create_overlay,
    png_data_url,
    validate_and_read_raster,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 320 * 320  # 50 MB

logger = setup_logger()

INPUT_CHANNELS = len(FEATURE_COLS)


service = WaterSegmentationService(
    checkpoint_path=str(CHECKPOINT_DIR / "best_model.pth"),
    normalization_path=str(CHECKPOINT_DIR / "normalization.npz"),
)


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(service.device),
        "input_channels": INPUT_CHANNELS,
    })


@app.errorhandler(413)
def upload_too_large(error):
    logger.warning("event=upload_rejected | reason=file_too_large")
    return jsonify({
        "error": "File exceeds the 50 MB upload limit."
    }), 413


@app.post("/predict")
def predict():
    started = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]

    uploaded = request.files.get("file")
    filename = uploaded.filename if uploaded else ""

    def log_failure(event, message, level="warning"):
        elapsed = round(time.perf_counter() - started, 3)
        getattr(logger, level)(
            "request_id=%s | event=%s | filename=%r | "
            "duration_seconds=%.3f | error=%r",
            request_id, event, filename, elapsed, message
        )

    if uploaded is None:
        log_failure("validation_failed", "Missing upload field 'file'.")
        return jsonify({
            "error": "Missing upload field 'file'.",
            "request_id": request_id
        }), 400

    if not filename:
        log_failure("validation_failed", "No file selected.")
        return jsonify({
            "error": "Choose a file first.",
            "request_id": request_id
        }), 400

    file_bytes = uploaded.read()

    if not file_bytes:
        log_failure("validation_failed", "Uploaded file is empty.")
        return jsonify({
            "error": "Uploaded file is empty.",
            "request_id": request_id
        }), 400

    try:
        image, metadata = validate_and_read_raster(
            file_bytes, filename
        )

        # image should have shape (height, width, channels)
        if image.ndim != 3 or image.shape[-1] != INPUT_CHANNELS:
            raise ValueError(
                f"Expected {INPUT_CHANNELS} input channels; "
                f"received shape {image.shape}."
            )

        mask = service.predict(image)
        mask = np.asarray(mask)

        if mask.shape != image.shape[:2]:
            raise ValueError(
                "Predicted mask dimensions do not match input."
            )

        if not np.isin(mask, [0, 255]).all():
            raise ValueError(
                "Predicted mask must contain only 0 and 255."
            )

        mask = mask.astype(np.uint8)

        rgb = create_rgb_composite(image)
        overlay = create_overlay(rgb, mask)

        water_pixels = int(np.count_nonzero(mask))
        total_pixels = int(mask.size)
        water_percentage = round(
            100.0 * water_pixels / total_pixels, 2
        )

        # Save input and prediction artifacts
        request_dir = OUTPUT_DIR / request_id
        request_dir.mkdir(parents=True, exist_ok=False)

        (request_dir / "input.tif").write_bytes(file_bytes)
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(
            request_dir / "original_rgb.png"
        )
        Image.fromarray(mask).save(
            request_dir / "predicted_mask.png"
        )
        Image.fromarray(np.asarray(overlay, dtype=np.uint8)).save(
            request_dir / "water_overlay.png"
        )

        duration = round(time.perf_counter() - started, 3)

        logger.info(
            "request_id=%s | event=predict_success | filename=%r | "
            "width=%d | height=%d | bands=%d | water_pixels=%d | "
            "water_pct=%.2f | duration_seconds=%.3f | saved=true",
            request_id,
            filename,
            metadata["width"],
            metadata["height"],
            INPUT_CHANNELS,
            water_pixels,
            water_percentage,
            duration,
        )

        return jsonify({
            "request_id": request_id,
            "original_rgb": png_data_url(rgb),
            "binary_mask": png_data_url(mask),
            "water_overlay": png_data_url(overlay),
            "metadata": {
                "width": metadata["width"],
                "height": metadata["height"],
               # "crs": metadata["crs"],
                "bands": INPUT_CHANNELS,
                "water_pixels": water_pixels,
                "water_percentage": water_percentage,
                "duration_seconds": duration,
            },
            "saved": True,
        }), 200

    except ValueError as exc:
        log_failure("validation_failed", str(exc))
        return jsonify({
            "error": str(exc),
            "request_id": request_id
        }), 400

    except Exception:
        logger.exception(
            "request_id=%s | event=predict_failed | filename=%r",
            request_id, filename
        )
        return jsonify({
            "error": "Prediction failed. Check logs/predictions.log.",
            "request_id": request_id
        }), 500


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
