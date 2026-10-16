import base64
from io import BytesIO

import numpy as np
import pandas as pd
import rasterio
from PIL import Image




ALL_CHANNELS = ["Coastal aerosol", "B", "G", "R", "NIR", "SWIR1", "SWIR2",
                "QA Band", "Merit DEM", "Copernicus DEM", "ESA world cover map", "Water occ prob"]   # real band order

FEATURE_COLS = ["Coastal aerosol", "B", "G", "R", "NIR", "SWIR1", "SWIR2",
                "QA Band", "Merit DEM", "Copernicus DEM", "ESA world cover map"]   # 11 input channels
TARGET_COL   = 'Water occ prob'                                                                    # 1 target

IMG_SIZE   = 128        # every image is 128 x 128
PATCH_SIZE = 16         # UNet input 16 x 16
BATCH_SIZE = 64


N_IMAGES = 306 // (IMG_SIZE * IMG_SIZE)

band_min = np.array([
    -1390, -1169, -722, -684, -412, -334,
    -251, 64, -9999, 8, 10
], dtype=np.float32)

band_max = np.array([
    6568, 9659, 11368, 12041, 15841, 15252,
    14647, 255, 2784, 2762, 100
], dtype=np.float32)


def normalize_bands(image, band_minN=band_min,
                    band_maxN=band_max, nodata=None):

    image = np.asarray(image, dtype=np.float32)
    band_minN = np.asarray(band_minN, dtype=np.float32)
    band_maxN = np.asarray(band_maxN, dtype=np.float32)

    if image.ndim != 3:
        raise ValueError("Expected shape (H, W, C).")

    if band_minN.ndim != 1 or band_maxN.ndim != 1:
        raise ValueError("Min/max values must be one-dimensional per-band arrays.")

    if image.shape[-1] != len(band_minN):
        raise ValueError(
            f"Expected {len(band_minN)} channels, "
            f"got {image.shape[-1]}."
        )

    if len(band_minN) != len(band_maxN):
        raise ValueError("Min/max arrays must have equal lengths.")

    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or infinite values.")

    if np.any(band_maxN <= band_minN):
        raise ValueError("Every maximum must exceed its minimum.")

    # Broadcast per-band limits over H and W
    mins = band_minN.reshape(1, 1, -1)
    maxs = band_maxN.reshape(1, 1, -1)

    # Mark nodata pixels before normalization
    if nodata is not None:
        if np.isscalar(nodata):
            valid = image != nodata
        else:
            valid = np.ones(image.shape, dtype=bool)
            for c, value in enumerate(nodata):
                if value is not None:
                    valid[..., c] = image[..., c] != value
    else:
        valid = np.ones(image.shape, dtype=bool)

    normalized = (image - mins) / (maxs - mins)
    normalized = np.clip(normalized, 0.0, 1.0)

    # Replace invalid pixels with zero, but retain a validity mask
    normalized[~valid] = 0.0

    return normalized.astype(np.float32), valid


# def create_patches(df, img_size=IMG_SIZE, patch=PATCH_SIZE):
  
#     g = img_size // patch                                   # patches per side
#     C = len(FEATURE_COLS)
#     WATER_THRESHOLD = 75.0

#     feats = df[FEATURE_COLS].to_numpy(np.float32).reshape( img_size, img_size, C)

#     # Convert water occurrence probability/percentage -> binary mask
#     target_raw = (
#         df[TARGET_COL]
#         .to_numpy(np.float32)
#         .reshape( img_size, img_size)
#     )

#     target = (target_raw >= WATER_THRESHOLD).astype(np.float32)

#     X = (
#         feats
#         .reshape(g, patch, g, patch, C)
#         .transpose(0, 2, 1, 3, 4)
#         .reshape(g * g, patch, patch, C)
#     )

#     y = target.reshape(g, patch, g, patch).transpose(0, 2, 1, 3).reshape(g * g, patch, patch)


#     return X, y



def create_inference_patches(image, patch=PATCH_SIZE):
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError("Input image must have shape (height, width, channels).")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Input image must have a non-zero height and width.")
    if patch <= 0:
        raise ValueError("Patch size must be a positive integer.")

    height, width, channels = image.shape
    pad_height = (-height) % patch
    pad_width = (-width) % patch
    if pad_height or pad_width:
        image = np.pad(
            image,
            ((0, pad_height), (0, pad_width), (0, 0)),
            mode="reflect",
        )

    rows, columns = image.shape[0] // patch, image.shape[1] // patch
    patches = (
        image.reshape(rows, patch, columns, patch, channels)
        .transpose(0, 2, 1, 3, 4)
        .reshape(rows * columns, patch, patch, channels)
    )
    return patches, pad_height, pad_width


def reconstruct_mask(patch_masks, height=128, width=128, patch=PATCH_SIZE):
    #Stitch image patches into a mask and crop it to the original size >>>>>>
    rows = (height + patch - 1) // patch
    columns = (width + patch - 1) // patch
    padded_mask = (
        patch_masks.reshape(rows, columns, patch, patch)
        .transpose(0, 2, 1, 3)
        .reshape(rows * patch, columns * patch)
    )
    return padded_mask[:height, :width]

def validate_and_read_raster(file_bytes, filename):
    #Read model bands from an uploaded GeoTIFF as a float32 HWC array
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    expected_bands = len(FEATURE_COLS)
    target_band_present = (
        len(ALL_CHANNELS) == expected_bands + 1
        and ALL_CHANNELS[:expected_bands] == FEATURE_COLS
        and ALL_CHANNELS[-1] == TARGET_COL
    )
    allowed_band_counts = {expected_bands}
    if target_band_present:
        allowed_band_counts.add(len(ALL_CHANNELS))

    try:
        with rasterio.MemoryFile(file_bytes) as memfile:
            with memfile.open() as dataset:
                if dataset.count not in allowed_band_counts:
                    band_label = "band" if dataset.count == 1 else "bands"
                    raise ValueError(
                        f"The uploaded file '{filename}' contains "
                        f"{dataset.count} {band_label}; this model requires "
                        f"{expected_bands} input bands"
                        + (
                            f" (or {len(ALL_CHANNELS)} bands when the final "
                            f"'{TARGET_COL}' band is included)."
                            if target_band_present
                            else "."
                        )
                    )

                metadata = {
                    "width": dataset.width,
                    "height": dataset.height,
                    "crs": str(dataset.crs) if dataset.crs else None,
                    "count": dataset.count,
                }

                # Rasterio returns (channels, height, width); the model expects
                # (height, width, channels). Ignore the target band if present.
                image = np.moveaxis(
                    dataset.read(indexes=list(range(1, expected_bands + 1))),
                    0,
                    -1,
                ).astype(
                    np.float32, copy=False
                )
                return image, metadata
    except rasterio.errors.RasterioError as exc:
        raise ValueError(
            f"The uploaded file '{filename}' is not a readable GeoTIFF. "
            "Please choose a valid .tif or .tiff file."
        ) from exc
    

def create_rgb_composite(image):
    rgb = image[:, :, [3, 2, 1]] #if image.shape[2] >= 4 else image[:, :, :3]
    rgb_min, rgb_max = rgb.min(), rgb.max()
    if rgb_max > rgb_min:
        rgb = (255 * (rgb - rgb_min) / (rgb_max - rgb_min))#normalization per image 
    return np.clip(rgb, 0, 255).astype(np.uint8)

def create_overlay(rgb, mask):
    overlay = rgb.copy()
    # Apply a blue tint where mask == 255
    water_indices = mask == 255
    overlay[water_indices, 0] = 0   # Red channel
    overlay[water_indices, 1] = 100 # Green channel
    overlay[water_indices, 2] = 255 # Blue channel
    
    #  transparency effect (smooth)
    alpha = 0.4
    blended = (1 - alpha) * rgb + alpha * overlay
    return np.clip(blended, 0, 255).astype(np.uint8)

def png_data_url(arr):
    """Convert a numpy image array into a base64 PNG data URL."""
    image = Image.fromarray(arr)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"