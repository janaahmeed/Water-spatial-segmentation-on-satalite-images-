from pathlib import Path
import numpy as np
import torch
import segmentation_models_pytorch as smp
from preprocessing import (
    normalize_bands,
    create_inference_patches,
    reconstruct_mask,
    FEATURE_COLS,
    band_min,
    band_max
)
Band_min=band_min
Band_max=band_max
THRESHOLD = 0.75

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

class WaterSegmentationService:
    def __init__(
        self,
        checkpoint_path,
        normalization_path,
        device=None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=len(FEATURE_COLS),
            classes=1,
            activation=None,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )

        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        with np.load(normalization_path) as stats:
            self.band_min = Band_min
            self.band_max = Band_max

        if (
            len(self.band_min) != len(FEATURE_COLS)
            or len(self.band_max) != len(FEATURE_COLS)
        ):
            raise ValueError("Normalization statistics have wrong channel count.")

    @torch.inference_mode()
    def predict(self, image):
        height, width, channels = image.shape

        if channels != len(FEATURE_COLS):
            raise ValueError(
                f"Expected {len(FEATURE_COLS)} input channels, "
                f"received {channels}."
            )

        normalized, valid = normalize_bands(
            image, self.band_min, self.band_max
        )

        patches, _, _ = create_inference_patches(normalized)
        patch_predictions = []
#mask per patch then compine    
        for start in range(0, len(patches), 256):
            #contigous memory 
            batch = np.ascontiguousarray(
                patches[start:start + 256].transpose(0, 3, 1, 2)
            )
            x = torch.from_numpy(batch).to(self.device)
            probabilities = torch.sigmoid(self.model(x))[:, 0]
            patch_predictions.append(
                (probabilities >= THRESHOLD).cpu().numpy().astype(np.uint8)
            )

        patch_masks = np.concatenate(patch_predictions, axis=0)
        mask = reconstruct_mask(patch_masks, height, width)
        mask[~valid.all(axis=-1)] = 0

        return (mask * 255).astype(np.uint8) 