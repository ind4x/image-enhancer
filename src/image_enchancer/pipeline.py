import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
import cv2
import numpy as np
import torch

from image_enchancer.models import load_neural_model, resolve_model_path
from image_enchancer.super_resolution import SuperResolutionEngine
from image_enchancer.face_restore import FaceRestorationEngine
from image_enchancer.postprocess import PostProcessor


@dataclass
class EnhancementConfig:
    """Configuration options for the complete image enhancement pipeline."""
    model_name: str = "realesrgan-x4plus"
    scale: Optional[float] = None
    face_restore: bool = True
    face_fidelity: float = 0.85
    tile_size: int = 512
    tile_pad: int = 32
    denoise_strength: float = 0.0
    clarity_strength: float = 1.2
    sharpen_strength: float = 0.35
    color_match: bool = True
    device: Optional[str] = None


class ImageEnhancementPipeline:
    """
    Unified end-to-end image enhancement pipeline.
    Combines deep generative super-resolution, blind face restoration,
    tiled memory-bounded processing, and perceptual micro-contrast grading.
    """

    def __init__(self, config: Optional[EnhancementConfig] = None):
        self.config = config or EnhancementConfig()
        
        # Determine device
        if self.config.device:
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize super-resolution engine
        sr_model = load_neural_model(self.config.model_name, device=self.device)
        self.sr_engine = SuperResolutionEngine(sr_model, device=self.device)

        # Initialize face restoration engine lazily or eagerly
        self.face_engine: Optional[FaceRestorationEngine] = None
        if self.config.face_restore:
            try:
                self.face_engine = FaceRestorationEngine(device=self.device)
            except Exception as ex:
                print(f"[Warning] Failed to initialize face restoration: {ex}. Continuing without face restoration.")
                self.face_engine = None

        self.post_processor = PostProcessor()

    def process_array(
        self,
        image_input: np.ndarray,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes enhancement pipeline on an in-memory numpy array.

        :param image_input: BGR or BGRA numpy array (uint8)
        :param progress_callback: Optional callback for progress monitoring
        :return: (Enhanced image array, dictionary of telemetry metrics)
        """
        start_time = time.perf_counter()
        h_orig, w_orig = image_input.shape[:2]

        # Handle alpha channel
        has_alpha = image_input.shape[2] == 4 if len(image_input.shape) > 2 else False
        if has_alpha:
            bgr = image_input[:, :, :3]
            alpha = image_input[:, :, 3]
        else:
            bgr = image_input
            alpha = None

        # Stage 1: Optional Pre-Denoising
        if self.config.denoise_strength > 0.0:
            bgr = self.post_processor.pre_denoise(bgr, strength=self.config.denoise_strength)

        # Stage 2: Super-Resolution (with seamless tiling)
        enhanced_bgr = self.sr_engine.upscale(
            image_bgr=bgr,
            tile_size=self.config.tile_size,
            tile_pad=self.config.tile_pad,
            target_scale=self.config.scale,
            progress_callback=progress_callback,
        )

        h_enhanced, w_enhanced = enhanced_bgr.shape[:2]

        # Stage 3: Face Restoration (GFPGAN on upscaled canvas)
        faces_restored = 0
        if self.face_engine is not None and self.config.face_restore:
            enhanced_bgr, faces_restored = self.face_engine.restore_faces(
                enhanced_bgr,
                fidelity=self.config.face_fidelity,
            )

        # Stage 4: Color Distribution Preservation
        if self.config.color_match:
            enhanced_bgr = self.post_processor.match_color_distribution(
                source_bgr=enhanced_bgr,
                reference_bgr=bgr,
            )

        # Stage 5: Perceptual Micro-Contrast & Detail Sharpening
        if self.config.clarity_strength > 0.0:
            enhanced_bgr = self.post_processor.apply_clahe(
                enhanced_bgr,
                clip_limit=self.config.clarity_strength,
            )

        if self.config.sharpen_strength > 0.0:
            enhanced_bgr = self.post_processor.apply_unsharp_mask(
                enhanced_bgr,
                strength=self.config.sharpen_strength,
            )

        # Recompose alpha channel if present
        if has_alpha and alpha is not None:
            upscaled_alpha = cv2.resize(
                alpha,
                (w_enhanced, h_enhanced),
                interpolation=cv2.INTER_LANCZOS4,
            )
            final_output = cv2.merge([
                enhanced_bgr[:, :, 0],
                enhanced_bgr[:, :, 1],
                enhanced_bgr[:, :, 2],
                upscaled_alpha,
            ])
        else:
            final_output = enhanced_bgr

        elapsed = time.perf_counter() - start_time

        telemetry = {
            "elapsed_seconds": elapsed,
            "original_dimensions": (w_orig, h_orig),
            "output_dimensions": (w_enhanced, h_enhanced),
            "effective_scale": w_enhanced / w_orig,
            "faces_restored": faces_restored,
            "device": str(self.device),
        }

        return final_output, telemetry

    def process_file(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Loads an image from disk, processes it, and writes the enhanced image to disk.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        # Read image including possible alpha channel
        img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to decode image file: {input_path}")

        enhanced_img, telemetry = self.process_array(img, progress_callback=progress_callback)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Optimal quality parameters based on file format
        ext = output_path.suffix.lower()
        params = []
        if ext in [".jpg", ".jpeg"]:
            params = [cv2.IMWRITE_JPEG_QUALITY, 98]
        elif ext == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 4]
        elif ext == ".webp":
            params = [cv2.IMWRITE_WEBP_QUALITY, 98]

        success = cv2.imwrite(str(output_path), enhanced_img, params)
        if not success:
            raise IOError(f"Failed to write enhanced image to: {output_path}")

        telemetry["input_path"] = str(input_path)
        telemetry["output_path"] = str(output_path)
        return telemetry
