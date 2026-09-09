import os
import tempfile
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
from image_enchancer.vulkan import is_vulkan_available, run_vulkan_upscale, MODEL_MAP


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
    engine: str = "auto"  # "auto", "vulkan", "cpu"
    device: Optional[str] = None


class ImageEnhancementPipeline:
    """
    Unified end-to-end image enhancement pipeline.
    Coordinates Vulkan GPU acceleration (Intel Iris Xe), PyTorch super-resolution,
    GFPGAN face restoration, seamless tiling, and perceptual micro-contrast grading.
    """

    def __init__(self, config: Optional[EnhancementConfig] = None):
        self.config = config or EnhancementConfig()
        
        # Determine device
        if self.config.device:
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize face restoration engine if requested
        self.face_engine: Optional[FaceRestorationEngine] = None
        if self.config.face_restore:
            try:
                self.face_engine = FaceRestorationEngine(device=self.device)
            except Exception as ex:
                print(f"[Warning] Failed to initialize face restoration: {ex}. Continuing without face restoration.")
                self.face_engine = None

        self.post_processor = PostProcessor()
        self._sr_engine: Optional[SuperResolutionEngine] = None

    @property
    def sr_engine(self) -> SuperResolutionEngine:
        """Lazily instantiates the PyTorch CPU super-resolution engine when needed."""
        if self._sr_engine is None:
            sr_model = load_neural_model(self.config.model_name, device=self.device)
            self._sr_engine = SuperResolutionEngine(sr_model, device=self.device)
        return self._sr_engine

    def process_array(
        self,
        image_input: np.ndarray,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes enhancement pipeline on an in-memory numpy array using PyTorch.
        """
        start_time = time.perf_counter()
        h_orig, w_orig = image_input.shape[:2]

        has_alpha = image_input.shape[2] == 4 if len(image_input.shape) > 2 else False
        if has_alpha:
            bgr = image_input[:, :, :3]
            alpha = image_input[:, :, 3]
        else:
            bgr = image_input
            alpha = None

        if self.config.denoise_strength > 0.0:
            bgr = self.post_processor.pre_denoise(bgr, strength=self.config.denoise_strength)

        enhanced_bgr = self.sr_engine.upscale(
            image_bgr=bgr,
            tile_size=self.config.tile_size,
            tile_pad=self.config.tile_pad,
            target_scale=self.config.scale,
            progress_callback=progress_callback,
        )

        h_enhanced, w_enhanced = enhanced_bgr.shape[:2]

        faces_restored = 0
        if self.face_engine is not None and self.config.face_restore:
            if progress_callback:
                progress_callback(90, 100, "Restoring human facial features (GFPGAN)...")
            enhanced_bgr, faces_restored = self.face_engine.restore_faces(
                enhanced_bgr,
                fidelity=self.config.face_fidelity,
            )

        if self.config.color_match:
            enhanced_bgr = self.post_processor.match_color_distribution(
                source_bgr=enhanced_bgr,
                reference_bgr=bgr,
            )

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
        Processes an image file, automatically choosing hardware-accelerated Vulkan GPU
        when available for fast execution, or falling back to tiled PyTorch CPU.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        # Check if Vulkan GPU acceleration is eligible
        use_vulkan = (
            self.config.engine in ["auto", "vulkan"]
            and is_vulkan_available()
            and self.config.model_name in MODEL_MAP
        )

        if use_vulkan:
            start_time = time.perf_counter()
            temp_dir = Path(tempfile.gettempdir()) / "image_enchancer_pipeline"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_upscaled = temp_dir / f"vk_sr_{int(time.time() * 1000)}.png"

            if progress_callback:
                progress_callback(1, 100, "Starting Intel Iris Xe Vulkan GPU engine...")

            target_scale = int(self.config.scale) if self.config.scale in [2, 3, 4] else 4
            success = run_vulkan_upscale(
                input_path=input_path,
                output_path=temp_upscaled,
                model_name=self.config.model_name,
                scale=target_scale,
                tile_size=0,  # Auto tile size on GPU
                progress_callback=progress_callback,
            )

            if success and temp_upscaled.exists():
                # Read original image for color matching and dimensions
                orig_img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
                h_orig, w_orig = orig_img.shape[:2]

                # Read upscaled image
                enhanced_bgr = cv2.imread(str(temp_upscaled), cv2.IMREAD_COLOR)
                temp_upscaled.unlink(missing_ok=True)

                # Face restoration stage
                faces_restored = 0
                if self.face_engine is not None and self.config.face_restore:
                    if progress_callback:
                        progress_callback(85, 100, "Restoring facial landmarks (GFPGAN)...")
                    enhanced_bgr, faces_restored = self.face_engine.restore_faces(
                        enhanced_bgr,
                        fidelity=self.config.face_fidelity,
                    )

                # Color matching stage
                if self.config.color_match:
                    ref_bgr = orig_img[:, :, :3] if len(orig_img.shape) == 3 and orig_img.shape[2] == 4 else orig_img
                    enhanced_bgr = self.post_processor.match_color_distribution(
                        source_bgr=enhanced_bgr,
                        reference_bgr=ref_bgr,
                    )

                # Micro-contrast and sharpening
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

                # Custom scale downsampling if needed
                h_enhanced, w_enhanced = enhanced_bgr.shape[:2]
                if self.config.scale is not None and abs(self.config.scale - target_scale) > 1e-3:
                    final_w = int(round(w_orig * self.config.scale))
                    final_h = int(round(h_orig * self.config.scale))
                    enhanced_bgr = cv2.resize(enhanced_bgr, (final_w, final_h), interpolation=cv2.INTER_LANCZOS4)
                    h_enhanced, w_enhanced = enhanced_bgr.shape[:2]

                output_path.parent.mkdir(parents=True, exist_ok=True)
                ext = output_path.suffix.lower()
                params = []
                if ext in [".jpg", ".jpeg"]:
                    params = [cv2.IMWRITE_JPEG_QUALITY, 98]
                elif ext == ".png":
                    params = [cv2.IMWRITE_PNG_COMPRESSION, 4]
                elif ext == ".webp":
                    params = [cv2.IMWRITE_WEBP_QUALITY, 98]

                cv2.imwrite(str(output_path), enhanced_bgr, params)
                elapsed = time.perf_counter() - start_time

                return {
                    "elapsed_seconds": elapsed,
                    "original_dimensions": (w_orig, h_orig),
                    "output_dimensions": (w_enhanced, h_enhanced),
                    "effective_scale": w_enhanced / w_orig,
                    "faces_restored": faces_restored,
                    "device": "Intel Iris Xe (Vulkan GPU)",
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                }

        # Fallback to pure PyTorch CPU pipeline
        img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to decode image file: {input_path}")

        enhanced_img, telemetry = self.process_array(img, progress_callback=progress_callback)

        output_path.parent.mkdir(parents=True, exist_ok=True)
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
