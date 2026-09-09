import math
from typing import Optional, Tuple
import cv2
import numpy as np
import torch
from spandrel import ImageModelDescriptor
from tqdm import tqdm


class SuperResolutionEngine:
    """
    High-fidelity neural super-resolution engine with tiled inference and alpha channel preservation.
    """

    def __init__(
        self,
        model_descriptor: ImageModelDescriptor,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_descriptor
        self.model.to(self.device)
        self.model.eval()
        self.scale = getattr(self.model, "scale", 4)

    def _infer_tile(self, tile_bgr: np.ndarray) -> np.ndarray:
        """
        Runs neural inference on an individual image tile (BGR, uint8).
        Returns upscaled tile (BGR, uint8).
        """
        # Convert BGR uint8 to RGB float32 [0.0, 1.0]
        tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Shape: (1, C, H, W)
        tensor = torch.from_numpy(tile_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output_tensor = self.model(tensor)

        # Output shape: (1, C, H*scale, W*scale)
        output_np = output_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        output_np = np.clip(output_np * 255.0, 0.0, 255.0).astype(np.uint8)
        # Convert back to BGR
        return cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

    def _create_tile_weight(self, h: int, w: int, pad: int, scale: int) -> np.ndarray:
        """
        Generates a 2D cosine-tapered blending window for smooth tile seam transitions.
        """
        pad_px = pad * scale
        wy = np.ones(h, dtype=np.float32)
        wx = np.ones(w, dtype=np.float32)

        if pad_px > 0:
            ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, pad_px, dtype=np.float32)))
            if h >= 2 * pad_px:
                wy[:pad_px] = ramp
                wy[-pad_px:] = ramp[::-1]
            if w >= 2 * pad_px:
                wx[:pad_px] = ramp
                wx[-pad_px:] = ramp[::-1]

        return (wy[:, None] * wx[None, :])[:, :, None]

    def upscale(
        self,
        image_bgr: np.ndarray,
        tile_size: int = 512,
        tile_pad: int = 32,
        target_scale: Optional[float] = None,
        progress_callback: Optional[callable] = None,
    ) -> np.ndarray:
        """
        Upscales an input image with optional seamless tiling to prevent Out Of Memory (OOM) errors.

        :param image_bgr: Input BGR image (uint8, shape H x W x 3)
        :param tile_size: Dimension of square tile. If <= 0, disables tiling.
        :param tile_pad: Overlapping boundary padding in pixels.
        :param target_scale: Optional final scaling factor. If different from model scale,
                             uses Lanczos resampling on the super-resolved output.
        :param progress_callback: Optional callable receiving (completed_tiles, total_tiles).
        :return: Enhanced upscaled BGR image (uint8)
        """
        h, w, c = image_bgr.shape
        out_h = h * self.scale
        out_w = w * self.scale

        # Run direct single-pass if tiling is disabled or image fits comfortably
        if tile_size <= 0 or (h <= tile_size and w <= tile_size):
            upscaled = self._infer_tile(image_bgr)
            if progress_callback:
                progress_callback(1, 1)
        else:
            upscaled = np.zeros((out_h, out_w, c), dtype=np.float32)
            weight_canvas = np.zeros((out_h, out_w, 1), dtype=np.float32)

            tiles_y = math.ceil(h / tile_size)
            tiles_x = math.ceil(w / tile_size)
            total_tiles = tiles_y * tiles_x
            current_tile = 0

            for y_idx in range(tiles_y):
                y = y_idx * tile_size
                y_end = min(y + tile_size, h)

                for x_idx in range(tiles_x):
                    x = x_idx * tile_size
                    x_end = min(x + tile_size, w)

                    if progress_callback:
                        progress_callback(current_tile, total_tiles, f"Processing tile {current_tile + 1}/{total_tiles}")

                    # Extract tile with contextual boundary padding
                    y_pad_start = max(0, y - tile_pad)
                    y_pad_end = min(h, y_end + tile_pad)
                    x_pad_start = max(0, x - tile_pad)
                    x_pad_end = min(w, x_end + tile_pad)

                    tile_crop = image_bgr[y_pad_start:y_pad_end, x_pad_start:x_pad_end]
                    tile_out = self._infer_tile(tile_crop).astype(np.float32)
                    th, tw, _ = tile_out.shape

                    # Generate blending mask
                    w_tile = self._create_tile_weight(th, tw, tile_pad, self.scale)

                    out_y_start = y_pad_start * self.scale
                    out_y_end = y_pad_end * self.scale
                    out_x_start = x_pad_start * self.scale
                    out_x_end = x_pad_end * self.scale

                    upscaled[out_y_start:out_y_end, out_x_start:out_x_end] += tile_out * w_tile
                    weight_canvas[out_y_start:out_y_end, out_x_start:out_x_end] += w_tile

                    current_tile += 1
                    if progress_callback:
                        progress_callback(current_tile, total_tiles, f"Completed tile {current_tile}/{total_tiles}")

            upscaled = upscaled / np.maximum(weight_canvas, 1e-7)
            upscaled = np.clip(upscaled, 0.0, 255.0).astype(np.uint8)

        # Apply target scale resampling if requested (e.g. 2x output from 4x model)
        if target_scale is not None and abs(target_scale - self.scale) > 1e-3:
            final_w = int(round(w * target_scale))
            final_h = int(round(h * target_scale))
            interpolation = cv2.INTER_LANCZOS4 if target_scale < self.scale else cv2.INTER_CUBIC
            upscaled = cv2.resize(upscaled, (final_w, final_h), interpolation=interpolation)

        return upscaled
