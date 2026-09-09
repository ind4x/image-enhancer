#!/usr/bin/env python3
"""
Self-contained, zero-boilerplate neural image enhancer.
Upscales and restores images to 4x resolution using Real-ESRGAN with automatic tiling.
"""

import sys
import urllib.request
from pathlib import Path
import cv2
import numpy as np
import torch
from spandrel import ModelLoader

MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_PATH = Path(__file__).parent / "models" / "RealESRGAN_x4plus.pth"


def ensure_model() -> Path:
    """Auto-downloads Real-ESRGAN model weights if missing."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading model weights to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")
    return MODEL_PATH


def infer_tile(model, tile_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    """Runs single tile inference through the neural network."""
    rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(tensor)
    out_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_np = np.clip(out_np * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)


def enhance_image(
    input_path: str | Path,
    output_path: str | Path | None = None,
    tile_size: int = 512,
    tile_pad: int = 32,
) -> Path:
    """
    Enhances an image to 4x resolution with seamless tiling to prevent OOM.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_enhanced{input_path.suffix}"
    else:
        output_path = Path(output_path)

    # Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_file = ensure_model()
    loader = ModelLoader(device=device)
    model = loader.load_from_file(str(model_file))
    model.eval()

    # Read image (including alpha if present)
    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not decode image: {input_path}")

    has_alpha = len(img.shape) == 3 and img.shape[2] == 4
    bgr = img[:, :, :3] if has_alpha else img
    alpha = img[:, :, 3] if has_alpha else None

    h, w = bgr.shape[:2]
    scale = model.scale

    # Processing: Direct if small, Tiled if large
    if h <= tile_size and w <= tile_size:
        upscaled_bgr = infer_tile(model, bgr, device)
    else:
        out_h, out_w = h * scale, w * scale
        upscaled_bgr = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weight_canvas = np.zeros((out_h, out_w, 1), dtype=np.float32)

        # Pre-compute 2D cosine blending window
        def make_weight(th: int, tw: int) -> np.ndarray:
            pad_px = tile_pad * scale
            wy, wx = np.ones(th, dtype=np.float32), np.ones(tw, dtype=np.float32)
            if pad_px > 0:
                ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, pad_px, dtype=np.float32)))
                if th >= 2 * pad_px:
                    wy[:pad_px], wy[-pad_px:] = ramp, ramp[::-1]
                if tw >= 2 * pad_px:
                    wx[:pad_px], wx[-pad_px:] = ramp, ramp[::-1]
            return (wy[:, None] * wx[None, :])[:, :, None]

        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                y_pad_s, y_pad_e = max(0, y - tile_pad), min(h, y + tile_size + tile_pad)
                x_pad_s, x_pad_e = max(0, x - tile_pad), min(w, x + tile_size + tile_pad)

                tile = bgr[y_pad_s:y_pad_e, x_pad_s:x_pad_e]
                tile_out = infer_tile(model, tile, device).astype(np.float32)
                th, tw = tile_out.shape[:2]

                w_tile = make_weight(th, tw)
                oy_s, oy_e = y_pad_s * scale, y_pad_e * scale
                ox_s, ox_e = x_pad_s * scale, x_pad_e * scale

                upscaled_bgr[oy_s:oy_e, ox_s:ox_e] += tile_out * w_tile
                weight_canvas[oy_s:oy_e, ox_s:ox_e] += w_tile

        upscaled_bgr = (upscaled_bgr / np.maximum(weight_canvas, 1e-7)).clip(0, 255).astype(np.uint8)

    # Edge-preserving clarity refinement (CLAHE on L-channel in LAB)
    lab = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    l_clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8)).apply(l)
    enhanced_bgr = cv2.cvtColor(cv2.merge([l_clahe, a, b_ch]), cv2.COLOR_LAB2BGR)

    # Recombine alpha channel if applicable
    if has_alpha and alpha is not None:
        alpha_up = cv2.resize(alpha, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
        final_img = cv2.merge([enhanced_bgr[:, :, 0], enhanced_bgr[:, :, 1], enhanced_bgr[:, :, 2], alpha_up])
    else:
        final_img = enhanced_bgr

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), final_img)
    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python enhance.py <input_image> [output_image]")
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    result = enhance_image(inp, out)
    print(f"Enhanced image saved to: {result}")


if __name__ == "__main__":
    main()
