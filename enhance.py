#!/usr/bin/env python3
"""
Self-contained, zero-boilerplate neural image enhancer.
Upscales and restores images to 4x resolution using Real-ESRGAN with automatic tiling.
Supports both local image files and remote image URLs.
"""

import sys
import tempfile
import urllib.parse
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
    input_target: str | Path,
    output_path: str | Path | None = None,
    tile_size: int = 512,
    tile_pad: int = 32,
) -> Path:
    """
    Enhances a local image or remote URL to 4x resolution with seamless tiling.
    """
    target_str = str(input_target)
    is_temp = False

    # Handle remote URLs
    if target_str.startswith("http://") or target_str.startswith("https://"):
        parsed = urllib.parse.urlparse(target_str)
        fname = Path(parsed.path).name or "remote_image.png"
        if not Path(fname).suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]:
            fname += ".png"

        temp_dir = Path(tempfile.gettempdir()) / "image_enchancer_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        local_input = temp_dir / fname
        print(f"Downloading remote image from: {target_str}...")
        req = urllib.request.Request(target_str, headers={"User-Agent": "Mozilla/5.0 ImageEnhancer/1.0"})
        with urllib.request.urlopen(req) as resp, open(local_input, "wb") as f:
            f.write(resp.read())
        is_temp = True
        display_stem = Path(fname).stem
        display_suffix = Path(fname).suffix
    else:
        local_input = Path(input_target)
        if not local_input.exists():
            raise FileNotFoundError(f"Input file not found: {input_target}")
        display_stem = local_input.stem
        display_suffix = local_input.suffix

    if output_path is None:
        target_dir = local_input.parent.resolve() if not is_temp else Path.cwd()
        output_path = target_dir / f"{display_stem}_enhanced{display_suffix}"
    else:
        output_path = Path(output_path)

    # Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_file = ensure_model()
    loader = ModelLoader(device=device)
    model = loader.load_from_file(str(model_file))
    model.eval()

    # Read image (including alpha if present)
    img = cv2.imread(str(local_input), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not decode image: {local_input}")

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

    if is_temp and local_input.exists():
        try:
            local_input.unlink()
        except Exception:
            pass

    # Send desktop notification with thumbnail
    try:
        import subprocess
        subprocess.run(
            ["notify-send", "Image Enhancement Complete", f"{output_path.name} saved ({w*scale}x{h*scale})", "-i", str(output_path), "--app-name=Image Enhancer"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        pass

    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python enhance.py <input_image_or_url> [output_image]")
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    result = enhance_image(inp, out)
    print(f"Enhanced image saved to: {result}")


if __name__ == "__main__":
    main()
