#!/usr/bin/env python3
"""
Modern, responsive web interface for Image Enhancer.
Provides real-time completion tracking, side-by-side comparison, and one-click downloading.
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple
import cv2
import gradio as gr
import numpy as np
import torch

from image_enchancer.models import load_neural_model, resolve_model_path
from image_enchancer.super_resolution import SuperResolutionEngine
from image_enchancer.face_restore import FaceRestorationEngine
from image_enchancer.postprocess import PostProcessor

# In-memory model cache to prevent reloading weights on repeated enhancements
_LOADED_SR_MODELS = {}
_LOADED_FACE_ENGINE = None

CUSTOM_CSS = """
/* Core layout & typography polish */
:root {
    --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

body, .gradio-container {
    font-family: var(--font-sans) !important;
    -webkit-font-smoothing: antialiased;
}

.tabular-numbers {
    font-variant-numeric: tabular-nums;
}

/* Concentric border radius & depth */
.main-card {
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    background: rgba(255, 255, 255, 0.02) !important;
    box-shadow: 0 4px 24px -1px rgba(0, 0, 0, 0.3) !important;
    padding: 16px !important;
}

/* Subtle image outlines */
.image-preview img {
    border-radius: 10px !important;
    box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.1) !important;
}

/* Tactile scale-on-press button feedback */
.enhance-btn {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: transform 0.15s cubic-bezier(0.2, 0, 0, 1), background 0.15s ease !important;
}

.enhance-btn:active {
    transform: scale(0.96) !important;
}

/* Metrics badge */
.metrics-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.1);
    font-size: 0.85rem;
    font-variant-numeric: tabular-nums;
}
"""


def get_cached_sr_engine(model_key: str, device: torch.device) -> SuperResolutionEngine:
    """Retrieves or loads the neural super-resolution model in memory."""
    global _LOADED_SR_MODELS
    if model_key not in _LOADED_SR_MODELS:
        descriptor = load_neural_model(model_key, device=device)
        _LOADED_SR_MODELS[model_key] = SuperResolutionEngine(descriptor, device=device)
    return _LOADED_SR_MODELS[model_key]


def get_cached_face_engine(device: torch.device) -> FaceRestorationEngine:
    """Retrieves or loads the GFPGAN face restoration model in memory."""
    global _LOADED_FACE_ENGINE
    if _LOADED_FACE_ENGINE is None:
        _LOADED_FACE_ENGINE = FaceRestorationEngine(device=device)
    return _LOADED_FACE_ENGINE


def run_enhancement(
    input_image: Optional[np.ndarray],
    model_name: str,
    target_scale: str,
    enable_face_restore: bool,
    face_fidelity: float,
    clarity: float,
    sharpen: float,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Executes image enhancement with step-by-step progress tracking for Gradio.
    """
    if input_image is None:
        return None, None, "<div class='metrics-pill'>No image uploaded.</div>"

    start_time = time.perf_counter()
    progress(0.05, desc="Initializing environment...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scale_factor = float(target_scale.replace("x", ""))

    # Convert RGB (Gradio format) to BGR for OpenCV processing
    if len(input_image.shape) == 3 and input_image.shape[2] == 4:
        # RGBA
        bgr = cv2.cvtColor(input_image[:, :, :3], cv2.COLOR_RGB2BGR)
        alpha = input_image[:, :, 3]
        has_alpha = True
    else:
        bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        alpha = None
        has_alpha = False

    h_orig, w_orig = bgr.shape[:2]

    # Step 1: Model Loading
    progress(0.12, desc=f"Loading {model_name} model...")
    sr_engine = get_cached_sr_engine(model_name, device=device)

    # Step 2: Super-Resolution with live tiling progress
    def tile_progress_hook(completed: int, total: int):
        pct = 0.15 + 0.60 * (completed / max(total, 1))
        progress(pct, desc=f"Upscaling tile {completed}/{total}...")

    upscaled_bgr = sr_engine.upscale(
        image_bgr=bgr,
        tile_size=512,
        tile_pad=32,
        target_scale=scale_factor,
        progress_callback=tile_progress_hook,
    )

    h_up, w_up = upscaled_bgr.shape[:2]

    # Step 3: Face Restoration
    faces_restored = 0
    if enable_face_restore:
        progress(0.78, desc="Detecting and restoring facial landmarks...")
        try:
            face_engine = get_cached_face_engine(device=device)
            upscaled_bgr, faces_restored = face_engine.restore_faces(
                upscaled_bgr,
                fidelity=face_fidelity,
            )
        except Exception as ex:
            print(f"[Warning] Face restoration skipped: {ex}")

    # Step 4: Post-Processing & Micro-Contrast
    progress(0.88, desc="Calibrating color distribution & micro-contrast...")
    post = PostProcessor()
    upscaled_bgr = post.match_color_distribution(source_bgr=upscaled_bgr, reference_bgr=bgr)

    if clarity > 0.0:
        upscaled_bgr = post.apply_clahe(upscaled_bgr, clip_limit=clarity)

    if sharpen > 0.0:
        upscaled_bgr = post.apply_unsharp_mask(upscaled_bgr, strength=sharpen)

    # Step 5: Recompose alpha if present
    if has_alpha and alpha is not None:
        alpha_up = cv2.resize(alpha, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
        final_bgr = cv2.merge([
            upscaled_bgr[:, :, 0],
            upscaled_bgr[:, :, 1],
            upscaled_bgr[:, :, 2],
            alpha_up,
        ])
        ext = ".png"
    else:
        final_bgr = upscaled_bgr
        ext = ".png"

    progress(0.96, desc="Encoding output image...")

    # Save to high-quality temporary file
    temp_dir = Path(tempfile.gettempdir()) / "image_enchancer_ui"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / f"enhanced_{int(time.time())}{ext}"
    cv2.imwrite(str(temp_file), final_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 4])

    elapsed = time.perf_counter() - start_time
    progress(1.0, desc="Complete!")

    telemetry_html = f"""
    <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px;">
        <span class="metrics-pill">Original: <strong>{w_orig}x{h_orig}</strong></span>
        <span class="metrics-pill">Enhanced: <strong>{w_up}x{h_up}</strong> ({scale_factor}x)</span>
        <span class="metrics-pill">Faces Restored: <strong>{faces_restored}</strong></span>
        <span class="metrics-pill">Time: <strong>{elapsed:.2f}s</strong></span>
    </div>
    """

    return str(temp_file), str(temp_file), telemetry_html


def build_app() -> gr.Blocks:
    """Constructs the Gradio application UI layout."""
    with gr.Blocks(title="Image Enhancer") as demo:
        gr.Markdown(
            """
            # Image Enhancer
            High-fidelity neural super-resolution and face restoration pipeline.
            """
        )

        with gr.Row(equal_height=True):
            # Left column: Input & Controls
            with gr.Column(scale=5, elem_classes=["main-card"]):
                gr.Markdown("### 1. Upload Source Image")
                input_image = gr.Image(
                    label="Drop or select image",
                    type="numpy",
                    sources=["upload", "clipboard"],
                    elem_classes=["image-preview"],
                )

                with gr.Accordion("Enhancement Settings", open=True):
                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            label="Model Architecture",
                            choices=[
                                ("Real-ESRGAN x4+ (Photographs)", "realesrgan-x4plus"),
                                ("Real-ESRGAN Anime (Digital Art)", "realesrgan-anime"),
                                ("Real-ESRNet (PSNR Clean)", "realesrnet-x4plus"),
                            ],
                            value="realesrgan-x4plus",
                        )
                        scale_radio = gr.Radio(
                            label="Output Scale",
                            choices=["4x", "2x"],
                            value="4x",
                        )

                    with gr.Row():
                        face_restore_cb = gr.Checkbox(
                            label="Enable GFPGAN Face Restoration",
                            value=True,
                        )
                        face_fidelity_slider = gr.Slider(
                            label="Face Fidelity",
                            minimum=0.1,
                            maximum=1.0,
                            value=0.85,
                            step=0.05,
                        )

                    with gr.Row():
                        clarity_slider = gr.Slider(
                            label="Micro-Contrast (CLAHE)",
                            minimum=0.0,
                            maximum=2.5,
                            value=1.2,
                            step=0.1,
                        )
                        sharpen_slider = gr.Slider(
                            label="Edge Sharpening",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.35,
                            step=0.05,
                        )

                enhance_btn = gr.Button(
                    "Enhance Image",
                    variant="primary",
                    size="lg",
                    elem_classes=["enhance-btn"],
                )

            # Right column: Output & Downloads
            with gr.Column(scale=5, elem_classes=["main-card"]):
                gr.Markdown("### 2. Enhanced Result")
                output_image = gr.Image(
                    label="Enhanced Output",
                    type="filepath",
                    interactive=False,
                    elem_classes=["image-preview"],
                )
                
                download_file = gr.File(
                    label="Download Full Resolution File",
                    visible=True,
                )

                telemetry_badge = gr.HTML(
                    value="<div class='metrics-pill'>Ready to process</div>",
                )

        # Wire click event with progress bar
        enhance_btn.click(
            fn=run_enhancement,
            inputs=[
                input_image,
                model_dropdown,
                scale_radio,
                face_restore_cb,
                face_fidelity_slider,
                clarity_slider,
                sharpen_slider,
            ],
            outputs=[
                output_image,
                download_file,
                telemetry_badge,
            ],
        )

    return demo


def main():
    port = int(os.environ.get("PORT", 7860))
    app = build_app()
    print(f"Launching Image Enhancer Web UI on http://127.0.0.1:{port}")
    app.launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=False,
        theme=gr.themes.Base(),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
