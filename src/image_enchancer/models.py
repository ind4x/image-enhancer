import os
import urllib.request
from pathlib import Path
from typing import Optional, Dict
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
import torch
from spandrel import ModelLoader, ImageModelDescriptor

# Default cache directory for weights
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "image-enchancer" / "models"

MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "realesrgan-x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
        "type": "sr",
        "scale": "4",
        "description": "General photograph enhancement and 4x super-resolution with realistic texture synthesis.",
    },
    "realesrgan-anime": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "type": "sr",
        "scale": "4",
        "description": "Optimized for digital art, anime, manga, and clean graphic illustrations.",
    },
    "realesrnet-x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
        "filename": "RealESRNet_x4plus.pth",
        "type": "sr",
        "scale": "4",
        "description": "PSNR-oriented super-resolution model without generative hallucination artifacts.",
    },
    "gfpgan": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "filename": "GFPGANv1.4.pth",
        "type": "face",
        "scale": "1",
        "description": "Blind face restoration prior (GFPGAN v1.4) for high-fidelity facial details.",
    },
    "yunet": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "filename": "face_detection_yunet_2023mar.onnx",
        "type": "detector",
        "scale": "1",
        "description": "High-efficiency ONNX face detector with 5-point facial landmark regression.",
    },
}


def get_cache_dir() -> Path:
    """Returns the path to the model cache directory, creating it if required."""
    cache_dir = Path(os.environ.get("IMAGE_ENCHANCER_CACHE_DIR", DEFAULT_CACHE_DIR))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_file(url: str, target_path: Path, description: str = "Downloading") -> Path:
    """
    Downloads a remote file with a live Rich progress bar display.
    """
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target_path.with_suffix(".tmp")

    with Progress(
        SpinnerColumn(),
        TextColumn(f"[bold blue]{description}[/]"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) ImageEnhancer/1.0"},
        )
        with urllib.request.urlopen(req) as response, open(temp_target, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            task_id = progress.add_task("download", total=total_size)
            chunk_size = 64 * 1024
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                progress.update(task_id, advance=len(chunk))

    temp_target.rename(target_path)
    return target_path


def resolve_model_path(model_identifier: str) -> Path:
    """
    Resolves a model name or user path to a local file.
    If the model is in MODEL_REGISTRY, it checks local directory, cache, or downloads it.
    """
    path_candidate = Path(model_identifier)
    if path_candidate.exists():
        return path_candidate

    local_models_dir = Path.cwd() / "models"
    if (local_models_dir / model_identifier).exists():
        return local_models_dir / model_identifier

    normalized_key = model_identifier.lower().replace("_", "-")
    if normalized_key in MODEL_REGISTRY:
        meta = MODEL_REGISTRY[normalized_key]
        filename = meta["filename"]

        # Check project-local models folder first
        local_candidate = local_models_dir / filename
        if local_candidate.exists() and local_candidate.stat().st_size > 0:
            return local_candidate

        # Check cache directory
        cache_candidate = get_cache_dir() / filename
        if cache_candidate.exists() and cache_candidate.stat().st_size > 0:
            return cache_candidate

        # Download to cache directory
        return download_file(
            url=meta["url"],
            target_path=cache_candidate,
            description=f"Fetching {meta['filename']}",
        )

    # Check if filename exists with extension
    for meta in MODEL_REGISTRY.values():
        if meta["filename"] == model_identifier or meta["filename"].startswith(model_identifier):
            local_candidate = local_models_dir / meta["filename"]
            if local_candidate.exists():
                return local_candidate
            cache_candidate = get_cache_dir() / meta["filename"]
            if cache_candidate.exists():
                return cache_candidate
            return download_file(
                url=meta["url"],
                target_path=cache_candidate,
                description=f"Fetching {meta['filename']}",
            )

    raise FileNotFoundError(
        f"Model '{model_identifier}' was not found locally and is not a recognized preconfigured model identifier. "
        f"Available presets: {', '.join(MODEL_REGISTRY.keys())}"
    )


def load_neural_model(
    model_identifier: str,
    device: Optional[torch.device] = None,
) -> ImageModelDescriptor:
    """
    Loads any super-resolution or restoration model via Spandrel.
    Automatically moves model to target device and sets eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file = resolve_model_path(model_identifier)
    loader = ModelLoader(device=device)
    descriptor = loader.load_from_file(str(model_file))
    
    if not isinstance(descriptor, ImageModelDescriptor):
        raise TypeError(
            f"Loaded model at {model_file} is of type {type(descriptor).__name__}, "
            f"expected ImageModelDescriptor."
        )

    descriptor.to(device)
    descriptor.eval()
    return descriptor
