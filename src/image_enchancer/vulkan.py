import os
import re
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Callable

VULKAN_DIR = Path.home() / ".local" / "share" / "image-enchancer" / "vulkan"
VULKAN_BINARY = VULKAN_DIR / "realesrgan-ncnn-vulkan"
VULKAN_MODELS = VULKAN_DIR / "models"
VULKAN_ZIP_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"

MODEL_MAP = {
    "realesrgan-x4plus": "realesrgan-x4plus",
    "realesrgan-anime": "realesrgan-x4plus-anime",
    "realesrnet-x4plus": "realesrnet-x4plus",
}


def has_vulkan_driver() -> bool:
    """Checks whether Vulkan ICD loader and drivers are installed on the host system."""
    icd_dir = Path("/usr/share/vulkan/icd.d")
    libvulkan = Path("/usr/lib64/libvulkan.so.1")
    return (icd_dir.exists() and any(icd_dir.glob("*.json"))) or libvulkan.exists()


def is_vulkan_available() -> bool:
    """Returns True if both the Vulkan binary and system drivers are present."""
    if not has_vulkan_driver():
        return False

    # Check shared location
    if VULKAN_BINARY.exists() and os.access(VULKAN_BINARY, os.X_OK):
        return True

    # Check project-local bin/
    local_bin = Path.cwd() / "bin" / "realesrgan-ncnn-vulkan"
    if local_bin.exists() and os.access(local_bin, os.X_OK):
        return True

    # Check system PATH
    return shutil.which("realesrgan-ncnn-vulkan") is not None


def ensure_vulkan_engine() -> Tuple[Path, Path]:
    """
    Ensures that the realesrgan-ncnn-vulkan binary and model directory exist,
    auto-downloading and extracting the release bundle if necessary.
    """
    # 1. Project-local directory
    local_bin = Path.cwd() / "bin" / "realesrgan-ncnn-vulkan"
    local_models = Path.cwd() / "bin" / "models"
    if local_bin.exists() and local_models.exists():
        return local_bin, local_models

    # 2. User data directory
    if VULKAN_BINARY.exists() and VULKAN_MODELS.exists():
        return VULKAN_BINARY, VULKAN_MODELS

    # 3. System PATH
    system_path = shutil.which("realesrgan-ncnn-vulkan")
    if system_path:
        bin_path = Path(system_path)
        candidate_models = bin_path.parent / "models"
        if candidate_models.exists():
            return bin_path, candidate_models

    # 4. Auto-download to ~/.local/share/image-enchancer/vulkan
    VULKAN_DIR.mkdir(parents=True, exist_ok=True)
    temp_zip = VULKAN_DIR / "vulkan_bundle.zip"

    try:
        req = urllib.request.Request(
            VULKAN_ZIP_URL,
            headers={"User-Agent": "Mozilla/5.0 ImageEnhancer/1.0"},
        )
        with urllib.request.urlopen(req) as resp, open(temp_zip, "wb") as f:
            f.write(resp.read())

        with zipfile.ZipFile(temp_zip, "r") as zip_ref:
            zip_ref.extractall(VULKAN_DIR)

        temp_zip.unlink(missing_ok=True)
        VULKAN_BINARY.chmod(0o755)

        return VULKAN_BINARY, VULKAN_MODELS
    except Exception as ex:
        temp_zip.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to auto-provision Vulkan NCNN engine: {ex}")


def run_vulkan_upscale(
    input_path: Path,
    output_path: Path,
    model_name: str = "realesrgan-x4plus",
    scale: int = 4,
    tile_size: int = 0,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> bool:
    """
    Executes hardware-accelerated super-resolution via the Vulkan NCNN engine.
    Parses live progress percentages and invokes progress_callback.
    """
    binary_path, models_dir = ensure_vulkan_engine()
    mapped_model = MODEL_MAP.get(model_name, "realesrgan-x4plus")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(binary_path),
        "-i", str(input_path),
        "-o", str(output_path),
        "-m", str(models_dir),
        "-n", mapped_model,
        "-s", str(int(scale)),
        "-t", str(tile_size),
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        pct_regex = re.compile(r"([0-9.]+)%")

        for line in iter(process.stdout.readline, ""):
            line = line.strip()
            match = pct_regex.search(line)
            if match:
                try:
                    pct = float(match.group(1))
                    if progress_callback:
                        progress_callback(int(pct), 100, f"GPU Vulkan upscaling {pct:.1f}%")
                except ValueError:
                    pass

        process.wait()
        return process.returncode == 0 and output_path.exists()
    except Exception as ex:
        print(f"[Warning] Vulkan upscaling encountered an error: {ex}")
        return False
