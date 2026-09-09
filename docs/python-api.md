# Python API

Image Enhancer exposes a modular Python API that lets you integrate deep
generative super-resolution, face restoration, and post-processing filters
directly into your Python applications and pipelines.

This document describes the primary classes, configuration objects, and code
patterns available in the package.

## Quick start

The fastest way to enhance an image programmatically is through the
`ImageEnhancementPipeline` class.

```python
from pathlib import Path
from image_enchancer import ImageEnhancementPipeline, EnhancementConfig

# Create configuration
config = EnhancementConfig(
    model_name="realesrgan-x4plus",
    scale=4.0,
    face_restore=True,
    face_fidelity=0.85,
)

# Initialize pipeline
pipeline = ImageEnhancementPipeline(config)

# Process file
metrics = pipeline.process_file(
    input_path=Path("input.jpg"),
    output_path=Path("output.png"),
)

print(f"Upscaled to {metrics['output_dimensions']} in {metrics['elapsed_seconds']:.2f}s")
```

## EnhancementConfig

The `EnhancementConfig` dataclass defines all operational parameters for the
pipeline.

```python
@dataclass
class EnhancementConfig:
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
```

### Parameter reference

The following list outlines the purpose of each configuration field:

- `model_name`: The preset identifier (for example, `realesrgan-x4plus` or
  `realesrgan-anime`) or a string path to a custom model file.
- `scale`: Target scaling factor. When `None`, the pipeline uses the native
  scale of the loaded model (typically 4x).
- `face_restore`: When set to `True`, automatically runs YuNet face detection
  and `GFPGAN v1.4` restoration on any detected human faces.
- `face_fidelity`: Controls the blending ratio between the restored face and the
  original context. Valid values range from `0.0` (original face) to `1.0` (pure
  reconstruction).
- `tile_size`: Maximum dimension of image tiles in pixels. Set to `0` to disable
  tiling and run inference on the full tensor in a single pass.
- `tile_pad`: Context padding in pixels added around each tile before upscaling to
  prevent edge artifacts.
- `denoise_strength`: Bilateral filtering strength applied prior to
  super-resolution.
- `clarity_strength`: Luminance CLAHE clip limit. Set to `0.0` to disable.
- `sharpen_strength`: Edge-preserving unsharp mask weight.
- `color_match`: Enforces Reinhard color transfer in CIELAB space to ensure the
  output matches the original input color profile.
- `device`: Device string (such as `"cpu"` or `"cuda"`). When omitted, Image
  Enhancer automatically selects CUDA if an NVIDIA GPU is available, or CPU
  otherwise.

## ImageEnhancementPipeline

The `ImageEnhancementPipeline` class orchestrates model loading, execution,
tiling, and image file I/O.

### Methods

#### `process_file`

Processes an image file stored on disk and saves the enhanced image to a target
path.

```python
def process_file(
    self,
    input_path: Path,
    output_path: Path,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]
```

Parameters:
- `input_path`: Path object pointing to the source image.
- `output_path`: Path object pointing to the desired output location.
- `progress_callback`: Optional callable receiving `(completed_tiles, total_tiles)`.

Returns:
- A dictionary containing execution telemetry (elapsed time, original dimensions,
  output dimensions, effective scale, and restored face count).

#### `process_array`

Processes an in-memory NumPy image array (uint8, BGR or BGRA format).

```python
def process_array(
    self,
    image_input: np.ndarray,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]
```

Parameters:
- `image_input`: NumPy array with shape $(H, W, 3)$ or $(H, W, 4)$ in uint8 format.
- `progress_callback`: Optional callable for tracking tile progress.

Returns:
- A tuple of `(enhanced_image_array, telemetry_dict)`.

## Standalone script API

If you prefer a minimal function call without instantiating the multi-module
pipeline class, use the `enhance_image` function from `enhance.py`.

```python
from enhance import enhance_image

# Enhance image with 4x Real-ESRGAN and automatic tiling
result_path = enhance_image("portrait.jpg", "portrait_4x.png")
print(f"Saved to: {result_path}")
```

## Next steps

- Read [Models](models.md) to inspect supported model architectures and custom
  checkpoints.
- Read [Architecture](architecture.md) for details on the underlying tiling math.
