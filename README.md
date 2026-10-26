# Image Enhancer

Image Enhancer is a modular deep learning image restoration pipeline designed to perform
4x neural super-resolution, blind human face restoration, and perceptual contrast
refinement while preserving authentic color science.

The system combines hardware-accelerated Vulkan GPU execution (Intel Iris Xe),
deep convolutional generative priors (`Real-ESRGAN`), StyleGAN facial priors
(`GFPGAN v1.4`), and memory-bounded tiled inference to process images of arbitrary
resolutions without memory exhaustion or boundary seams.

---

## Pipeline Architecture

```text
+-----------------------+
|  Input Image / URL    |  (RGB / RGBA / Remote HTTP Resource)
+-----------------------+
            |
            v
+-----------------------+
|  Color & Alpha Cache  |  - Extract & cache CIELAB color statistics (Reinhard)
|                       |  - Separate & Lanczos-resample alpha channel (RGBA)
+-----------------------+
            |
            v
+-----------------------+
|  2D Cosine Tiling     |  - Split canvas into overlapping tiles with boundary pads
|  (Memory-Bounded)     |  - Smooth blending window: w(x,y) = cos^2(pi*x/2W) * cos^2(pi*y/2H)
+-----------------------+
            |
            v
+-----------------------+
|  Vulkan GPU Engine    |  - Real-ESRGAN (RRDBNet / Anime architecture)
|  (Intel Iris Xe FP16) |  - 4x High-frequency feature synthesis
+-----------------------+
            |
            v
+-----------------------+
| Blind Face Restorer   |  - Face detection & 5-point landmark alignment (YuNet)
| (GFPGAN v1.4)         |  - StyleGAN generative facial feature synthesis
+-----------------------+
            |
            v
+-----------------------+
| Color Restoration     |  - Reinhard statistical color transfer in CIELAB space
| & Contrast Tuning     |  - Luminance-isolated CLAHE + unsharp masking
+-----------------------+
            |
            v
+-----------------------+
| Final Enhanced Output |  - Saved to disk with desktop thumbnail alert (notify-send)
+-----------------------+
```

---

## Key Features

- **Hardware GPU Acceleration:** Executes neural inference using an integrated Intel Iris Xe Vulkan NCNN engine with FP16 shaders, cutting processing latency substantially on high-resolution canvases.
- **Deep Super-Resolution:** Synthesizes realistic high-frequency textures using `Real-ESRGAN` (RRDBNet) and anime-specialized architectures loaded via `spandrel`.
- **Memory-Bounded Seamless Tiling:** Employs a 2D cosine blending window across overlapping tile boundaries to eliminate grid seams and prevent VRAM exhaustion.
- **Blind Face Restoration:** Detects human faces with OpenCV YuNet, regresses 5 facial landmarks, and restores degraded facial features using `GFPGAN v1.4`.
- **Direct Remote URL Enhancement:** Ingests web-hosted images directly from HTTP and HTTPS URLs with automatic temporary caching and cleanup.
- **Desktop Notifications:** Sends desktop completion alerts via `notify-send` with embedded thumbnail previews of the enhanced output.
- **Real-Time Tile Indicators:** Displays live percentage and active tile counters in the terminal for real-time progress tracking.
- **Color Preservation:** Applies Reinhard statistical color transfer in CIELAB space to ensure the upscaled result matches the original white balance.
- **Perceptual Post-Processing:** Enhances local micro-contrast using luminance-isolated CLAHE and edge-preserving unsharp masking.

---

## Quick Start

### Installation

Image Enhancer requires Python 3.12+ and the `uv` package manager:

```bash
git clone git@github.com:ind4x/image-enhancer.git
cd image-enhancer
uv sync
```

This registers the global `image-enhance` executable in your local bin directory (`~/.local/bin/`).

### Usage Examples

#### 1. Enhance a local image
```bash
image-enhance photo.jpg
```
Saves the output as `<stem>_enhanced.<ext>` and sends a desktop notification with thumbnail preview.

#### 2. Enhance a remote web image
```bash
image-enhance https://example.com/wallpaper.jpeg
```

#### 3. High-resolution wallpaper mode (Skip face restoration)
```bash
image-enhance wallpaper.jpeg --no-face-restore
```

#### 4. Digital art and anime mode
```bash
image-enhance drawing.png --anime
```

#### 5. Custom output destination
```bash
image-enhance input.jpg ~/Desktop/retouched.png
```

---

## Testing

Run the automated test suite:

```bash
uv run pytest -v
```

---

## Documentation

Comprehensive guides are available in the `docs/` directory:
- [Getting started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [CLI reference](docs/cli-reference.md)
- [Python API](docs/python-api.md)
- [Models](docs/models.md)
- [Changelog](CHANGELOG.md)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

- **Author:** Youssef Fellah ([ind4x_](https://github.com/ind4x))
