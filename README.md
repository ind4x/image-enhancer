# Image Enhancer

Image Enhancer is a modular image restoration pipeline designed to perform
4x neural super-resolution, blind human face restoration, and perceptual contrast
refinement while preserving original color science.

The system combines hardware-accelerated Vulkan GPU execution (Intel Iris Xe),
deep convolutional generative priors (`Real-ESRGAN`), StyleGAN facial priors
(`GFPGAN v1.4`), and memory-bounded tiled inference to process images of any
resolution without memory exhaustion or boundary seams.

## Key features

Image Enhancer brings together deep generative architectures and signal
processing algorithms into a streamlined CLI tool.

- **Hardware GPU acceleration:** Executes neural inference using an integrated
  Intel Iris Xe Vulkan NCNN engine with FP16 shaders, cutting processing latency
  substantially on high-resolution canvases.
- **Deep super-resolution:** Synthesizes realistic high-frequency textures
  using `Real-ESRGAN` (RRDBNet) and anime-specialized architectures loaded via
  `spandrel`.
- **Memory-bounded seamless tiling:** Employs a 2D cosine blending window
  across overlapping tile boundaries to eliminate grid seams and keep memory
  consumption low.
- **Blind face restoration:** Detects human faces with OpenCV YuNet, regresses
  5 facial landmarks, and restores facial features using `GFPGAN v1.4`.
- **Direct remote URL enhancement:** Ingests web-hosted images directly from
  HTTP and HTTPS URLs with automatic temporary caching and cleanup.
- **Desktop notifications:** Sends desktop completion alerts via `notify-send`
  with embedded thumbnail previews of the enhanced output.
- **Real-time tile indicators:** Displays live percentage and active tile
  counters in the terminal to provide immediate visual feedback.
- **Color preservation:** Applies Reinhard statistical color transfer in
  CIELAB space to ensure the upscaled result matches the original white balance.
- **Perceptual post-processing:** Enhances local micro-contrast using
  luminance-isolated CLAHE and edge-preserving unsharp masking.
- **Transparency support:** Isolates and Lanczos-resamples alpha channels on
  4-channel RGBA inputs (such as transparent PNG files).

## Quick start

You can install Image Enhancer, set up dependencies, and run your first image
enhancement in a few commands.

### Installation

Image Enhancer requires Python 3.12 or later and the `uv` package manager.

```bash
git clone https://github.com/ind4x/image-enchancer.git
cd image-enchancer
uv sync
```

This registers the global `image-enchance` and `image-enhance` executables in
your local bin directory (`~/.local/bin/`).

### Usage examples

The following examples demonstrate how to enhance local images and remote URLs
from any working directory in your terminal.

#### 1. Enhance a local image

To upscale a photograph with default 4x super-resolution and face restoration:

```bash
image-enchance photo.jpg
```

Image Enhancer saves the output file in your current working directory as
`./photo_enhanced.jpg` and triggers a desktop notification when finished.

#### 2. Enhance a remote web image

You can pass a web link directly without manually downloading the file first:

```bash
image-enchance https://example.com/wallpaper.jpeg
```

#### 3. High-resolution wallpaper mode

For wallpapers, landscapes, and non-portrait images, disable face restoration
to accelerate processing:

```bash
image-enchance wallpaper.jpeg --no-face-restore
```

#### 4. Digital art and anime mode

To upscale line drawings, manga, or digital art with the specialized compact
architecture:

```bash
image-enchance drawing.png --anime
```

#### 5. Custom output destination

To save the enhanced file to a specific path or target folder:

```bash
image-enchance input.jpg ~/Desktop/retouched.png
```

## Documentation

Comprehensive guides and technical documentation are available in the
`docs/` directory:

- [Getting started](docs/getting-started.md): Installation prerequisites,
  virtual environments, and initial execution.
- [Architecture](docs/architecture.md): Deep dive into the 5-stage pipeline,
  convolutional priors, and cosine tiling math.
- [CLI reference](docs/cli-reference.md): Complete list of command-line
  arguments, defaults, and usage examples.
- [Python API](docs/python-api.md): Programmatic integration using
  `ImageEnhancementPipeline`, `EnhancementConfig`, and `enhance_image`.
- [Models](docs/models.md): Technical specifications of preconfigured models
  and instructions for loading custom weights.
- [Changelog](CHANGELOG.md): Historical record of all version releases,
  additions, fixes, and architectural revisions.

## Testing

Run the automated test suite to verify that your local environment functions
correctly:

```bash
uv run pytest -v
```

## License

This project is open-source software licensed under the MIT License.
