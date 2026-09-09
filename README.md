# Image Enhancer

Image Enhancer is a modular image restoration pipeline designed to perform
4x neural super-resolution, blind human face restoration, and perceptual contrast
refinement while preserving color science.

The system combines deep convolutional generative priors (`Real-ESRGAN`), StyleGAN
facial priors (`GFPGAN v1.4`), and memory-bounded tiled inference to process
images of any dimension without memory exhaustion or boundary seams.

## Key features

- **Deep super-resolution:** Synthesizes realistic high-frequency textures
  using `Real-ESRGAN` (RRDBNet) and anime-specialized architectures loaded via
  `spandrel`.
- **Memory-bounded seamless tiling:** Employs a 2D cosine blending window
  across overlapping tile boundaries to eliminate grid seams and keep memory
  consumption low.
- **Blind face restoration:** Detects faces with OpenCV YuNet, regresses 5
  facial landmarks, and restores facial details using `GFPGAN v1.4`.
- **Color preservation:** Applies Reinhard statistical color transfer in
  CIELAB space to ensure the upscaled result matches the original white balance.
- **Perceptual post-processing:** Enhances local micro-contrast using
  luminance-isolated CLAHE and edge-preserving unsharp masking.
- **Transparency support:** Isolates and Lanczos-resamples alpha channels on
  4-channel RGBA inputs (such as transparent PNG files).
- **Zero-boilerplate standalone script:** Includes `enhance.py` for direct
  single-command execution.

## Quick start

Follow these steps to set up the project and run your first enhancement.

### 1. Installation

Ensure Python 3.12+ and `uv` are installed on your system.

```bash
git clone https://github.com/ind4x/image-enchancer.git
cd image-enchancer
uv sync
```

### 2. Run the enhancement CLI

To enhance a photograph with default photographic weights:

```bash
uv run image-enchancer enhance input.jpg -o enhanced.png
```

To enhance digital art, anime, or line drawings:

```bash
uv run image-enchancer enhance drawing.png -o drawing_4x.png --model realesrgan-anime --no-face-restore
```

### 3. Run the standalone script

If you want a lightweight solution without subcommands:

```bash
uv run python enhance.py input.jpg output.png
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

## Testing

Run the automated test suite to verify that your local environment functions
correctly:

```bash
uv run pytest -v
```

## License

This project is open-source software licensed under the MIT License.
