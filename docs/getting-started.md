# Getting started

Image Enhancer is a high-performance image restoration tool that performs
4x neural super-resolution, blind face restoration, and perceptual contrast
refinement while preserving original color fidelity.

This guide walks you through installing the environment, verifying dependencies,
and running your first image enhancement.

## Prerequisites

Before you install Image Enhancer, verify that your machine meets the following
system requirements.

- Operating system: Linux (x86_64), macOS, or Windows with WSL2.
- Python: Version 3.12 or later.
- Package manager: We recommend [`uv`](https://docs.astral.sh/uv/) for fast
  and reproducible dependency resolution.
- Hardware: Multi-core CPU (AVX2 supported) or an Intel Iris Xe / discrete GPU.

To check your Python version, run:

```bash
python3 --version
```

## Installation

Follow these steps to clone the project repository and set up the local virtual
environment.

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/ind4x/image-enchancer.git
   cd image-enchancer
   ```
2. Sync dependencies and create the virtual environment using `uv`:
   ```bash
   uv sync
   ```
   This command installs PyTorch (CPU-optimized build), Spandrel, OpenCV,
   Pillow, NumPy, Click, and Rich.

<!-- prettier-ignore -->
> [!NOTE]
> The project configuration specifies the official PyTorch CPU wheel index by
> default. This avoids downloading unnecessary multi-gigabyte CUDA libraries on
> machines running Intel integrated graphics.

## Running your first enhancement

You can run Image Enhancer using either the full-featured command-line interface
or the lightweight single-file script.

### Using the CLI

The command-line interface provides rich progress feedback and automatic model
management.

To enhance a photograph with default photographic weights:

```bash
uv run image-enchancer enhance samples/test_input.png -o output.png
```

When execution completes, Image Enhancer prints a summary table displaying the
original dimensions, upscaled output dimensions, detected faces, and elapsed
processing time.

### Using the standalone script

If you want a lightweight solution without subcommands or multi-stage options,
use `enhance.py`.

To run the standalone script:

```bash
uv run python enhance.py samples/test_input.png output.png
```

If you omit the output path, the script automatically saves the result next to
the input image with the `_enhanced` suffix.

### Using the direct global command

Image Enhancer installs global shell commands (`image-enchance` and `image-enhance`)
directly into your user path (`~/.local/bin/`).

This lets you enhance any local file or remote image URL from any terminal
directory:

```bash
# Enhance a local file
image-enchance photo.jpg

# Enhance a remote image URL directly
image-enchance https://example.com/sample.png
```

Where the enhanced version is saved:
- When you omit the destination path, the enhanced image is saved in your
  **current working directory** (`pwd`) with `_enhanced` added to the filename
  (for example, `./photo_enhanced.jpg`).
- When you provide a destination argument, it saves exactly to that path or folder:
  ```bash
  image-enchance https://example.com/sample.png ./output.png
  ```

## Verifying the installation

We recommend running the automated test suite to confirm that your local
environment functions properly.

To run the tests:

```bash
uv run pytest -v
```

All unit tests must pass before you process critical production images.

## Next steps

Now that you set up the project, explore the following documentation pages:

- Read [Architecture](architecture.md) to understand the tiling algorithm and
  restoration stages.
- Read [CLI reference](cli-reference.md) for a comprehensive list of commands
  and tuning flags.
- Read [Python API](python-api.md) to integrate the pipeline into your own
  applications.
- Read [Models](models.md) to switch between photographic, anime, and custom
  Transformer weights.
