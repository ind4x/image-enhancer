# Changelog

All notable changes to the Image Enhancer project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-09

### Added

- Added direct global shell commands (`image-enchance` and `image-enhance`)
  installed in `~/.local/bin/` for single-command terminal execution from any
  working directory.
- Added native HTTP and HTTPS remote image URL resolution, allowing direct
  enhancement of web-hosted images with automatic temporary cache cleanup.
- Added hardware-accelerated Vulkan GPU execution engine (`src/image_enchancer/vulkan.py`)
  targeting Intel Iris Xe integrated graphics using FP16 compute shaders.
- Added `--engine` option (`auto`, `vulkan`, `cpu`) across CLI and direct commands
  with automatic GPU detection and fallback to PyTorch CPU tiling.
- Added desktop notification integration using `/usr/bin/notify-send` with live
  preview image thumbnails and execution metrics on completion.
- Added real-time active tile indicators and percentage displays to the terminal
  progress bar to eliminate perceived execution freezes on large canvases.

### Changed

- Enhanced pipeline architecture to use hybrid acceleration: Intel Iris Xe
  Vulkan engine for high-resolution tile upscaling combined with PyTorch CPU
  GFPGAN face restoration.
- Refactored `super_resolution.py` to broadcast progress state before and after
  tile processing.
- Configured default destination for single-image CLI commands to automatically
  save into the source image's parent directory rather than the terminal working
  directory.
- Updated documentation across `docs/` and `README.md` to reflect direct command
  usage, URL fetching, and GPU acceleration.

### Removed

- Removed Gradio web interface and associated web dependencies (`gradio`,
  `fastapi`, `uvicorn`, `pandas`, `pydantic`) to maintain a lean, lightweight
  package footprint.

## [0.1.0] - 2026-09-09

### Added

- Initial release of the Image Enhancer pipeline.
- Implemented deep generative super-resolution with `Real-ESRGAN` (RRDBNet) and
  `RealESRGAN_x4plus_anime_6B` architectures via `spandrel`.
- Implemented memory-bounded spatial tiling engine using a 2D cosine-tapered
  Hann blending window for seam-free high-resolution upscaling.
- Implemented blind face restoration using OpenCV DNN YuNet 5-point landmark
  detection and `GFPGAN v1.4` affine similarity warping.
- Implemented radiometric and chromatic post-processing: CIELAB Lightness-channel
  CLAHE, edge-preserving deadband unsharp masking, and Reinhard color matching.
- Implemented 4-channel alpha transparency isolation and Lanczos4 resampling.
- Configured PyTorch CPU wheel index in `pyproject.toml` to prevent downloading
  unnecessary CUDA packages on non-NVIDIA systems.
- Created test suite with unit tests in `tests/test_pipeline.py`.
