# Models

Image Enhancer uses deep convolutional networks and facial generative priors to
reconstruct damaged image details. By integrating the `spandrel` universal model
loader, Image Enhancer lets you select from built-in preconfigured models or
load custom community architectures.

This document describes each preconfigured model preset and explains how to load
custom model checkpoints.

## Preconfigured models

Image Enhancer includes five preconfigured models that cover photographic
upscaling, digital art restoration, blind face repair, and face detection.

| Preset key | Architecture | Scale | File size | Recommended use case |
| :--- | :--- | :--- | :--- | :--- |
| `realesrgan-x4plus` | RRDBNet | 4x | 67 MB | General photography, natural textures, and compressed images. |
| `realesrgan-anime` | Compact (6B) | 4x | 18 MB | Anime, manga, line art, cartoons, and digital graphics. |
| `realesrnet-x4plus` | RRDBNet | 4x | 67 MB | Photography where generative hallucination must be minimized (PSNR-oriented). |
| `gfpgan` | GFPGAN v1.4 | 1x | 348 MB | Blind restoration of human eyes, teeth, and facial skin texture. |
| `yunet` | YuNet ONNX | 1x | 336 KB | High-speed real-time face and 5-point facial landmark detection. |

### Model specifications

The following sections provide technical details for each model architecture.

#### RealESRGAN_x4plus

`RealESRGAN_x4plus` is the primary super-resolution model for real-world
photographs. It uses a 23-layer Residual-in-Residual Dense Block network
(RRDBNet) with 64 feature channels.

The model is trained using a pure synthetic second-order degradation pipeline:
$$\mathbf{y} = \left[\left(\mathbf{x} \otimes \mathbf{k}_1\right) \downarrow_r + \mathbf{n}_1\right]_{\text{JPEG}_{q_1}} \otimes \mathbf{k}_2 \dots$$

This training procedure simulates repeated motion blurs ($\mathbf{k}$), sensor
noises ($\mathbf{n}$), downsampling steps ($\downarrow$), and JPEG block
quantization artifacts ($q$). As a result, the network restores sharp edges and
reconstructs plausible fine textures on degraded inputs.

#### RealESRGAN_x4plus_anime_6B

`RealESRGAN_x4plus_anime_6B` uses a lightweight 6-block architecture. It is
specifically tuned for digital illustrations and animated content.

Unlike photographic models that attempt to hallucinate realistic grain and micro-textures,
the anime model focuses on sharpening crisp line art, preserving flat color fills,
and eliminating ringing artifacts around high-contrast contours.

#### GFPGAN v1.4

`GFPGAN v1.4` is a blind face restoration model based on a generative facial
prior encapsulated within a StyleGAN2 architecture.

During inference, it takes aligned $512 \times 512$ facial crops and uses spatial
feature transform (SFT) layers to project generative facial priors into the
degraded features. This allows the model to reconstruct realistic facial
features even when the input face is heavily pixelated or blurred.

## Using custom models

Image Enhancer supports custom third-party models compatible with `spandrel`.
You can pass any supported PyTorch (`.pth`, `.pt`), SafeTensors (`.safetensors`),
or binary (`.bin`) file directly to the `--model` argument.

Supported custom model families include:

- **HAT (Hybrid Attention Transformer):** Combines channel attention and
  self-attention for state-of-the-art detail preservation.
- **DAT (Dual Aggregation Transformer):** Aggregates features across spatial and
  channel dimensions.
- **SwinIR / Swin2SR:** Shifted-window attention networks for image restoration.
- **OmniSR:** Omnidirectional self-attention networks.
- **SCUNet:** Swin-Conv unet for image denoising.

To run Image Enhancer with a custom model file:

```bash
uv run image-enchancer enhance input.jpg \
  -o output.png \
  --model /path/to/custom_model.safetensors
```

<!-- prettier-ignore -->
> [!NOTE]
> When you provide a custom model file, Image Enhancer automatically inspects
> the state dictionary to determine its architecture, input channels, and native
> scale.

## Model storage and cache location

Image Enhancer checks for model checkpoints in two locations:

1. The local project directory: `./models/`
2. The global user cache directory: `~/.cache/image-enchancer/models/`

You can override the default cache location by setting the
`IMAGE_ENCHANCER_CACHE_DIR` environment variable:

```bash
export IMAGE_ENCHANCER_CACHE_DIR="/path/to/shared/models"
```

## Next steps

- Read [Getting started](getting-started.md) to set up dependencies.
- Read [CLI reference](cli-reference.md) for full argument options.
