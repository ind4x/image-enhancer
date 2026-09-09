# CLI reference

Image Enhancer provides a command-line interface built with Click and Rich.
It lets you process single images or entire directories, control tiling parameters,
adjust face restoration fidelity, and manage cached neural weights.

This document provides a complete reference for all available commands, options,
and argument configurations.

## The enhance command

The `enhance` command executes the neural super-resolution and enhancement
pipeline on one or more images.

```bash
uv run image-enchancer enhance <input_path> [options]
```

`<input_path>` is a required argument that points to either a single image file
or a directory containing images. Supported image formats include JPEG, PNG,
WebP, BMP, and TIFF.

### Options table

The following table summarizes all command options, their accepted types,
default values, and descriptions.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `-o, --output` | Path | None | Output file path or target directory. |
| `-m, --model` | String | `realesrgan-x4plus` | Preset model key or file path to custom weights. |
| `-s, --scale` | Float | Model native (4.0) | Target upscale multiplier. Uses Lanczos4 if different. |
| `--face-restore / --no-face-restore` | Flag | `True` | Enables or disables GFPGAN blind face restoration. |
| `--face-fidelity` | Float | `0.85` | Balance between restored and original face (0.0 to 1.0). |
| `--tile-size` | Integer | `512` | Square tile size in pixels. Set to `0` to disable tiling. |
| `--tile-pad` | Integer | `32` | Overlap boundary padding in pixels for tile blending. |
| `--denoise` | Float | `0.0` | Pre-processing bilateral filter strength (0.0 to 1.0). |
| `--clarity` | Float | `1.2` | CIELAB CLAHE clip limit. Set to `0.0` to disable. |
| `--sharpen` | Float | `0.35` | Edge-preserving unsharp mask strength. Set to `0.0` to disable. |
| `--color-match / --no-color-match` | Flag | `True` | Matches color distribution with the original input. |
| `--engine` | String | `auto` | Processing engine (`auto`, `vulkan`, or `cpu`). |
| `--notify / --no-notify` | Flag | `True` | Sends desktop notification with thumbnail upon completion. |
| `--device` | String | Auto-detected | Target compute device (`cpu` or `cuda`). |

### Examples

The following examples demonstrate common usage patterns for different image
types and requirements.

#### Enhancing a high-resolution portrait

When you enhance a portrait, keep face restoration enabled and adjust the
fidelity to preserve natural facial characteristics:

```bash
uv run image-enchancer enhance portrait.jpg \
  -o portrait_enhanced.png \
  --model realesrgan-x4plus \
  --face-restore \
  --face-fidelity 0.80
```

#### Enhancing digital art and illustrations

For anime, line art, or digital paintings, use the specialized anime model and
disable face restoration to prevent human facial priors from interfering with
drawn characters:

```bash
uv run image-enchancer enhance sketch.png \
  -o sketch_4x.png \
  --model realesrgan-anime \
  --no-face-restore
```

#### Downscaling to 2x output resolution

To achieve a 2x upscale with greater sharpness than direct 2x interpolation,
Image Enhancer runs the 4x super-resolution network and downsamples the result
with high-quality Lanczos resampling:

```bash
uv run image-enchancer enhance image.jpg -o image_2x.jpg --scale 2.0
```

#### Batch processing an entire folder

When the input path is a directory, Image Enhancer automatically processes all
compatible images and saves the results in the destination folder:

```bash
uv run image-enchancer enhance ./input_photos/ -o ./enhanced_photos/
```

## The models commands

The `models` command group lets you inspect and pre-download preconfigured neural
weights.

### Listing registered models

To view the status of all preconfigured models and check whether they are
cached on your local machine, run:

```bash
uv run image-enchancer models list
```

This displays a table containing the preset key, model type, native scale, local
availability, and architecture description.

### Pre-downloading model weights

You can pre-download weights so that Image Enhancer runs without requiring an
active internet connection during processing.

To download all preconfigured models:

```bash
uv run image-enchancer models download all
```

To download a specific model preset (for example, `realesrgan-anime`):

```bash
uv run image-enchancer models download realesrgan-anime
```

Downloaded weights reside in `~/.cache/image-enchancer/models/` or the local
`./models/` directory.

## The image-enchance direct command

The `image-enchance` (and `image-enhance`) command provides a streamlined,
single-command entry point that operates on local paths or remote URLs.

```bash
image-enchance <target> [output_path] [options]
```

`<target>` can be a local image path, a directory, or an HTTP/HTTPS image URL.

### Options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `-m, --model` | String | `realesrgan-x4plus` | Preset model key or path to weights. |
| `-s, --scale` | Float | Model native (4.0) | Target upscale multiplier. |
| `--face-restore / --no-face-restore` | Flag | `True` | Enables or disables GFPGAN face restoration. |
| `--anime` | Flag | `False` | Shortcut for anime mode (`realesrgan-anime` with no face restoration). |
| `--engine` | String | `auto` | Processing engine (`auto`, `vulkan`, or `cpu`). |
| `--notify / --no-notify` | Flag | `True` | Sends desktop notification with thumbnail upon completion. |

### Output destination

When you run `image-enchance`:
- **Without an output argument:** The enhanced file is saved directly in the
  current working directory from which you invoked the command (`pwd`), using
  the pattern `<stem>_enhanced.<ext>`. The terminal prints the absolute path
  (`Saved: /path/to/...`) upon completion.
- **With an explicit destination:** If you provide a second argument, the output
  is written to that exact file path or directory.

## Next steps

- Read [Python API](python-api.md) to integrate these options programmatically.
- Read [Models](models.md) for technical specifications of each neural network.
