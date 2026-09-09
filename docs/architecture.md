# Architecture

Image Enhancer coordinates multiple specialized computer vision models and
classical signal processing algorithms into a cohesive pipeline. This design
lets you restore degraded inputs, synthesize natural fine textures, and repair
facial features without introducing visual seams or color shifts.

This document describes the pipeline stages, the mathematical principles behind
seamless tiling, and how each component operates.

## Pipeline overview

The image enhancement process follows five sequential stages to balance detail
synthesis with artifact suppression.

1. **Pre-processing and validation:** Loads the input image, validates dimensions,
   and separates the alpha channel if transparency is present. An optional
   bilateral filter removes high-ISO sensor noise.
2. **Neural super-resolution:** Passes image tiles through a deep convolutional
   network (`Real-ESRGAN`) to synthesize realistic high-frequency priors at 4x
   scale.
3. **Face restoration:** Detects human faces with OpenCV YuNet, regresses 5
   facial landmarks, aligns the faces to a 512x512 template, restores them with
   `GFPGAN v1.4`, and blends them back into the upscaled canvas.
4. **Color preservation:** Applies Reinhard statistical color matching in CIELAB
   space to keep the exact white balance and color distribution of the original.
5. **Perceptual post-processing:** Executes localized micro-contrast enhancement
   using CIELAB CLAHE and edge-preserving unsharp masking.

```
Input Image ───► [Pre-filter] ───► [Tiled Super-Resolution]
                                             │
                                             ▼
Output Image ◄─── [Post-Process] ◄─── [Face Restoration]
```

## Neural super-resolution stage

Traditional upscaling methods like Bicubic and Lanczos interpolation calculate
weighted averages of existing pixels. While these methods are computationally
inexpensive, they cannot recover information lost during compression or downsampling.

Image Enhancer uses deep generative models trained with second-order degradation
modeling. The network learns to synthesize realistic textures (such as skin pores,
fabric weave, and fine typography) based on generative priors.

Through the `spandrel` universal architecture loader, Image Enhancer supports
multiple model architectures:

- **ESRGAN / Real-ESRGAN:** Residual-in-Residual Dense Blocks (RRDB) for
  general photography.
- **RealESRGAN-Anime:** Compact 6-block architecture optimized for digital art,
  cartoons, and line art.
- **Transformers (HAT, DAT, SwinIR):** High-capacity attention models for
  complex structural reconstruction.

## Memory-bounded seamless tiling

Deep neural networks require substantial intermediate activation memory during
inference. When upscaling a high-resolution image (such as 1080p to 4K), running
inference on the entire tensor at once can exhaust system memory.

To solve this issue, Image Enhancer implements a tiled inference engine with
contextual boundary padding and cosine-tapered blending.

### The boundary seam problem

If you divide an image into independent tiles and upscale each tile separately,
the reconstructed image shows visible grid seams along the borders. This happens
because the convolutional filters at the edges of a tile lack the surrounding
contextual pixels.

### The cosine feathering solution

Image Enhancer solves boundary seams by extracting each tile with an overlap
margin (`tile_pad`, typically 32 pixels). After upscaling each tile, the engine
applies a two-dimensional cosine blending window across the overlapping margin.

The one-dimensional cosine ramp function is defined as:

$$w(x) = \frac{1}{2} \left(1 - \cos\left(\frac{\pi x}{P}\right)\right)$$

Where $P$ represents the padding width scaled to the output resolution.

The engine multiplies the horizontal and vertical ramps to create a smooth 2D
feather mask:

$$W(x, y) = w_x(x) \times w_y(y)$$

The engine accumulates each weighted tile into the canvas and normalizes the
result by dividing by the accumulated weights:

$$I_{\text{final}}(x, y) = \frac{\sum_i W_i(x, y) \cdot T_i(x, y)}{\sum_i W_i(x, y) + \epsilon}$$

This mathematical formulation guarantees continuous blending transitions with
zero boundary seams and keeps memory consumption constant regardless of image
dimensions.

## Face restoration stage

Generic super-resolution models frequently introduce unnatural distortions on
human faces. Human perception is acutely sensitive to subtle facial asymmetries,
often resulting in an uncanny valley effect.

Image Enhancer includes a dedicated face restoration module that operates in
four steps:

1. **Detection:** OpenCV's DNN YuNet detector locates faces and regresses 5
   facial landmarks (left eye, right eye, nose tip, left mouth corner, and right
   mouth corner).
2. **Alignment:** An affine similarity transform calculates the optimal
   rotation, scale, and translation to map the detected landmarks to a canonical
   $512 \times 512$ facial template.
3. **Restoration:** The aligned face passes through `GFPGAN v1.4`, which uses
   facial component dictionaries and generative priors to reconstruct natural
   eyes, teeth, and skin micro-textures.
4. **Inverse warping and blending:** The engine computes the inverse affine
   transformation matrix and projects the restored face back to its original
   position. An elliptical feather mask with Gaussian blur ($51 \times 51$
   kernel) blends the boundary smoothly into the surrounding hair and clothing.

<!-- prettier-ignore -->
> [!NOTE]
> You can tune the balance between the generative facial reconstruction and the
> original photograph using the `--face-fidelity` flag.

## Perceptual post-processing

The final stage fine-tunes the visual presentation of the upscaled image using
classical color science techniques.

### Luminance-isolated CLAHE

Contrast Limited Adaptive Histogram Equalization (CLAHE) improves local contrast
and reveals hidden shadow details. However, applying CLAHE directly to RGB
channels alters color saturation and causes unwanted chromatic shifts.

Image Enhancer converts the image to the CIELAB color space and applies CLAHE
exclusively to the Lightness ($L^*$) channel. The chrominance channels ($a^*$
and $b^*$) remain untouched, preserving exact color science.

### Edge-preserving unsharp masking

To give textures a crisp, defined appearance without amplifying flat sensor
grain, Image Enhancer uses thresholded unsharp masking:

$$\Delta = I - \text{GaussianBlur}(I, \sigma)$$

$$\text{Mask} = |\Delta| \ge \tau$$

$$I_{\text{sharp}} = I + \Delta \times \text{strength} \times \text{Mask}$$

By zeroing out sharpening updates where the difference falls below the noise
threshold ($\tau$), flat backgrounds remain clean while sharp edges receive
targeted enhancement.

## Next steps

- Explore [CLI reference](cli-reference.md) to learn how to adjust tiling and
  filter thresholds.
- Read [Python API](python-api.md) to inspect code examples for custom pipelines.
