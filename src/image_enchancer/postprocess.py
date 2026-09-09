import cv2
import numpy as np


class PostProcessor:
    """
    High-fidelity image post-processing suite: micro-contrast enhancement,
    edge-preserving unsharp masking, and color space preservation.
    """

    @staticmethod
    def apply_clahe(
        image_bgr: np.ndarray,
        clip_limit: float = 1.5,
        tile_grid_size: tuple = (8, 8),
    ) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization strictly on the
        Lightness (L*) channel in CIELAB color space, preserving chromatic balance.
        """
        if clip_limit <= 0.0:
            return image_bgr

        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_l = clahe.apply(l_channel)

        merged_lab = cv2.merge([enhanced_l, a_channel, b_channel])
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def apply_unsharp_mask(
        image_bgr: np.ndarray,
        strength: float = 0.4,
        sigma: float = 1.0,
        threshold: int = 4,
    ) -> np.ndarray:
        """
        Applies thresholded edge-preserving unsharp masking.
        High-frequency components below `threshold` are ignored to prevent noise amplification.
        """
        if strength <= 0.0:
            return image_bgr

        blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigma)
        diff = image_bgr.astype(np.int16) - blurred.astype(np.int16)

        # Apply threshold deadband to protect flat areas
        mask = np.abs(diff) >= threshold
        enhanced = image_bgr.astype(np.float32) + (diff * strength * mask).astype(np.float32)

        return np.clip(enhanced, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def match_color_distribution(
        source_bgr: np.ndarray,
        reference_bgr: np.ndarray,
    ) -> np.ndarray:
        """
        Reinhard color transfer in CIELAB space to ensure the enhanced output matches
        the exact color profile and white balance of the original input.
        """
        # Resize reference to match source dimensions for statistical comparison
        ref_resized = cv2.resize(
            reference_bgr,
            (source_bgr.shape[1], source_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Calculate mean and standard deviation for each channel
        src_mean, src_std = src_lab.mean(axis=(0, 1)), src_lab.std(axis=(0, 1))
        ref_mean, ref_std = ref_lab.mean(axis=(0, 1)), ref_lab.std(axis=(0, 1))

        # Avoid zero division
        src_std = np.maximum(src_std, 1e-6)

        # Standardize and scale
        matched_lab = (src_lab - src_mean) * (ref_std / src_std) + ref_mean
        matched_lab = np.clip(matched_lab, 0.0, 255.0).astype(np.uint8)

        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def pre_denoise(
        image_bgr: np.ndarray,
        strength: float = 0.5,
    ) -> np.ndarray:
        """
        Bilateral filtering for edge-preserving noise reduction prior to super-resolution.
        """
        if strength <= 0.0:
            return image_bgr

        diameter = int(5 + strength * 4)  # 5 to 9
        sigma_color = 20.0 + strength * 40.0
        sigma_space = 20.0 + strength * 40.0

        return cv2.bilateralFilter(image_bgr, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)
