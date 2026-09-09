from pathlib import Path
from typing import Optional, List, Tuple
import cv2
import numpy as np
import torch
from spandrel import ImageModelDescriptor
from image_enchancer.models import resolve_model_path, load_neural_model

# Canonical 5-point face landmarks for 512x512 aligned canvas
STD_FACE_TEMPLATE_512 = np.array(
    [
        [192.9814, 239.9470],  # right eye
        [318.9027, 240.1936],  # left eye
        [256.6341, 314.0193],  # nose
        [201.2611, 371.4104],  # right mouth corner
        [313.9890, 371.1511],  # left mouth corner
    ],
    dtype=np.float32,
)


class FaceRestorationEngine:
    """
    State-of-the-art face enhancement engine using YuNet landmark detection and GFPGAN restoration.
    """

    def __init__(
        self,
        gfpgan_model: Optional[ImageModelDescriptor] = None,
        yunet_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load GFPGAN
        if gfpgan_model is not None:
            self.gfpgan = gfpgan_model
        else:
            self.gfpgan = load_neural_model("gfpgan", device=self.device)

        # Load YuNet detector
        if yunet_path is None:
            yunet_path = resolve_model_path("yunet")

        self.detector = cv2.FaceDetectorYN.create(
            model=str(yunet_path),
            config="",
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )

        # Pre-compute 512x512 elliptical feather mask
        self.soft_mask = self._create_face_mask(512, 512)

    @staticmethod
    def _create_face_mask(h: int, w: int) -> np.ndarray:
        """
        Creates an elliptical soft mask with Gaussian blur feathering to eliminate border seams.
        """
        mask = np.zeros((h, w), dtype=np.float32)
        center = (w // 2, int(h * 0.55))
        axes = (int(w * 0.38), int(h * 0.46))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        # Gaussian feathering
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        return mask

    def detect_faces(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Detects faces and 5 facial landmarks using YuNet.
        Returns array of detected faces or None.
        """
        h, w = image_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image_bgr)
        return faces

    def restore_faces(
        self,
        image_bgr: np.ndarray,
        fidelity: float = 0.85,
    ) -> Tuple[np.ndarray, int]:
        """
        Detects and enhances all faces in the image, blending them back seamlessly.

        :param image_bgr: Full image (BGR, uint8)
        :param fidelity: Balance between restored face and original context (0.0 to 1.0)
        :return: (Enhanced image, count of faces restored)
        """
        faces = self.detect_faces(image_bgr)
        if faces is None or len(faces) == 0:
            return image_bgr, 0

        h, w = image_bgr.shape[:2]
        output_image = image_bgr.copy()
        restored_count = 0

        for face in faces:
            # Score threshold filter
            confidence = face[-1]
            if confidence < 0.5:
                continue

            # Extract 5 landmarks: right eye, left eye, nose, right mouth, left mouth
            landmarks = face[4:14].reshape((5, 2)).astype(np.float32)

            # Compute affine partial similarity transformation (rotation, scale, translation)
            matrix, _ = cv2.estimateAffinePartial2D(landmarks, STD_FACE_TEMPLATE_512, method=cv2.LMEDS)
            if matrix is None:
                continue

            # Warp face to 512x512
            aligned_face = cv2.warpAffine(
                image_bgr,
                matrix,
                (512, 512),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

            # Convert to RGB float32 tensor [0, 1]
            face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

            # Neural restoration pass
            with torch.inference_mode():
                restored_tensor = self.gfpgan(tensor)

            restored_np = restored_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            restored_np = np.clip(restored_np * 255.0, 0.0, 255.0).astype(np.uint8)
            restored_bgr = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)

            # Blend with fidelity weight
            if fidelity < 1.0:
                restored_bgr = cv2.addWeighted(
                    restored_bgr, fidelity,
                    aligned_face, 1.0 - fidelity,
                    0.0
                )

            # Inverse affine transform to paste back into original canvas
            inv_matrix = cv2.invertAffineTransform(matrix)

            warped_restored = cv2.warpAffine(
                restored_bgr,
                inv_matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

            warped_mask = cv2.warpAffine(
                self.soft_mask,
                inv_matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

            # Alpha blend back into image
            mask_3c = warped_mask[:, :, None]
            output_image = (
                output_image.astype(np.float32) * (1.0 - mask_3c)
                + warped_restored.astype(np.float32) * mask_3c
            ).astype(np.uint8)

            restored_count += 1

        return output_image, restored_count
