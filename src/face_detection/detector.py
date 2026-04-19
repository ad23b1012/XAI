"""
Face Detection & Landmark Extraction using MediaPipe Face Mesh.

Detects faces in images and extracts 468/478 facial landmarks for
downstream AU feature extraction and face cropping.
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image


@dataclass
class FaceDetectionResult:
    """Result of face detection on a single image."""
    face_found: bool
    landmarks: Optional[np.ndarray] = None  # Shape: (468, 3) — normalized (x, y, z)
    landmarks_pixel: Optional[np.ndarray] = None  # Shape: (468, 2) — pixel coordinates
    face_crop: Optional[np.ndarray] = None  # Cropped face region (BGR)
    face_crop_pil: Optional[Image.Image] = None  # Cropped face as PIL Image
    bbox: Optional[tuple] = None  # (x1, y1, x2, y2) bounding box
    image_shape: Optional[tuple] = None  # (height, width) of original image


class FaceDetector:
    """
    MediaPipe Face Mesh wrapper for face detection and landmark extraction.

    Extracts 468 facial landmarks (or 478 with iris refinement) from input images.
    Provides face cropping with configurable padding for downstream processing.
    """

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        refine_landmarks: bool = True,
        face_crop_padding: float = 0.2,
    ):
        """
        Initialize the face detector.

        Args:
            static_image_mode: Process each image independently (True for static images).
            max_num_faces: Maximum number of faces to detect.
            min_detection_confidence: Minimum confidence threshold for face detection.
            refine_landmarks: Whether to use 478 landmarks (includes iris).
            face_crop_padding: Fraction of face bounding box to add as padding.
        """
        self.face_crop_padding = face_crop_padding

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            refine_landmarks=refine_landmarks,
        )

    def detect(self, image: np.ndarray) -> FaceDetectionResult:
        """
        Detect face and extract landmarks from an image.

        Args:
            image: Input image in BGR format (as read by cv2.imread).

        Returns:
            FaceDetectionResult containing landmarks, face crop, and bounding box.
        """
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return FaceDetectionResult(face_found=False, image_shape=(h, w))

        # Take the first detected face
        face_landmarks = results.multi_face_landmarks[0]

        # Extract normalized landmarks (x, y, z) — values in [0, 1]
        landmarks = np.array(
            [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark],
            dtype=np.float32,
        )

        # Convert to pixel coordinates
        landmarks_pixel = np.array(
            [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark],
            dtype=np.float32,
        )

        # Compute bounding box from landmarks
        x_coords = landmarks_pixel[:, 0]
        y_coords = landmarks_pixel[:, 1]
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())

        # Add padding
        face_w = x_max - x_min
        face_h = y_max - y_min
        pad_x = int(face_w * self.face_crop_padding)
        pad_y = int(face_h * self.face_crop_padding)

        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(w, x_max + pad_x)
        y2 = min(h, y_max + pad_y)

        # Crop face
        face_crop = image[y1:y2, x1:x2].copy()
        face_crop_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

        return FaceDetectionResult(
            face_found=True,
            landmarks=landmarks,
            landmarks_pixel=landmarks_pixel,
            face_crop=face_crop,
            face_crop_pil=face_crop_pil,
            bbox=(x1, y1, x2, y2),
            image_shape=(h, w),
        )

    def detect_from_path(self, image_path: str) -> FaceDetectionResult:
        """Load an image from path and detect face."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.detect(image)

    def detect_from_pil(self, pil_image: Image.Image) -> FaceDetectionResult:
        """Detect face from a PIL Image."""
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return self.detect(image)

    def get_face_width(self, landmarks: np.ndarray) -> float:
        """
        Compute face width from landmarks for normalization.

        Uses the distance between landmarks 234 (left cheek) and 454 (right cheek)
        as a stable face width measurement.

        Args:
            landmarks: Normalized landmarks array of shape (468, 3).

        Returns:
            Face width as a normalized distance.
        """
        left_cheek = landmarks[234, :2]
        right_cheek = landmarks[454, :2]
        return np.linalg.norm(left_cheek - right_cheek)

    def close(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
