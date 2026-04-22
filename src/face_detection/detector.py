"""
Face Detection & Landmark Extraction using MediaPipe Tasks SDK.

Natively supports Python 3.12+ and MediaPipe 0.10.x+ by utilizing the
modern `FaceLandmarker` object configurations.

Detects faces in images and extracts 468/478 facial landmarks for
downstream AU feature extraction and face cropping.
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional
from PIL import Image

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


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
    MediaPipe Tasks FaceLandmarker wrapper for face detection.

    Extracts 468 facial landmarks from input images using the offline
    `face_landmarker.task` model blob.
    Provides face cropping with configurable padding for downstream processing.
    """

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        refine_landmarks: bool = True,
        face_crop_padding: float = 0.2,
        model_asset_path: str = "src/face_detection/models/face_landmarker.task"
    ):
        """
        Initialize the face detector via MediaPipe Tasks SDK.
        """
        self.face_crop_padding = face_crop_padding

        if not os.path.exists(model_asset_path):
            raise FileNotFoundError(f"Missing MediaPipe task file at {model_asset_path}. "
                                     "Please download the float16 task blob from Google.")

        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, image: np.ndarray) -> FaceDetectionResult:
        """
        Detect face and extract landmarks from an image.

        Args:
            image: Input image in BGR format (as read by cv2.imread).
        """
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to MediaPipe internal Image object required by modern Tasks SDK
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        results = self.detector.detect(mp_image)

        if not results.face_landmarks:
            return FaceDetectionResult(face_found=False, image_shape=(h, w))

        # Take the first detected face (list of normalized landmarks)
        face_landmarks = results.face_landmarks[0]

        # Extract normalized landmarks (x, y, z) — values in [0, 1]
        landmarks = np.array(
            [(lm.x, lm.y, lm.z) for lm in face_landmarks],
            dtype=np.float32,
        )

        # Truncate to exactly 468 points to perfectly match the POSTER V2 structural requirements
        # (FaceLandmarker might return 478 if irises are computed)
        landmarks = landmarks[:468]

        # Convert to pixel coordinates
        landmarks_pixel = np.array(
            [(lm[0] * w, lm[1] * h) for lm in landmarks],
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
        try:
            face_crop_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        except Exception:
            face_crop_pil = None

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
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.detect(image)

    def detect_from_pil(self, pil_image: Image.Image) -> FaceDetectionResult:
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return self.detect(image)

    def get_face_width(self, landmarks: np.ndarray) -> float:
        left_cheek = landmarks[234, :2]
        right_cheek = landmarks[454, :2]
        return np.linalg.norm(left_cheek - right_cheek)

    def close(self):
        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
