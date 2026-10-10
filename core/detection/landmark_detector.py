"""
Road Landmark and Stationary Object Detector Module.
Detects static infrastructure such as dashed lane markings, zebra crossings,
turn arrows, road signs, and poles to provide immovable anchor points
for ultra-accurate ego speed estimation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass(frozen=True)
class RoadLandmark:
    """Represents a detected stationary landmark or road marking."""
    landmark_id: int
    class_name: str  # 'lane_dash', 'road_marking', 'pole', 'curb'
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    ground_contact_pt: Tuple[float, float]  # (u, v) bottom-center contact point


class BaseLandmarkDetector(ABC):
    """Abstract base class for stationary landmark detectors."""

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        road_roi_mask: Optional[np.ndarray] = None,
    ) -> List[RoadLandmark]:
        """Detects stationary road landmarks in the given video frame."""
        pass


class YOLOLandmarkDetector(BaseLandmarkDetector):
    """
    Deep-learning landmark detector powered by Ultralytics YOLO.
    Loads custom-trained weights (e.g. models/road_landmarks.pt).
    """

    def __init__(
        self,
        model_path: str = "models/road_landmarks.pt",
        confidence_threshold: float = 0.35,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._init_model()

    def _init_model(self) -> None:
        if self.model_path.exists():
            try:
                from ultralytics import YOLO
                self._model = YOLO(str(self.model_path))
            except Exception:
                self._model = None

    def detect(
        self,
        frame: np.ndarray,
        road_roi_mask: Optional[np.ndarray] = None,
    ) -> List[RoadLandmark]:
        if self._model is None:
            return []

        landmarks: List[RoadLandmark] = []
        try:
            results = self._model(frame, conf=self.confidence_threshold, verbose=False)
            if not results or len(results) == 0:
                return []

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return []

            names = self._model.names or {}
            for i, box in enumerate(boxes):
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = [float(v) for v in coords]
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                raw_name = names.get(cls_id, "road_marking").lower()

                # Normalize class name
                if "dash" in raw_name or "lane" in raw_name:
                    cls_name = "lane_dash"
                elif "pole" in raw_name or "sign" in raw_name:
                    cls_name = "pole"
                elif "curb" in raw_name or "barrier" in raw_name:
                    cls_name = "curb"
                else:
                    cls_name = "road_marking"

                contact_u = (x1 + x2) / 2.0
                contact_v = y2

                # If ROI mask is given, ensure contact point is within the road corridor
                if road_roi_mask is not None:
                    cu, cv_idx = int(round(contact_u)), int(round(contact_v))
                    if 0 <= cv_idx < road_roi_mask.shape[0] and 0 <= cu < road_roi_mask.shape[1]:
                        if road_roi_mask[cv_idx, cu] == 0:
                            continue

                landmarks.append(
                    RoadLandmark(
                        landmark_id=i + 1,
                        class_name=cls_name,
                        bbox=(x1, y1, x2, y2),
                        confidence=round(conf, 3),
                        ground_contact_pt=(round(contact_u, 1), round(contact_v, 1)),
                    )
                )
        except Exception:
            return []

        return landmarks


class ClassicalRoadMarkingDetector(BaseLandmarkDetector):
    """
    High-performance classical computer vision detector for road markings.
    Extracts high-contrast dashed lane lines, stop lines, and zebra crossings
    using adaptive thresholding, HLS lightness decomposition, and contour geometry.
    Operates immediately out-of-the-box without requiring pretrained weights.
    """

    def __init__(
        self,
        min_contour_area: float = 40.0,
        max_contour_area: float = 25000.0,
    ):
        self.min_area = min_contour_area
        self.max_area = max_contour_area

    def detect(
        self,
        frame: np.ndarray,
        road_roi_mask: Optional[np.ndarray] = None,
    ) -> List[RoadLandmark]:
        h, w = frame.shape[:2]
        # Only process lower road area (below 40% height)
        road_y_start = int(h * 0.40)
        road_crop = frame[road_y_start:, :]

        # Convert to HLS color space (L channel isolates high-reflectance paint)
        hls = cv2.cvtColor(road_crop, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]

        # Contrast enhancement & adaptive thresholding
        blurred = cv2.GaussianBlur(l_channel, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, -12
        )

        # Apply ROI mask if provided
        if road_roi_mask is not None:
            mask_crop = road_roi_mask[road_y_start:, :]
            thresh = cv2.bitwise_and(thresh, thresh, mask=mask_crop)

        # Morphological opening to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        landmarks: List[RoadLandmark] = []
        l_id = 1

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            abs_y1 = float(y + road_y_start)
            abs_y2 = float(y + ch + road_y_start)
            abs_x1 = float(x)
            abs_x2 = float(x + cw)

            aspect_ratio = float(ch) / max(1.0, float(cw))

            # Classify marking type based on geometric profile in perspective
            if aspect_ratio >= 1.2 and cw < (w * 0.15):
                cls_name = "lane_dash"
            elif aspect_ratio < 0.8 and cw > (w * 0.10):
                cls_name = "road_marking"  # Transversal stop line or zebra stripe
            else:
                cls_name = "road_marking"

            contact_u = (abs_x1 + abs_x2) / 2.0
            contact_v = abs_y2

            landmarks.append(
                RoadLandmark(
                    landmark_id=l_id,
                    class_name=cls_name,
                    bbox=(abs_x1, abs_y1, abs_x2, abs_y2),
                    confidence=0.80,
                    ground_contact_pt=(round(contact_u, 1), round(contact_v, 1)),
                )
            )
            l_id += 1

        return landmarks


class HybridLandmarkDetector(BaseLandmarkDetector):
    """
    Intelligent wrapper: Automatically uses YOLOLandmarkDetector when custom
    weights exist; seamlessly falls back to ClassicalRoadMarkingDetector otherwise.
    """

    def __init__(self, yolo_model_path: str = "models/road_landmarks.pt"):
        self.yolo_detector = YOLOLandmarkDetector(model_path=yolo_model_path)
        self.classical_detector = ClassicalRoadMarkingDetector()

    @property
    def is_yolo_active(self) -> bool:
        return self.yolo_detector._model is not None

    def detect(
        self,
        frame: np.ndarray,
        road_roi_mask: Optional[np.ndarray] = None,
    ) -> List[RoadLandmark]:
        if self.is_yolo_active:
            landmarks = self.yolo_detector.detect(frame, road_roi_mask)
            if landmarks:
                return landmarks
        return self.classical_detector.detect(frame, road_roi_mask)
