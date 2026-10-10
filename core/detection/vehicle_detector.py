"""
Vehicle detection module (Phase 6).
Provides modular vehicle detection supporting YOLOv8 and synthetic/fallback detectors.
Detects cars, trucks, buses, and motorcycles.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """Represents a single object detection in an image."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Returns (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Ground contact point of the vehicle."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    @property
    def area(self) -> float:
        return self.width * self.height


# COCO dataset vehicle category mappings
COCO_VEHICLE_CLASSES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class BaseVehicleDetector(ABC):
    """Abstract base class for vehicle detectors to ensure modularity and replaceability."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detects vehicles in the input frame.

        Args:
            image: BGR image frame (numpy array).

        Returns:
            List of Detection objects.
        """
        pass


class YOLOVehicleDetector(BaseVehicleDetector):
    """
    YOLO-based vehicle detector using Ultralytics.
    Filters detections to target vehicle classes: car, motorcycle, bus, truck.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
        target_classes: Optional[Set[str]] = None,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.conf_threshold = confidence_threshold
        self.target_classes = target_classes or {"car", "truck", "bus", "motorcycle"}
        self.device = device
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)
        except Exception as e:
            # Fallback to None if model download/weights fail
            self._model = None

    def detect(self, image: np.ndarray) -> List[Detection]:
        if self._model is None:
            return []

        results = self._model(
            image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = self._model.names.get(cls_id, "unknown")

            if cls_name in self.target_classes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                detections.append(
                    Detection(
                        x1=round(xyxy[0], 1),
                        y1=round(xyxy[1], 1),
                        x2=round(xyxy[2], 1),
                        y2=round(xyxy[3], 1),
                        confidence=round(conf, 3),
                        class_id=cls_id,
                        class_name=cls_name,
                    )
                )

        return detections


class SyntheticVehicleDetector(BaseVehicleDetector):
    """
    Fast color and contour-based detector specifically designed for synthetic
    benchmark footage or offline test environments without pretrained weight downloads.
    """

    def __init__(self, target_class: str = "car", min_area: float = 400.0):
        self.target_class = target_class
        self.min_area = min_area

    def detect(self, image: np.ndarray) -> List[Detection]:
        detections: List[Detection] = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Detect red vehicle in synthetic footage
        mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        mask = mask1 | mask2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append(
                    Detection(
                        x1=float(x),
                        y1=float(y),
                        x2=float(x + w),
                        y2=float(y + h),
                        confidence=0.92,
                        class_id=2,
                        class_name=self.target_class,
                    )
                )

        return detections
