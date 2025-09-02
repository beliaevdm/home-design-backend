from __future__ import annotations
import os
from typing import Any, Dict, List

import numpy as np
from ultralytics import YOLO

# где лежит вес?
# по умолчанию: models/floorplan_yolov8.pt (как в твоём репо)
_DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "floorplan_yolov8.pt")

_CLASS_NAMES = {
    0: "Wall",
    1: "Window",
    2: "Door",        # если в весах есть
    3: "Column",
    4: "Dimension",   # метки размеров/подписи — будем игнорить при линиях
    # остальное как есть в модели
}

class PlanDetector:
    def __init__(self, weights_path: str | None = None):
        weights_path = weights_path or os.getenv("FLOORPLAN_YOLO", _DEFAULT_WEIGHTS)
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
        self.model = YOLO(weights_path)

    def detect(self, img_bgr: np.ndarray, conf: float = 0.30) -> Dict[str, Any]:
        """
        Возвращает:
        {
          "image_size": {"w_px": W, "h_px": H},
          "detections": [{"class": "Wall", "conf": 0.9, "bbox":[x1,y1,x2,y2]}, ...]
        }
        """
        H, W = img_bgr.shape[:2]
        results = self.model.predict(img_bgr[:, :, ::-1], conf=conf, verbose=False)  # RGB
        detections: List[Dict[str, Any]] = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and r.boxes.xyxy is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                    name = _CLASS_NAMES.get(int(k), str(int(k)))
                    detections.append({
                        "class": name,
                        "conf": float(c),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    })
        return {"image_size": {"w_px": int(W), "h_px": int(H)}, "detections": detections}
