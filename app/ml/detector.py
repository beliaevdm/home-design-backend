# app/ml/detector.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError(
        "Ultralytics (YOLOv8) is required. Make sure 'ultralytics' is in requirements.txt and installed."
    ) from e


CLASS_COLORS = {
    # базовые цвета для отрисовки (BGR)
    "door": (0, 165, 255),    # оранжевый
    "window": (255, 191, 0),  # голубой (в BGR это желто-голубой, но норм)
    "wall": (0, 0, 255),      # красный (как ориентир)
}

class FloorPlanDetector:
    def __init__(self, weights_path: str = "app/models/floorplan_yolov8.pt", device: Optional[str] = None):
        p = Path(weights_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Weights not found: {p}. Put your YOLOv8 weights to this path."
            )
        # device=None -> сам выберет (CPU на Mac — ок)
        self.model = YOLO(str(p))
        self.names = self.model.names  # словарь id->имя класса

    def detect(self, img_bgr: np.ndarray, conf: float = 0.3, imgsz: int = 1280) -> List[Dict[str, Any]]:
        """
        Возвращает список детекций:
        [{ 'cls_id': int, 'cls_name': str, 'score': float, 'xyxy': [x1,y1,x2,y2] }, ...]
        """
        # YOLO ожидает RGB
        img_rgb = img_bgr[..., ::-1]
        results = self.model.predict(
            source=img_rgb,
            conf=conf,
            imgsz=imgsz,
            verbose=False,
            device=None,
        )
        out: List[Dict[str, Any]] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None:
            return out
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            name = str(self.names.get(int(cls_ids[i]), int(cls_ids[i])))
            out.append(
                {
                    "cls_id": int(cls_ids[i]),
                    "cls_name": name.lower(),
                    "score": float(confs[i]),
                    "xyxy": [float(v) for v in xyxy[i].tolist()],
                }
            )
        return out
