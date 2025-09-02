from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


def _read_any_image(bytes_in: bytes) -> np.ndarray:
    """Читает bytes -> BGR np.ndarray. (PDF сейчас не поддерживаем тут напрямую)."""
    arr = np.frombuffer(bytes_in, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Не удалось прочитать изображение (поддерживаются JPG/PNG).")
    return img


@dataclass
class PlanDetector:
    model_path: Path

    def __post_init__(self):
        self.model = YOLO(str(self.model_path))  # загружаем один раз
        # имена классов из модели
        self.names = self.model.names

    def detect(self, data: bytes, conf: float = 0.3) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Возвращает (annotated_bgr, detections_list),
        где detections_list = [{cls, name, conf, box:[x1,y1,x2,y2]}, ...]
        """
        img = _read_any_image(data)

        # предикт
        results = self.model.predict(
            img,
            conf=conf,
            imgsz=1536,   # можно увеличить/уменьшить
            verbose=False,
        )[0]

        dets: List[Dict[str, Any]] = []
        if results.boxes is not None and len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.cpu().numpy()
            cls = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), c, p in zip(xyxy, cls, confs):
                dets.append(
                    {
                        "cls": int(c),
                        "name": self.names.get(int(c), str(int(c))),
                        "conf": float(p),
                        "box": [float(x1), float(y1), float(x2), float(y2)],
                    }
                )

        # картинка с оверлеем
        annotated = results.plot()  # BGR numpy array
        return annotated, dets
