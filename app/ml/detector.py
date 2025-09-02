# ml/detector.py
from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
from PIL import Image
import cv2  # ultralytics рисует в BGR, так удобнее кодировать PNG
from ultralytics import YOLO


class PlanDetector:
    """
    Обёртка над YOLOv8 для детекции элементов на планах квартир.
    - загружает веса из models/floorplan_yolov8.pt (по умолчанию)
    - принимает PDF/JPG/PNG как байты (для PDF нужен внешний препроцесс; здесь работаем с изображениями)
    - возвращает: (annotated_png_bytes, detections_json)
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        if model_path is None:
            # ml/detector.py -> .. (root) / models / floorplan_yolov8.pt
            root = Path(__file__).resolve().parents[1]
            model_path = root / "models" / "floorplan_yolov8.pt"

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Не найден файл весов YOLO: {self.model_path}\n"
                f"Положи модель в {self.model_path} (имя: floorplan_yolov8.pt)."
            )

        # Загружаем модель один раз
        self.model = YOLO(str(self.model_path))

        # Если в весах сохранена своя карта классов — возьмём её
        # Иначе — используем дефолтные имена
        try:
            names = self.model.names  # dict or list
            if isinstance(names, dict):
                # {0: 'wall', 1: 'door', ...}
                self.class_names = [names[k] for k in sorted(names)]
            else:
                self.class_names = list(names)
        except Exception:
            self.class_names = [
                "wall", "door", "window", "column", "stair", "text", "other"
            ]

    # ---------- utils ----------

    @staticmethod
    def _bytes_to_bgr(data: bytes) -> np.ndarray:
        """
        Надёжно превращаем входные байты в BGR-изображение (H, W, 3).
        Работает для PNG/JPG. Для PDF используй препроцесс в app.preprocess (в /render).
        """
        # Через PIL (надёжнее, чем imdecode на разном контенте)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.array(img)  # RGB
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return bgr

    @staticmethod
    def _bgr_to_png_bytes(img_bgr: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".png", img_bgr)
        if not ok:
            raise RuntimeError("Не удалось закодировать PNG.")
        return buf.tobytes()

    # ---------- public API ----------

    def detect(self, data: bytes, conf: float = 0.3) -> Tuple[bytes, Dict[str, Any]]:
        """
        Запускает инференс и возвращает:
        - annotated_png_bytes: PNG с нарисованными прямоугольниками
        - detections: JSON со списком найденных боксов
        """
        img_bgr = self._bytes_to_bgr(data)

        # Ultralytics принимает numpy в RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.model.predict(img_rgb, conf=conf, verbose=False)

        if not results:
            annotated = img_bgr
            detections_json: Dict[str, Any] = {"detections": []}
        else:
            r = results[0]

            # Картинка с разметкой (BGR)
            annotated = r.plot()  # already BGR

            det_list: List[Dict[str, Any]] = []
            boxes = r.boxes  # Boxes object
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()

                for i in range(xyxy.shape[0]):
                    x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                    score = float(confs[i])
                    cls_id = int(clss[i]) if not np.isnan(clss[i]) else -1
                    cls_name = (
                        self.class_names[cls_id]
                        if 0 <= cls_id < len(self.class_names)
                        else str(cls_id)
                    )
                    det_list.append(
                        {
                            "bbox_xyxy": [x1, y1, x2, y2],
                            "score": score,
                            "class_id": cls_id,
                            "class_name": cls_name,
                        }
                    )

            detections_json = {"detections": det_list}

        png_bytes = self._bgr_to_png_bytes(annotated)
        return png_bytes, detections_json
