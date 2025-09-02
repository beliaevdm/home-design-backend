from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
from ultralytics import YOLO


class PlantDetector:
    """
    Обёртка над YOLOv8 для детекции элементов на планах квартир.
    Может отдавать PNG-оверлей и «сырой» JSON детекций.
    """

    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        self._model: Optional[YOLO] = None
        self._weights_path = self._resolve_weights(weights_path)
        self._device = device  # на mac mps, на CPU/без CUDA — None

    # -------------------- helpers -------------------- #

    def _resolve_weights(self, manual: Optional[str]) -> Path:
        if manual:
            p = Path(manual).expanduser().resolve()
            if not p.exists():
                raise FileNotFoundError(p)
            return p

        here = Path(__file__).resolve().parent      # .../app/ml
        app_root = here.parent                      # .../app
        cwd = Path.cwd()

        candidates: List[Path] = [
            app_root / "models" / "floorplan_yolov8.pt",
            app_root / "models" / "best.pt",
            cwd / "models" / "floorplan_yolov8.pt",
            cwd / "models" / "best.pt",
        ]
        for c in candidates:
            if c.exists():
                return c

        raise FileNotFoundError(
            "Не найден файл весов YOLO. Положи 'floorplan_yolov8.pt' или 'best.pt' "
            "в app/models или в models/ в корне проекта."
        )

    def _ensure_model(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(str(self._weights_path))
        return self._model

    @staticmethod
    def _decode_image(img_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Не удалось прочитать изображение")
        return img

    @staticmethod
    def _draw_boxes(img: np.ndarray,
                    boxes_xyxy: np.ndarray,
                    clses: np.ndarray,
                    confs: np.ndarray,
                    names: dict) -> np.ndarray:
        out = img.copy()
        for (x1, y1, x2, y2), cls_id, conf in zip(boxes_xyxy, clses, confs):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{names.get(int(cls_id), int(cls_id))} {float(conf):.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 2, y1), (0, 0, 255), -1)
            cv2.putText(out, label, (x1 + 1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    @staticmethod
    def _encode_png(img_bgr: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".png", img_bgr)
        if not ok:
            raise RuntimeError("PNG encode failed")
        return buf.tobytes()

    # -------------------- public API -------------------- #

    def detect_png(self, img_bytes: bytes, conf: float = 0.3) -> bytes:
        """
        На вход: байты JPG/PNG.
        Возвращает: PNG-байты с прямоугольниками поверх.
        """
        model = self._ensure_model()
        img = self._decode_image(img_bytes)

        res = model.predict(
            source=img,
            conf=float(conf),
            imgsz=1280,
            device=self._device,
            verbose=False
        )[0]

        boxes = res.boxes
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            return self._encode_png(img)

        overlay = self._draw_boxes(
            img,
            boxes.xyxy.cpu().numpy(),
            boxes.cls.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            res.names
        )
        return self._encode_png(overlay)

    def detect_json(self, img_bytes: bytes, conf: float = 0.3) -> Dict[str, Any]:
        """
        На вход: байты JPG/PNG.
        Возвращает: JSON: список боксов [{class, conf, bbox:[x1,y1,x2,y2]}], размер изображения.
        """
        model = self._ensure_model()
        img = self._decode_image(img_bytes)

        res = model.predict(
            source=img,
            conf=float(conf),
            imgsz=1280,
            device=self._device,
            verbose=False
        )[0]

        W = int(img.shape[1]); H = int(img.shape[0])
        out: Dict[str, Any] = {"image_size": {"w_px": W, "h_px": H}, "detections": []}

        boxes = res.boxes
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            return out

        xyxy = boxes.xyxy.cpu().numpy()
        clses = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        names = res.names

        for (x1, y1, x2, y2), cls_id, score in zip(xyxy, clses, confs):
            out["detections"].append({
                "class": names.get(int(cls_id), str(int(cls_id))),
                "conf": float(score),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })

        return out
