# app/main.py
from __future__ import annotations

from io import BytesIO
from typing import Optional

import numpy as np
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response

from app.preprocess import (
    read_image_or_pdf,
    auto_crop_plan,
    binarize_for_lines,
)
from app.plan_parser import parse_plan_image
from app.vectorize import json_to_svg

# YOLOv8 детектор
from app.ml.detector import FloorPlanDetector

app = FastAPI(title="Home Design Backend", version="0.2")

# инициализируем один раз (CPU на Mac — норм)
_detector: Optional[FloorPlanDetector] = None
def get_detector() -> FloorPlanDetector:
    global _detector
    if _detector is None:
        _detector = FloorPlanDetector("app/models/floorplan_yolov8.pt")
    return _detector


@app.get("/")
def root():
    return {"ok": True}


@app.post("/render")
async def render(file: UploadFile = File(...)):
    """
    Единый роут «минимум входа»: принимает PDF/JPG/PNG и отдаёт SVG линий (как раньше).
    На следующих шагах сюда же встроим семантику (двери/окна) и топологию.
    """
    raw = await file.read()

    # 1) читаем и автокропим цветное изображение
    img_bgr = read_image_or_pdf(raw, file.filename)
    img_bgr = auto_crop_plan(img_bgr)

    # 2) бинаризация под существующий парсер
    bin_img = binarize_for_lines(img_bgr)
    # превращаем в PNG bytes
    from PIL import Image
    buf = BytesIO()
    Image.fromarray(bin_img).save(buf, format="PNG")
    prepared_png = buf.getvalue()

    # 3) парсим текущим пайпом (да, пока всё ещё «криво», но это будет заменено)
    plan = parse_plan_image(prepared_png, known_scale_mm_per_px=None)
    plan.setdefault("image_size", {"w_px": bin_img.shape[1], "h_px": bin_img.shape[0]})

    svg = json_to_svg(plan)
    return Response(content=svg, media_type="image/svg+xml")


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = Query(0.3, ge=0.05, le=0.95)):
    """
    Временный DEV-роут: показывает, что видит YOLOv8 на твоём плане.
    Возвращает PNG с прямоугольниками детекций.
    Потом сольём в /render.
    """
    raw = await file.read()
    img_bgr = read_image_or_pdf(raw, file.filename)
    img_bgr = auto_crop_plan(img_bgr)

    det = get_detector()
    preds = det.detect(img_bgr, conf=conf, imgsz=1280)

    # нарисуем боксы поверх кадра
    vis = img_bgr.copy()
    for p in preds:
        x1, y1, x2, y2 = [int(v) for v in p["xyxy"]]
        name = p["cls_name"]
        score = p["score"]
        # подкраска
        color = (0, 255, 0)
        if "door" in name:
            color = (0, 165, 255)   # оранжевый
        elif "window" in name:
            color = (255, 191, 0)   # голубой
        elif "wall" in name:
            color = (0, 0, 255)     # красный
        import cv2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {score:.2f}"
        cv2.putText(vis, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # отдадим PNG
    import cv2
    _, png = cv2.imencode(".png", vis)
    return Response(content=png.tobytes(), media_type="image/png")
