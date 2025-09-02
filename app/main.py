from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response

from app.preprocess import read_image_or_pdf, auto_crop_plan
from app.vectorize_yolo import svg_from_image_and_yolo
from ml.detector import PlanDetector

app = FastAPI(title="Home Design Backend", version="0.1")

# грузим YOLO один раз
detector = PlanDetector()

@app.get("/")
def root():
    return {"ok": True, "endpoints": ["/render"]}

@app.post("/render")
async def render(
    file: UploadFile = File(...),
    conf: float = Query(0.30, ge=0.05, le=0.95, description="YOLO confidence"),
) -> Response:
    """
    Один вход — PDF/JPG/PNG. На выход — SVG с векторными стенами и окнами.
    """
    raw = await file.read()

    # 1) загрузка + автообрезка
    img = read_image_or_pdf(raw, file.filename if file.filename else None)
    img = auto_crop_plan(img)

    # 2) YOLO-детекции
    detections = detector.detect(img, conf=conf)

    # 3) векторизация по маске стен (LSD + склейка) и отрисовка
    svg = svg_from_image_and_yolo(img, detections)

    return Response(content=svg, media_type="image/svg+xml")
