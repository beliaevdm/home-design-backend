# app/main.py
from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response

from ml.detector import PlanDetector

app = FastAPI(
    title="Home Design Backend",
    version="0.2",
    description=(
        "DEV: YOLOv8 детекция элементов на плане. "
        "Эндпоинт /detect принимает изображение (JPG/PNG), "
        "возвращает PNG с прямоугольниками детекций."
    ),
)

# загружаем модель один раз на старте
detector = PlanDetector()


@app.get("/")
def root():
    return {"ok": True}


@app.post(
    "/detect",
    summary="Показать, что видит YOLOv8 (DEV)",
    responses={200: {"content": {"image/png": {}}}},
)
async def detect(
    file: UploadFile = File(..., description="JPG/PNG кадр плана"),
    conf: float = Query(
        0.30,
        ge=0.05,
        le=0.95,
        description="Порог уверенности YOLOv8 (по умолчанию 0.30)",
    ),
):
    """
    Принимает картинку → детектит → возвращает PNG с нарисованными боксами.
    """
    data = await file.read()
    annotated_png, _ = detector.detect(data, conf=conf)
    # Возвращаем как изображение, а не JSON-строку
    return Response(
        content=annotated_png,
        media_type="image/png",
        headers={"Content-Disposition": 'inline; filename="detections.png"'},
    )
