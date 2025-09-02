from __future__ import annotations

from pathlib import Path

import cv2
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import Response

# импортируем из под-пакета app.ml (после того как ml переместили в app/)
from app.ml.detector import PlanDetector

app = FastAPI(title="Home Design Backend", version="0.2-dev")

# путь к корню репозитория (…/home-design-backend)
ROOT = Path(__file__).resolve().parents[1]

# лениво инициализируем YOLO один раз
_detector: PlanDetector | None = None


def detector() -> PlanDetector:
    global _detector
    if _detector is None:
        _detector = PlanDetector(model_path=ROOT / "models" / "floorplan_yolov8.pt")
    return _detector


@app.get("/")
def root():
    return {"ok": True, "msg": "Home Design Backend is running"}


@app.post(
    "/detect",
    responses={200: {"content": {"image/png": {}}}},
    summary="DEV: показать детекции YOLO",
)
async def detect(
    file: UploadFile = File(...),
    conf: float = Query(0.30, ge=0.05, le=0.95, description="Confidence threshold"),
):
    """
    Временная DEV-ручка:
    - принимает JPG/PNG (PDF пока не конвертируем здесь);
    - прогоняет через YOLO;
    - возвращает **PNG** с прямоугольниками/классами (оверлей).
    """
    raw = await file.read()
    ann_bgr, dets = detector().detect(raw, conf=conf)

    ok, buf = cv2.imencode(".png", ann_bgr)
    if not ok:
        return Response(status_code=500, content=b"Encode error")

    # для быстрой проверки — кладём первые детекции в заголовок (обрезано)
    headers = {"X-Detections": str(dets[:50])}
    return Response(content=buf.tobytes(), media_type="image/png", headers=headers)


@app.post("/render")
async def render(file: UploadFile = File(...)):
    # заглушка — сюда вернём финальную сборку плана поверх детекций YOLO
    return {
        "ok": False,
        "msg": "render() будет собирать финальный чертёж на базе YOLO-детекций. Сейчас заглушка.",
    }
