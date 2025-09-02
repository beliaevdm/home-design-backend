from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import Response

from .plan_parser import parse_plan_image
from .vectorize import json_to_svg
from .scale import calc_mm_per_px, apply_scale
from .normalize import normalize

from .ml.detector import PlantDetector
from .yolo_to_plan import detections_to_plan

app = FastAPI(title="Home Design Backend", version="0.1")

detector = PlantDetector()


@app.get("/")
def root():
    return {"ok": True}


# -------- DEV: показать «как видит YOLO» -------- #

@app.post("/detect", summary="DEV: показать детекции YOLO (оверлей PNG)")
async def detect(file: UploadFile = File(...),
                 conf: float = Query(0.55, ge=0.05, le=0.95)):
    try:
        data = await file.read()
        png = detector.detect_png(data, conf=conf)
        return Response(content=png, media_type="image/png")
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"YOLO weights not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")


@app.post("/detect-json", summary="DEV: сырые детекции YOLO (JSON)")
async def detect_json(file: UploadFile = File(...),
                      conf: float = Query(0.55, ge=0.05, le=0.95)):
    try:
        data = await file.read()
        result = detector.detect_json(data, conf=conf)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"YOLO weights not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")


# -------- Основной путь: файл -> YOLO -> линии -> SVG -------- #

@app.post("/render-yolo", summary="Файл (JPG/PNG) -> YOLO -> векторный план (SVG)")
async def render_yolo(file: UploadFile = File(...),
                      conf: float = Query(0.55, ge=0.05, le=0.95)):
    """
    Минимальный путь для продакшена: загрузил картинку — получил пригодный SVG.
    """
    try:
        raw = await file.read()
        det = detector.detect_json(raw, conf=conf)        # 1) детекции
        plan = detections_to_plan(det, min_conf=conf)     # 2) постпроцесс в линии
        svg = json_to_svg(plan)                           # 3) рендер SVG
        return Response(content=svg, media_type="image/svg+xml")
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"YOLO weights not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Render failed: {e}")


# -------- Твоё прежнее API оставлено без изменений -------- #

@app.post("/parse-plan")
async def parse_plan(file: UploadFile = File(...), known_scale_mm_per_px: float | None = None):
    content = await file.read()
    plan = parse_plan_image(content, known_scale_mm_per_px=known_scale_mm_per_px)
    svg = json_to_svg(plan)
    return {"plan": plan, "svg": svg}


@app.post("/calibrate-scale")
async def calibrate_scale(payload: dict):
    plan = payload["plan"]
    p1 = payload["p1"]
    p2 = payload["p2"]
    real_length_mm = float(payload["real_length_mm"])
    mm_per_px = calc_mm_per_px(p1, p2, real_length_mm)
    updated = apply_scale(plan, mm_per_px)
    return {"mm_per_px": mm_per_px, "plan": updated}


@app.post("/normalize-plan")
def normalize_plan(payload: dict):
    plan = payload["plan"]
    p = payload.get("params", {})
    cleaned = normalize(
        plan,
        snap_px=float(p.get("snap_px", 6)),
        angle_snap_deg=float(p.get("angle_snap_deg", 7)),
        min_len_px=float(p.get("min_len_px", 12)),
        min_room_area_m2=float(p.get("min_room_area_m2", 1.0)),
    )
    svg = json_to_svg(cleaned)
    return {"plan": cleaned, "svg": svg}


@app.post("/plan-to-svg")
def plan_to_svg(payload: dict):
    svg = json_to_svg(payload["plan"])
    return Response(content=svg, media_type="image/svg+xml")
