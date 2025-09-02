# app/main.py
from __future__ import annotations
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response

from app.preprocess import prepare_for_parse
from app.plan_parser import parse_plan_image
from app.vectorize import json_to_svg

app = FastAPI(title="Home Design Backend", version="0.2")


@app.get("/")
def root():
    return {"ok": True}


@app.post("/render")
async def render(file: UploadFile = File(...)) -> Response:
    """
    Единственный роут.
    Принимает PDF/JPG/PNG — возвращает SVG с линиями плана.
    Никаких дополнительных данных от пользователя не требуется.
    """
    raw = await file.read()

    # 1) препроцесс: чистка печатей, выравнивание, авто-кроп, бинарная маска линий
    mask, (w, h) = prepare_for_parse(raw, file.filename)

    # 2) векторизация стен (Hough + фильтры)
    plan = parse_plan_image(mask)

    # гарантия размеров (на всякий случай)
    plan["image_size"] = {"w_px": w, "h_px": h}

    # 3) SVG
    svg = json_to_svg(plan)
    return Response(content=svg, media_type="image/svg+xml")
