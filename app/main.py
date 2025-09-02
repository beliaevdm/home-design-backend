# app/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response

from app.preprocess import prepare_bytes_for_parser
from app.plan_parser import parse_plan_image
from app.vectorize import json_to_svg

app = FastAPI(title="Home Design Backend", version="0.1")

@app.get("/")
def root():
    return {"ok": True}

# Единственная точка входа: принимает PDF/JPG/PNG и возвращает SVG
@app.post("/render")
async def render(file: UploadFile = File(...)):
    # 1) читаем файл и прогоняем через препроцесс (автокадрирование + бинеризация)
    raw = await file.read()
    prepared_png, (w, h) = prepare_bytes_for_parser(raw, file.filename)

    # 2) парсинг линий в JSON-план
    plan = parse_plan_image(prepared_png)

    # на всякий случай проставим размеры, если парсер их не заполнил
    plan.setdefault("image_size", {"w_px": w, "h_px": h})

    # 3) генерация SVG
    svg = json_to_svg(plan)
    return Response(content=svg, media_type="image/svg+xml")
