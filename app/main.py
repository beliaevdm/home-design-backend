from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response

from app.plan_parser import parse_plan_image
from app.vectorize import json_to_svg
from app.scale import calc_mm_per_px, apply_scale
from app.normalize import normalize

app = FastAPI(title="Home Design Backend", version="0.1")


@app.get("/")
def root():
    return {"ok": True}


@app.post("/parse-plan")
async def parse_plan(
    file: UploadFile = File(...),
    known_scale_mm_per_px: float | None = None,
):
    content = await file.read()
    plan = parse_plan_image(content, known_scale_mm_per_px=known_scale_mm_per_px)
    svg = json_to_svg(plan)
    return {"plan": plan, "svg": svg}


@app.post("/calibrate-scale")
async def calibrate_scale(payload: dict):
    """
    Accepts either of these shapes:

    1) {
         "plan": {...},
         "p1": [x1, y1],
         "p2": [x2, y2],
         "real_length_mm": 4200
       }

    2) {
         "plan": {...},
         "meter_line": {
           "p1": [x1, y1],
           "p2": [x2, y2],
           "real_length_mm": 4200
         }
       }
    """
    plan = payload.get("plan")
    if plan is None:
        raise HTTPException(status_code=422, detail="Missing 'plan' in body.")

    # allow both top-level fields and nested meter_line
    meter = payload.get("meter_line", payload)

    try:
        p1 = meter["p1"]
        p2 = meter["p2"]
        real_length_mm = float(meter["real_length_mm"])
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing key: {e.args[0]}")
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="'real_length_mm' must be a number.")

    mm_per_px = calc_mm_per_px(p1, p2, real_length_mm)
    updated = apply_scale(plan, mm_per_px)
    return {"mm_per_px": mm_per_px, "plan": updated}


@app.post("/normalize-plan")
def normalize_plan(payload: dict):
    """
    payload = {
      "plan": {...},               # plan from /calibrate-scale (with mm_per_px)
      "params": {                  # all optional
        "snap_px": 6,
        "angle_snap_deg": 7,
        "min_len_px": 12,
        "min_room_area_m2": 1.0
      }
    }
    """
    plan = payload.get("plan")
    if plan is None:
        raise HTTPException(status_code=422, detail="Missing 'plan' in body.")

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
    """
    Body: { "plan": {...} }
    Returns: raw SVG image
    """
    plan = payload.get("plan")
    if plan is None:
        raise HTTPException(status_code=422, detail="Body must be {'plan': {...}}")
    svg = json_to_svg(plan)
    return Response(content=svg, media_type="image/svg+xml")
