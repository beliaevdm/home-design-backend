from __future__ import annotations
import math
from typing import Dict, List, Tuple
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union, linemerge, snap, polygonize

Point = Tuple[float, float]

def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _snap_pt(pt: Point, step: float) -> Point:
    return (round(pt[0] / step) * step, round(pt[1] / step) * step)

def _angle_snap(p1: Point, p2: Point, deg: float) -> Tuple[Point, Point]:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return p1, p2
    ang = math.degrees(math.atan2(dy, dx))
    # ближайшее к 0/90/180/270
    cands = [0, 90, 180, -90]
    best = min(cands, key=lambda a: min(abs(ang - a), abs(ang - (a + 360))))
    err = min(abs(ang - best), abs(ang - (best + 360)))
    if err <= deg:
        if abs(best) in (0, 180):          # горизонталь
            return p1, (p2[0], p1[1])
        else:                               # вертикаль
            return p1, (p1[0], p2[1])
    return p1, p2

def normalize(plan: Dict,
              snap_px: float = 6.0,
              angle_snap_deg: float = 7.0,
              min_len_px: float = 12.0,
              min_room_area_m2: float = 1.0) -> Dict:
    mm_per_px = float(plan.get("scale", {}).get("mm_per_px", 1.0))

    # 1) снап концов к сетке + фильтр коротышей
    raw = []
    for w in plan.get("walls", []):
        p1 = tuple(w["p1"]); p2 = tuple(w["p2"])
        a = _snap_pt(p1, snap_px); b = _snap_pt(p2, snap_px)
        if a != b and _dist(a, b) >= min_len_px:
            a, b = _angle_snap(a, b, angle_snap_deg)
            if _dist(a, b) >= min_len_px:
                raw.append(LineString([a, b]))
    if not raw:
        plan.setdefault("rooms", [])
        return plan

    # 2) слить/подтянуть и линемёрдж
    mls = MultiLineString(raw)
    merged = linemerge(unary_union(snap(mls, mls, snap_px)))

    lines = []
    if merged.geom_type == "LineString":
        lines = [merged]
    else:
        lines = list(merged.geoms)

    # 3) собрать стены обратно
    new_walls = []
    for ln in lines:
        c0 = tuple(ln.coords[0]); c1 = tuple(ln.coords[-1])
        if _dist(c0, c1) >= min_len_px:
            new_walls.append({"p1": [round(c0[0]), round(c0[1])],
                              "p2": [round(c1[0]), round(c1[1])]})

    # 4) построить помещения
    # порог по площади в px^2
    # м^2 -> мм^2 : *1e6; затем / (mm_per_px^2) -> px^2
    min_area_px2 = (min_room_area_m2 * 1_000_000.0) / (mm_per_px ** 2)
    faces = list(polygonize(unary_union(lines)))

    rooms = []
    for poly in faces:
        if not poly.is_valid:
            continue
        if poly.area < min_area_px2:
            continue
        # внешняя граница (без повторения последней точки)
        pts = [(round(x), round(y)) for x, y in list(poly.exterior.coords)[:-1]]
        area_m2 = round(poly.area * (mm_per_px ** 2) / 1_000_000.0, 2)
        rooms.append({"polygon": pts, "area_m2": area_m2})

    plan["walls"] = new_walls
    plan["rooms"] = rooms
    return plan