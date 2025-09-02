# app/plan_parser.py
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import cv2


def _decode_png_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("parse_plan_image: cannot decode image bytes")
    return img


def _merge_segments_horiz(segs: List[Tuple[int,int,int,int]],
                          y_tol: int, gap: int) -> List[Tuple[int,int,int,int]]:
    if not segs:
        return []
    buckets: Dict[int, List[Tuple[int,int,int,int]]] = {}
    for x1,y1,x2,y2 in segs:
        y = int(round((y1 + y2) / 2))
        key = int(round(y / max(1, y_tol)))
        buckets.setdefault(key, []).append((x1,y1,x2,y2))

    merged: List[Tuple[int,int,int,int]] = []
    for _, group in buckets.items():
        group = sorted(group, key=lambda s: min(s[0], s[2]))
        cur_y = int(round((group[0][1] + group[0][3]) / 2))
        cur_x1 = min(group[0][0], group[0][2])
        cur_x2 = max(group[0][0], group[0][2])
        for x1,y1,x2,y2 in group[1:]:
            x1_, x2_ = sorted((x1, x2))
            y_ = int(round((y1 + y2) / 2))
            if abs(y_ - cur_y) <= y_tol and x1_ <= cur_x2 + gap:
                cur_x2 = max(cur_x2, x2_)
                cur_y = int(round((cur_y + y_) / 2))
            else:
                merged.append((cur_x1, cur_y, cur_x2, cur_y))
                cur_y = y_
                cur_x1, cur_x2 = x1_, x2_
        merged.append((cur_x1, cur_y, cur_x2, cur_y))
    return merged


def _merge_segments_vert(segs: List[Tuple[int,int,int,int]],
                          x_tol: int, gap: int) -> List[Tuple[int,int,int,int]]:
    if not segs:
        return []
    buckets: Dict[int, List[Tuple[int,int,int,int]]] = {}
    for x1,y1,x2,y2 in segs:
        x = int(round((x1 + x2) / 2))
        key = int(round(x / max(1, x_tol)))
        buckets.setdefault(key, []).append((x1,y1,x2,y2))

    merged: List[Tuple[int,int,int,int]] = []
    for _, group in buckets.items():
        group = sorted(group, key=lambda s: min(s[1], s[3]))
        cur_x = int(round((group[0][0] + group[0][2]) / 2))
        cur_y1 = min(group[0][1], group[0][3])
        cur_y2 = max(group[0][1], group[0][3])
        for x1,y1,x2,y2 in group[1:]:
            y1_, y2_ = sorted((y1, y2))
            x_ = int(round((x1 + x2) / 2))
            if abs(x_ - cur_x) <= x_tol and y1_ <= cur_y2 + gap:
                cur_y2 = max(cur_y2, y2_)
                cur_x = int(round((cur_x + x_) / 2))
            else:
                merged.append((cur_x, cur_y1, cur_x, cur_y2))
                cur_x = x_
                cur_y1, cur_y2 = y1_, y2_
        merged.append((cur_x, cur_y1, cur_x, cur_y2))
    return merged


def parse_plan_image(png_bytes: bytes,
                     known_scale_mm_per_px: Optional[float] = None) -> Dict[str, Any]:
    """
    Ждём бинарное PNG (фон=255, линии=0). Ищем длинные H/V-отрезки (HoughLinesP),
    склеиваем коллинеарные сегменты, отбрасываем коротышей.
    """
    img = _decode_png_bytes(png_bytes)
    H, W = img.shape[:2]
    S = max(H, W)

    # edge-карта для Хафа
    edges = cv2.Canny(255 - img, 30, 90)

    # относительные параметры
    min_len = max(20, int(0.06 * S))       # минимальная длина линии
    max_gap = max(6,  int(0.02 * S))       # допустимый разрыв
    thr     = 80                           # аккумулятор Хафа

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=thr,
                            minLineLength=min_len, maxLineGap=max_gap)

    horizontals: List[Tuple[int,int,int,int]] = []
    verticals:   List[Tuple[int,int,int,int]] = []

    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if dx >= dy:  # горизонталь
                y = int(round((y1 + y2) / 2))
                horizontals.append((min(x1, x2), y, max(x1, x2), y))
            else:         # вертикаль
                x = int(round((x1 + x2) / 2))
                verticals.append((x, min(y1, y2), x, max(y1, y2)))

    # склейка (толерансы и зазор тоже относительные)
    y_tol = max(3, int(0.01 * S))
    x_tol = max(3, int(0.01 * S))
    horizontals = _merge_segments_horiz(horizontals, y_tol=y_tol, gap=max_gap)
    verticals   = _merge_segments_vert(verticals,   x_tol=x_tol, gap=max_gap)

    # финальный отбор
    min_keep = max(30, int(0.05 * S))
    walls: List[Dict[str, List[int]]] = []
    for x1, y, x2, _ in horizontals:
        if x2 - x1 >= min_keep:
            walls.append({"p1": [int(x1), int(y)], "p2": [int(x2), int(y)]})
    for x, y1, _, y2 in verticals:
        if y2 - y1 >= min_keep:
            walls.append({"p1": [int(x), int(y1)], "p2": [int(x), int(y2)]})

    plan: Dict[str, Any] = {
        "image_size": {"w_px": int(W), "h_px": int(H)},
        "scale": {"mm_per_px": float(known_scale_mm_per_px) if known_scale_mm_per_px else 1.0},
        "walls": walls,
        "rooms": []
    }
    return plan
