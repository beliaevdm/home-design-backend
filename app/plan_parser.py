# app/plan_parser.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
from shapely.geometry import LineString
from shapely.ops import unary_union


def _png_bytes_to_gray(png_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(png_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("parse_plan_image: не смог открыть PNG")
    return img


def _detect_lines(gray_bin: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    На входе — бинарная картинка: 255 фон, 0 линии.
    Возвращает список отрезков (x1,y1,x2,y2) из Probabilistic Hough.
    """
    # работаем по краям
    edges = cv2.Canny(255 - gray_bin, 50, 120, apertureSize=3, L2gradient=True)

    H, W = gray_bin.shape
    min_len = max(30, int(min(H, W) * 0.05))     # минимум 5% от меньшей стороны
    max_gap = int(min(H, W) * 0.01)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=60,
        minLineLength=min_len,
        maxLineGap=max_gap,
    )
    if lines is None:
        return []
    return [tuple(map(int, l[0])) for l in lines]


def _merge_collinear(segments: List[Tuple[int, int, int, int]], tol_px: int = 6, ang_tol_deg: float = 3) -> List[Tuple[int, int, int, int]]:
    """
    Грубое объединение коллинеарных отрезков в более длинные стены.
    """
    if not segments:
        return []

    # нормализуем направление (слева-направо / сверху-вниз)
    norm = []
    for x1, y1, x2, y2 in segments:
        if (x2, y2) < (x1, y1):
            x1, y1, x2, y2 = x2, y2, x1, y1
        norm.append((x1, y1, x2, y2))

    used = [False] * len(norm)
    merged: List[Tuple[int, int, int, int]] = []

    def angle_deg(a, b, c, d):
        ang = np.degrees(np.arctan2(d - b, c - a))
        # приводим к { ~0, ~90 }
        if ang < 0:
            ang += 180
        if 45 < ang < 135:
            ang = 90.0
        else:
            ang = 0.0
        return ang

    for i, s in enumerate(norm):
        if used[i]:
            continue
        x1, y1, x2, y2 = s
        base_ang = angle_deg(x1, y1, x2, y2)
        line = LineString([(x1, y1), (x2, y2)])

        group = [i]
        for j in range(i + 1, len(norm)):
            if used[j]:
                continue
            u1, v1, u2, v2 = norm[j]
            ang = angle_deg(u1, v1, u2, v2)
            if abs(ang - base_ang) > ang_tol_deg:
                continue

            # расстояние между линиями
            other = LineString([(u1, v1), (u2, v2)])
            if line.distance(other) <= tol_px:
                group.append(j)
                line = unary_union([line, other]).envelope.boundary  # расширяем

        # финальная оболочка → берём bbox и вытягиваем вдоль базового направления
        xs = []
        ys = []
        for idx in group:
            used[idx] = True
            a, b, c, d = norm[idx]
            xs += [a, c]
            ys += [b, d]
        xmin, xmax = int(min(xs)), int(max(xs))
        ymin, ymax = int(min(ys)), int(max(ys))

        if base_ang == 0.0:   # горизонталь
            y = int(round(np.median(ys)))
            merged.append((xmin, y, xmax, y))
        else:                 # вертикаль
            x = int(round(np.median(xs)))
            merged.append((x, ymin, x, ymax))

    # удалим очень короткие
    out = []
    for x1, y1, x2, y2 in merged:
        if np.hypot(x2 - x1, y2 - y1) >= 25:
            out.append((x1, y1, x2, y2))
    return out


def parse_plan_image(prepared_png_bytes: bytes, known_scale_mm_per_px: float | None = None) -> Dict[str, Any]:
    """
    На входе — PNG (после препроцесса).
    Выдаём plan JSON с пучком «стен».
    """
    gray = _png_bytes_to_gray(prepared_png_bytes)  # 255 фон, 0 линии
    H, W = gray.shape

    raw_segments = _detect_lines(gray)
    walls = _merge_collinear(raw_segments)

    plan = {
        "image_size": {"w_px": int(W), "h_px": int(H)},
        "scale": {"mm_per_px": float(known_scale_mm_per_px) if known_scale_mm_per_px else 1.0},
        "walls": [{"p1": [int(x1), int(y1)], "p2": [int(x2), int(y2)]} for (x1, y1, x2, y2) in walls],
        "rooms": [],
    }
    return plan
