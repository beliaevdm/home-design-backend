# app/plan_parser.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2


def _dedup_segments(segments: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """
    Грубое удаление дублей: квантование координат.
    """
    seen = set()
    out = []
    for x1, y1, x2, y2 in segments:
        key = (int(round(x1 / 2)), int(round(y1 / 2)), int(round(x2 / 2)), int(round(y2 / 2)))
        if key not in seen:
            seen.add(key)
            out.append((x1, y1, x2, y2))
    return out


def parse_plan_image(mask_or_bytes: np.ndarray, known_scale_mm_per_px: Optional[float] = None) -> Dict[str, Any]:
    """
    Принимает бинарную маску (2D np.uint8): 0 – линии, 255 – фон.
    Возвращает {image_size, scale, walls, rooms[]}.
    """
    if mask_or_bytes.ndim != 2:
        raise ValueError("Ожидалась бинарная маска 2D (0 – линии, 255 – фон)")

    mask = mask_or_bytes
    H, W = mask.shape
    edges = cv2.Canny(255 - mask, 50, 150)

    # параметры Hough зависят от размера
    min_len = max(40, min(W, H) // 25)
    max_gap = max(5, min(W, H) // 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=min_len,
        maxLineGap=max_gap,
    )

    segs: List[Tuple[int, int, int, int]] = []
    if lines is not None:
        for l in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, l.tolist())
            # фильтр: почти горизонтальные/вертикальные
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if dx < 2 and dy < 2:
                continue
            if dx >= dy and dy > 0:
                slope = dy / float(dx)
                if slope > 0.1:
                    continue
            if dy > dx and dx > 0:
                slope = dx / float(dy)
                if slope > 0.1:
                    continue
            segs.append((x1, y1, x2, y2))

    segs = _dedup_segments(segs)

    walls = [{"p1": [x1, y1], "p2": [x2, y2]} for (x1, y1, x2, y2) in segs]

    plan: Dict[str, Any] = {
        "image_size": {"w_px": int(W), "h_px": int(H)},
        "scale": {"mm_per_px": float(known_scale_mm_per_px) if known_scale_mm_per_px else None},
        "walls": walls,
        "rooms": [],
    }
    return plan
