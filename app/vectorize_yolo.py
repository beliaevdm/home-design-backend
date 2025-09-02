from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math

import numpy as np
import cv2

# ---------- геометрия ----------

def _length(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def _is_horizontal(p1: Tuple[float, float], p2: Tuple[float, float], tol_deg: float = 7.0) -> bool:
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    ang = abs(math.degrees(math.atan2(dy, dx)))
    return min(ang, 180-ang) <= tol_deg

def _is_vertical(p1: Tuple[float, float], p2: Tuple[float, float], tol_deg: float = 7.0) -> bool:
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    ang = abs(math.degrees(math.atan2(dy, dx)))
    return abs(90-ang) <= tol_deg

def _merge_axis_segments(
    segments: List[Tuple[Tuple[float,float], Tuple[float,float]]],
    axis: str,
    coord_tol: float,
    gap_tol: float,
) -> List[Tuple[Tuple[float,float], Tuple[float,float]]]:
    if not segments:
        return []

    groups: Dict[int, List[Tuple[Tuple[float,float], Tuple[float,float]]]] = {}

    def bucket_key(val: float) -> int:
        return int(round(val / coord_tol))

    for (p1, p2) in segments:
        if axis == 'h':
            y = (p1[1] + p2[1]) * 0.5
            key = bucket_key(y)
        else:
            x = (p1[0] + p2[0]) * 0.5
            key = bucket_key(x)
        groups.setdefault(key, []).append((p1, p2))

    result: List[Tuple[Tuple[float,float], Tuple[float,float]]] = []
    for _, segs in groups.items():
        if axis == 'h':
            norm = [((min(p1[0], p2[0]), (p1[1]+p2[1])*0.5),
                     (max(p1[0], p2[0]), (p1[1]+p2[1])*0.5)) for (p1, p2) in segs]
            norm.sort(key=lambda s: s[0][0])
            cur_x1, cur_y = norm[0][0][0], norm[0][0][1]
            cur_x2 = norm[0][1][0]
            for (a1, a2) in norm[1:]:
                x1, y1 = a1
                x2, _ = a2
                if x1 <= cur_x2 + gap_tol:
                    cur_x2 = max(cur_x2, x2)
                else:
                    result.append(((cur_x1, cur_y), (cur_x2, cur_y)))
                    cur_x1, cur_y, cur_x2 = x1, y1, x2
            result.append(((cur_x1, cur_y), (cur_x2, cur_y)))
        else:
            norm = [(((p1[0]+p2[0])*0.5, min(p1[1], p2[1])),
                     ((p1[0]+p2[0])*0.5, max(p1[1], p2[1]))) for (p1, p2) in segs]
            norm.sort(key=lambda s: s[0][1])
            cur_y1, cur_x = norm[0][0][1], norm[0][0][0]
            cur_y2 = norm[0][1][1]
            for (a1, a2) in norm[1:]:
                x1, y1 = a1
                _, y2 = a2
                if y1 <= cur_y2 + gap_tol:
                    cur_y2 = max(cur_y2, y2)
                else:
                    result.append(((cur_x, cur_y1), (cur_x, cur_y2)))
                    cur_y1, cur_x, cur_y2 = y1, x1, y2
            result.append(((cur_x, cur_y1), (cur_x, cur_y2)))
    return result

# ---------- маски/линии ----------

def _wall_mask_from_detections(W: int, H: int, detections: List[Dict[str, Any]], dilate_px: int = 8) -> np.ndarray:
    mask = np.zeros((H, W), np.uint8)
    for det in detections:
        if det.get("class") in ("Wall", "Column"):
            x1, y1, x2, y2 = det["bbox"]
            x1 = max(0, int(round(x1))); y1 = max(0, int(round(y1)))
            x2 = min(W-1, int(round(x2))); y2 = min(H-1, int(round(y2)))
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    if dilate_px > 0:
        kernel = np.ones((dilate_px, dilate_px), np.uint8)
        mask = cv2.dilate(mask, kernel, 1)
    return mask

def _lsd_segments(img_gray: np.ndarray) -> List[Tuple[Tuple[float,float], Tuple[float,float]]]:
    # совместимость с разными opencv: иногда нельзя передать _refine по имени
    try:
        lsd = cv2.createLineSegmentDetector(_refine=cv2.LSD_REFINE_STD)
    except TypeError:
        lsd = cv2.createLineSegmentDetector()
    lines, _, _, _ = lsd.detect(img_gray)
    segs: List[Tuple[Tuple[float,float], Tuple[float,float]]] = []
    if lines is not None:
        for L in lines:
            x1, y1, x2, y2 = L[0]
            segs.append(((float(x1), float(y1)), (float(x2), float(y2))))
    return segs

def _filter_segments_by_mask(
    segments: List[Tuple[Tuple[float,float], Tuple[float,float]]],
    mask: np.ndarray,
    min_len_px: float = 30.0,
    inside_ratio: float = 0.35,
) -> List[Tuple[Tuple[float,float], Tuple[float,float]]]:
    H, W = mask.shape
    keep: List[Tuple[Tuple[float,float], Tuple[float,float]]] = []
    for p1, p2 in segments:
        L = _length(p1, p2)
        if L < min_len_px:
            continue
        n = max(10, int(L / 8))
        xs = np.linspace(p1[0], p2[0], n)
        ys = np.linspace(p1[1], p2[1], n)
        hit = 0
        for x, y in zip(xs, ys):
            xi = int(round(x)); yi = int(round(y))
            if 0 <= xi < W and 0 <= yi < H and mask[yi, xi] > 0:
                hit += 1
        if hit / n >= inside_ratio:
            keep.append((p1, p2))
    return keep

def _split_by_orientation(segments):
    horiz, vert, other = [], [], []
    for p1, p2 in segments:
        if _is_horizontal(p1, p2):
            horiz.append((p1, p2))
        elif _is_vertical(p1, p2):
            vert.append((p1, p2))
        else:
            other.append((p1, p2))
    return horiz, vert, other

def _windows_from_detections(detections: List[Dict[str,Any]]) -> List[Tuple[int,int,int,int]]:
    out = []
    for det in detections:
        if det.get("class") == "Window":
            x1, y1, x2, y2 = det["bbox"]
            out.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))
    return out

# ---------- публичная функция ----------

def svg_from_image_and_yolo(img_bgr: np.ndarray, det_json: Dict[str, Any]) -> str:
    """
    BGR-изображение + JSON детекций → SVG (стены как линии, окна прямоугольниками).
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    dets = det_json["detections"]
    wall_mask = _wall_mask_from_detections(W, H, dets, dilate_px=8)

    segs_raw = _lsd_segments(gray)
    segs = _filter_segments_by_mask(segs_raw, wall_mask, min_len_px=28, inside_ratio=0.35)

    horiz, vert, _ = _split_by_orientation(segs)
    merged_h = _merge_axis_segments(horiz, axis='h', coord_tol=6.0, gap_tol=14.0)
    merged_v = _merge_axis_segments(vert,  axis='v', coord_tol=6.0, gap_tol=14.0)

    windows = _windows_from_detections(dets)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    for (p1, p2) in merged_h + merged_v:
        x1, y1 = p1; x2, y2 = p2
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="black" stroke-width="3"/>')
    for (x1, y1, x2, y2) in windows:
        parts.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="#2ecc71" stroke-width="3"/>')
    parts.append("</svg>")
    return "\n".join(parts)
