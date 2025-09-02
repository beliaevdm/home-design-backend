from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Iterable
import math


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def expand(self, dx: float, dy: float) -> "Box":
        return Box(self.x1 - dx, self.y1 - dy, self.x2 + dx, self.y2 + dy)


def _to_box(b: Iterable[float]) -> Box:
    x1, y1, x2, y2 = map(float, b)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return Box(x1, y1, x2, y2)


def _merge_1d(segments: List[Tuple[float, float]], gap: float) -> List[Tuple[float, float]]:
    """Склеить отрезки на прямой, если они перекрываются или ближе gap."""
    if not segments:
        return []
    segments = sorted(segments)
    out: List[Tuple[float, float]] = []
    cur_l, cur_r = segments[0]
    for l, r in segments[1:]:
        if l <= cur_r + gap:
            cur_r = max(cur_r, r)
        else:
            out.append((cur_l, cur_r))
            cur_l, cur_r = l, r
    out.append((cur_l, cur_r))
    return out


def _group_by_band(vals: List[float], band: float) -> List[List[int]]:
    """Группировка индексов по близости (для y горизонталей / x вертикалей)."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    groups: List[List[int]] = []
    for i in order:
        v = vals[i]
        if not groups:
            groups.append([i])
            continue
        last_grp = groups[-1]
        if abs(vals[last_grp[-1]] - v) <= band:
            last_grp.append(i)
        else:
            groups.append([i])
    return groups


def _wall_boxes_to_segments(walls: List[Box],
                            W: int,
                            H: int,
                            band_px: float,
                            gap_px: float,
                            min_len_px: float) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Преобразуем коробки стен в набор длинных отрезков (гориз/верт).
    1) Разделяем на горизонтальные/вертикальные по аспекту
    2) Группируем по «полосам» (у горизонталей — по y; у вертикалей — по x)
    3) В каждой полосе склеиваем пересекающиеся сегменты
    """
    horiz_idx, vert_idx = [], []
    for i, b in enumerate(walls):
        if b.w >= b.h * 1.7:  # горизонтальные
            horiz_idx.append(i)
        elif b.h >= b.w * 1.7:  # вертикальные
            vert_idx.append(i)
        else:
            # почти квадратные: отнесём по большей стороне
            (horiz_idx if b.w >= b.h else vert_idx).append(i)

    segs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

    # Горизонтальные
    if horiz_idx:
        ys = [walls[i].cy for i in horiz_idx]
        groups = _group_by_band(ys, band_px)
        for g in groups:
            y_avg = int(round(sum(ys[j] for j in g) / len(g)))
            parts = [(walls[horiz_idx[j]].x1, walls[horiz_idx[j]].x2) for j in g]
            merged = _merge_1d(parts, gap_px)
            for l, r in merged:
                if r - l >= min_len_px:
                    segs.append(((int(l), y_avg), (int(r), y_avg)))

    # Вертикальные
    if vert_idx:
        xs = [walls[i].cx for i in vert_idx]
        groups = _group_by_band(xs, band_px)
        for g in groups:
            x_avg = int(round(sum(xs[j] for j in g) / len(g)))
            parts = [(walls[vert_idx[j]].y1, walls[vert_idx[j]].y2) for j in g]
            merged = _merge_1d(parts, gap_px)
            for t, b in merged:
                if b - t >= min_len_px:
                    segs.append(((x_avg, int(t)), (x_avg, int(b))))

    # Обрезка в пределах изображения
    clipped: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for (x1, y1), (x2, y2) in segs:
        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H - 1, y2))
        clipped.append(((x1, y1), (x2, y2)))

    return clipped


def _windows_to_segments(windows: List[Box],
                         wall_segs: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                         band_px: float,
                         min_len_px: float) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Преобразуем окна в короткие отрезки и «прилепляем» их к ближайшей стене той же ориентации.
    """
    out: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    # Подготовим полосы стен
    horiz_bands: List[Tuple[int, int, int]] = []  # y, x1, x2
    vert_bands: List[Tuple[int, int, int]] = []   # x, y1, y2
    for (x1, y1), (x2, y2) in wall_segs:
        if y1 == y2:
            y = y1
            horiz_bands.append((y, min(x1, x2), max(x1, x2)))
        elif x1 == x2:
            x = x1
            vert_bands.append((x, min(y1, y2), max(y1, y2)))

    for b in windows:
        if b.w >= b.h:  # горизонтально ориентированное окно
            y0 = b.cy
            x1, x2 = int(b.x1), int(b.x2)
            # Найти ближайшую горизонтальную стену по y
            best = None
            best_d = 1e9
            for y, a, c in horiz_bands:
                if x2 < a or x1 > c:  # не пересекается по X — пропускаем
                    continue
                d = abs(y - y0)
                if d < best_d:
                    best_d = d
                    best = (max(a, x1), y, min(c, x2))
            if best and best[2] - best[0] >= min_len_px and best_d <= band_px:
                out.append(((best[0], best[1]), (best[2], best[1])))
        else:           # вертикально ориентированное окно
            x0 = b.cx
            y1, y2 = int(b.y1), int(b.y2)
            best = None
            best_d = 1e9
            for x, a, c in vert_bands:
                if y2 < a or y1 > c:
                    continue
                d = abs(x - x0)
                if d < best_d:
                    best_d = d
                    best = (x, max(a, y1), min(c, y2))
            if best and best[2] - best[1] >= min_len_px and best_d <= band_px:
                out.append(((best[0], best[1]), (best[0], best[2])))

    return out


def detections_to_plan(det_json: Dict[str, Any],
                       min_conf: float = 0.5) -> Dict[str, Any]:
    """
    Принимает JSON из /detect-json и возвращает черновой план:
    {
      "image_size": { "w_px":..., "h_px":... },
      "walls": [ {"p1":[x,y], "p2":[x,y]}, ... ],
      "windows": [ {"p1":[x,y], "p2":[x,y]}, ... ]
    }
    """
    W = int(det_json["image_size"]["w_px"])
    H = int(det_json["image_size"]["h_px"])
    dets = det_json.get("detections", [])

    # Параметры (подобраны под планы БТІ; можно подкрутить)
    base = max(W, H)
    band_px = max(8.0, 0.008 * base)      # «полоса» выравнивания
    gap_px = max(10.0, 0.012 * base)      # зазор, в котором склеиваем отрезки
    min_len_px = max(30.0, 0.03 * base)   # минимальная длина результата

    # Собираем боксы по классам
    walls: List[Box] = []
    windows: List[Box] = []
    for d in dets:
        if float(d.get("conf", 0.0)) < float(min_conf):
            continue
        cls_ = str(d.get("class", "")).strip().lower()
        box = _to_box(d.get("bbox", [0, 0, 0, 0]))
        if cls_ == "wall":
            walls.append(box)
        elif cls_ == "window":
            windows.append(box)

    # 1) стены → сегменты
    wall_segs = _wall_boxes_to_segments(walls, W, H, band_px, gap_px, min_len_px)

    # 2) окна → сегменты, привязанные к ближайшей стене
    window_segs = _windows_to_segments(windows, wall_segs, band_px, min_len_px * 0.5)

    plan: Dict[str, Any] = {
        "image_size": {"w_px": W, "h_px": H},
        "scale": {},
        "walls": [{"p1": [x1, y1], "p2": [x2, y2]} for (x1, y1), (x2, y2) in wall_segs],
        "windows": [{"p1": [x1, y1], "p2": [x2, y2]} for (x1, y1), (x2, y2) in window_segs],
        "rooms": []
    }
    return plan
