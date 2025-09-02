from __future__ import annotations

from typing import Dict, Any, List, Tuple


def _line(x1: int, y1: int, x2: int, y2: int, color: str, width: int, dash: str = "") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{dash_attr} />'


def json_to_svg(plan: Dict[str, Any]) -> str:
    W = int(plan.get("image_size", {}).get("w_px", 800))
    H = int(plan.get("image_size", {}).get("h_px", 600))

    walls: List[Dict[str, List[int]]] = plan.get("walls", [])
    windows: List[Dict[str, List[int]]] = plan.get("windows", [])
    rooms: List[Dict[str, Any]] = plan.get("rooms", [])

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="white"/>')

    # стены — чёрные жирные линии
    for w in walls:
        (x1, y1) = w["p1"]; (x2, y2) = w["p2"]
        parts.append(_line(x1, y1, x2, y2, "#000000", 4))

    # окна — синим пунктиром
    for o in windows:
        (x1, y1) = o["p1"]; (x2, y2) = o["p2"]
        parts.append(_line(x1, y1, x2, y2, "#0A84FF", 3, dash="6 4"))

    # контуры комнат, если когда-нибудь появятся (сейчас пусто)
    for r in rooms:
        poly = r.get("contour", [])
        if len(poly) >= 3:
            pts = " ".join(f"{x},{y}" for x, y in poly)
            parts.append(f'<polygon points="{pts}" fill="rgba(0,120,255,0.12)" stroke="none"/>')

    parts.append("</svg>")
    return "\n".join(parts)
