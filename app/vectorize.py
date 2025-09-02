# app/vectorize.py
from __future__ import annotations
from typing import Dict, Any, List


def json_to_svg(plan: Dict[str, Any]) -> str:
    w = int(plan["image_size"]["w_px"])
    h = int(plan["image_size"]["h_px"])

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')

    # стены
    for wll in plan.get("walls", []):
        x1, y1 = wll["p1"]
        x2, y2 = wll["p2"]
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="3"/>')

    # комнаты (если когда-то появятся контуры)
    for r in plan.get("rooms") or []:
        poly = r.get("contour", [])
        if len(poly) >= 3:
            pts = " ".join(f"{x},{y}" for x, y in poly)
            parts.append(f'<polygon points="{pts}" fill="rgba(0,120,255,0.12)" stroke="none"/>')

    parts.append("</svg>")
    return "\n".join(parts)
