def _points_attr(points):
    return " ".join(f"{x},{y}" for x, y in points)

def json_to_svg(plan: dict) -> str:
    w = plan["image_size"]["w_px"]; h = plan["image_size"]["h_px"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']

    # комнаты (полупрозрачная заливка, контур синим)
    for room in plan.get("rooms", []):
        pts = " ".join(f"{x},{y}" for x, y in room.get("polygon", []))
        parts.append(f'<polygon points="{pts}" fill="rgba(0,128,255,0.15)" stroke="blue" stroke-width="2"/>')

    # стены
    for wline in plan.get("walls", []):
        x1, y1 = wline["p1"]; x2, y2 = wline["p2"]
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="3"/>')

    parts.append("</svg>")
    return "\n".join(parts)

def json_to_svg(plan: dict, overlays: dict | None = None) -> str:
    w = plan["image_size"]["w_px"]
    h = plan["image_size"]["h_px"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']

    # стены
    for wline in plan.get("walls", []):
        (x1, y1) = wline["p1"]; (x2, y2) = wline["p2"]
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="3"/>')

    # комнаты
    for room in plan.get("rooms", []):
        parts.append(
            f'<polygon points="{_points_attr(room["polygon"])}" '
            f'stroke="gray" stroke-width="1" fill="lightgray" fill-opacity="0.15"/>'
        )

    # оверлеи (электрика)
    if overlays and "electrical" in overlays:
        for item in overlays["electrical"]:
            x, y = item["position_px"]
            color = "red" if item["type"] == "socket" else "blue"
            parts.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{color}"/>')

    parts.append("</svg>")
    return "\n".join(parts)
