import math

def calc_mm_per_px(p1, p2, real_length_mm: float) -> float:
    x1, y1 = p1; x2, y2 = p2
    dpx = math.hypot(x2 - x1, y2 - y1)
    if dpx == 0:
        raise ValueError("Точки совпадают")
    return float(real_length_mm) / dpx

def apply_scale(plan: dict, mm_per_px: float) -> dict:
    plan["scale"] = {"mm_per_px": float(mm_per_px)}
    return plan
