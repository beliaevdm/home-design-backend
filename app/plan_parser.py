import cv2, numpy as np
from shapely.geometry import LineString
from shapely.ops import polygonize
from PIL import Image
import io

def _bytes_to_gray(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        pil = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = np.array(pil)
    return img

def parse_plan_image(image_bytes: bytes, known_scale_mm_per_px: float | None = None) -> dict:
    img = _bytes_to_gray(image_bytes)
    h, w = img.shape[:2]

    # 1) контуры
    edges = cv2.Canny(cv2.GaussianBlur(img, (3,3), 0), 50, 150, apertureSize=3)

    # 2) линии (стены)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi/180, threshold=70,
        minLineLength=int(min(w, h)*0.05), maxLineGap=5
    )
    wall_lines = []
    if lines is not None:
        for (x1,y1,x2,y2) in lines[:,0,:]:
            wall_lines.append(((int(x1),int(y1)), (int(x2),int(y2))))

    # 3) полигоны (примерные комнаты)
    shapely_lines = [LineString([p1, p2]) for p1, p2 in wall_lines]
    rooms = []
    if shapely_lines:
        for poly in polygonize(shapely_lines):
            if poly.area > (w*h)*0.002:
                coords = [(int(x), int(y)) for x, y in list(poly.exterior.coords)[:-1]]
                rooms.append(coords)

    mm_per_px = known_scale_mm_per_px or 1.0

    return {
        "image_size": {"w_px": int(w), "h_px": int(h)},
        "scale": {"mm_per_px": float(mm_per_px)},
        "walls": [{"p1": [p1[0], p1[1]], "p2": [p2[0], p2[1]]} for p1, p2 in wall_lines],
        "rooms": [{"id": f"room_{i+1}", "polygon": poly} for i, poly in enumerate(rooms)]
    }
