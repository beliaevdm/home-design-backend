# app/preprocess.py
from __future__ import annotations
import io
import os
from typing import Tuple, Optional

import numpy as np
import cv2
from PIL import Image
import fitz  # PyMuPDF


def _is_pdf(data: bytes, filename: Optional[str]) -> bool:
    if filename and filename.lower().endswith(".pdf"):
        return True
    return data.startswith(b"%PDF")


def read_image_or_pdf(data: bytes, filename: Optional[str] = None, dpi: int = 300) -> np.ndarray:
    """
    Возвращает BGR изображение. Для PDF рендерим 1-ю страницу с указанным DPI.
    """
    if _is_pdf(data, filename):
        doc = fitz.open(stream=data, filetype="pdf")
        page = doc[0]
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("read_image_or_pdf: cannot decode")
    return img


def _four_point_warp(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    # упорядочиваем вершины
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    widthA = np.hypot(*(br - bl))
    widthB = np.hypot(*(tr - tl))
    heightA = np.hypot(*(tr - br))
    heightB = np.hypot(*(tl - bl))
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (maxW, maxH), flags=cv2.INTER_CUBIC)


def auto_crop_plan(img_bgr: np.ndarray) -> np.ndarray:
    """
    Ищем большую прямоугольную область чертежа и выравниваем. Если не нашли — bbox по контенту.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    H, W = gray.shape
    best, best_area = None, 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if 0.2 * W * H < area < 0.98 * W * H and area > best_area:
                best = approx.reshape(-1, 2)
                best_area = area

    if best is not None:
        return _four_point_warp(img_bgr, best.astype("float32"))

    # fallback: content bbox
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    nz = cv2.findNonZero(thr)
    if nz is None:
        return img_bgr
    x, y, w, h = cv2.boundingRect(nz)
    pad = int(max(W, H) * 0.01)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    return img_bgr[y0:y1, x0:x1]


def _remove_colored_stamps(img_bgr: np.ndarray) -> np.ndarray:
    """
    Съедаем цветные печати/подписи (оставляем белым).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[..., 1] > 60) & (hsv[..., 2] > 80)
    out = img_bgr.copy()
    out[mask] = (255, 255, 255)
    return out


def binarize_for_lines(img_bgr: np.ndarray) -> np.ndarray:
    """
    Бинаризация с усилением длинных горизонталей/вертикалей.
    Возвращаем: фон=255, линии=0.
    """
    img_bgr = _remove_colored_stamps(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # выравнивание фона
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    norm = cv2.divide(gray, blur, scale=255)

    bw = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 51, 10)
    inv = 255 - bw

    H, W = inv.shape
    S = max(H, W)

    # убираем мелочь (текст), порог по площади
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((inv > 0).astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(inv)
    area_min = max(80, int(0.00005 * (H * W)))
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= area_min:
            cleaned[labels == i] = 255

    # ядра зависят от размера картинки
    k_long = max(15, int(0.02 * S))   # длина для вытягивания линий
    h = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (k_long, 1)))
    v = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_long)))
    lines = cv2.max(h, v)

    # чуть залечим зазоры
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    out = 255 - lines

    # debug снимки (по желанию)
    if os.getenv("DEBUG_PREPROC") == "1":
        os.makedirs("/tmp/plan_debug", exist_ok=True)
        cv2.imwrite("/tmp/plan_debug/01_gray.png", gray)
        cv2.imwrite("/tmp/plan_debug/02_norm.png", norm)
        cv2.imwrite("/tmp/plan_debug/03_inv.png", inv)
        cv2.imwrite("/tmp/plan_debug/04_cleaned.png", cleaned)
        cv2.imwrite("/tmp/plan_debug/05_lines.png", lines)
        cv2.imwrite("/tmp/plan_debug/06_out.png", out)

    return out


def prepare_bytes_for_parser(data: bytes, filename: Optional[str]) -> Tuple[bytes, Tuple[int, int]]:
    img = read_image_or_pdf(data, filename)
    img = auto_crop_plan(img)
    bin_img = binarize_for_lines(img)
    pil = Image.fromarray(bin_img)  # L
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue(), (bin_img.shape[1], bin_img.shape[0])
