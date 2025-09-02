# app/preprocess.py
from __future__ import annotations
import io
from typing import Optional, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image


# ----------------------------
# чтение PDF/изображений
# ----------------------------
def _is_pdf(data: bytes, filename: Optional[str]) -> bool:
    if filename and filename.lower().endswith(".pdf"):
        return True
    return data.startswith(b"%PDF")


def read_image_or_pdf(data: bytes, filename: Optional[str] = None, dpi: int = 300) -> np.ndarray:
    """
    Возвращает BGR-изображение (np.ndarray, uint8).
    Если PDF — рендерим 1-ю страницу в заданном DPI.
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
        raise ValueError("Не удалось прочитать изображение")
    return img


# ----------------------------
# утилиты препроцесса
# ----------------------------
def _deskew(gray: np.ndarray) -> np.ndarray:
    """Автоповорот по доминирующим линиям (HoughLines)."""
    edges = cv2.Canny(gray, 60, 160, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, threshold=int(0.004 * gray.size))
    if lines is None:
        return gray

    angles = []
    for rho_theta in lines[:500]:
        rho, theta = rho_theta[0]
        ang = (theta * 180.0 / np.pi) % 180.0
        if ang > 90:
            ang -= 180
        angles.append(ang)

    if not angles:
        return gray

    angle = float(np.median(angles))
    if abs(angle) < 0.3:
        return gray

    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _binarize_for_lines(gray: np.ndarray) -> np.ndarray:
    """Бинаризация под линии: 255 фон, 0 линии."""
    gray = cv2.convertScaleAbs(gray, alpha=1.25, beta=0)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 9)
    inv = 255 - bw
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1)))
    vert  = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35)))
    lines = cv2.max(horiz, vert)
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8), iterations=1)
    lines = cv2.erode(lines, np.ones((3, 3), np.uint8), iterations=1)
    return 255 - lines  # фон белый, линии чёрные


def _crop_plan_bbox_from_gray(gray: np.ndarray) -> Tuple[int, int, int, int]:
    """Находим bbox области плана (по маске линий). Возвращает (x0,y0,x1,y1)."""
    bw_lines = _binarize_for_lines(gray)
    inv = 255 - bw_lines
    closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)),
                              iterations=1)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h, w = gray.shape
        return 0, 0, w, h

    h, w = gray.shape
    best = max(cnts, key=cv2.contourArea)
    x, y, ww, hh = cv2.boundingRect(best)
    pad = max(5, int(0.01 * max(w, h)))
    x0 = max(0, x + pad)
    y0 = max(0, y + pad)
    x1 = min(w, x + ww - pad)
    y1 = min(h, y + hh - pad)
    if x1 - x0 < 50 or y1 - y0 < 50:
        return 0, 0, w, h
    return x0, y0, x1, y1


def auto_crop_plan(img_bgr: np.ndarray) -> np.ndarray:
    """
    Кропим исходное BGR-изображение до области плана (без шапок/штампов).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = _deskew(gray)
    x0, y0, x1, y1 = _crop_plan_bbox_from_gray(gray)
    return img_bgr[y0:y1, x0:x1]


def prepare_bytes_for_parser(data: bytes, filename: Optional[str]) -> Tuple[bytes, Tuple[int, int]]:
    """
    Полный препроцесс для «классического» парсера:
      - PDF/JPG → BGR
      - GRAY → deskew
      - бинаризация/усиление осевых линий
      - кроп по области плана
      - отдаём PNG-байты и размер (w, h)
    """
    bgr = read_image_or_pdf(data, filename)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = _deskew(gray)

    # bbox по серому (устойчиво) → применяем к бинарке
    x0, y0, x1, y1 = _crop_plan_bbox_from_gray(gray)
    bw_lines = _binarize_for_lines(gray)[y0:y1, x0:x1]

    h, w = bw_lines.shape
    pil = Image.fromarray(bw_lines)  # L
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue(), (w, h)
