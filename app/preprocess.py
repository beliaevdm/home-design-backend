# app/preprocess.py
from __future__ import annotations
import io
from typing import Optional, Tuple

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
    Возвращает BGR изображение np.ndarray.
    Если это PDF — рендерим первую страницу с заданным DPI.
    """
    if _is_pdf(data, filename):
        doc = fitz.open(stream=data, filetype="pdf")
        page = doc[0]
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Не удалось прочитать изображение")
    return img


def _suppress_blue_stamps(img: np.ndarray) -> np.ndarray:
    """Съедаем синие/голубые печати: делаем их белыми, чтобы не мешали."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # диапазон синего/циан
    m1 = cv2.inRange(hsv, (90, 40, 40), (140, 255, 255))
    # размыть края маски
    m1 = cv2.dilate(m1, np.ones((5, 5), np.uint8), 1)
    out = img.copy()
    out[m1 > 0] = (255, 255, 255)
    return out


def _rotate_small_angle(img: np.ndarray) -> np.ndarray:
    """Выпрямляем небольшой наклон (±5°) по доминирующим линиям."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=180)
    if lines is None:
        return img

    angs = []
    for l in lines[:200]:
        theta = l[0, 1]
        # приводим к -90..90
        deg = (theta * 180.0 / np.pi) % 180.0
        if deg > 90:
            deg -= 180
        if abs(deg) < 15 or abs(abs(deg) - 90) < 15:
            if abs(deg) > 0.2:
                angs.append(deg)
    if not angs:
        return img

    angle = float(np.median(angs))
    if abs(angle) < 0.2 or abs(angle) > 5:
        return img

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def _content_crop(img: np.ndarray) -> np.ndarray:
    """
    Кадрируем по контенту: берём bbox по инвертированному бинарному изображению.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # лёгкая нормализация контраста
    gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - bw
    # убираем шум
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    nz = cv2.findNonZero(inv)
    if nz is None:
        return img
    x, y, w, h = cv2.boundingRect(nz)
    pad = int(0.01 * max(img.shape[:2]))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img.shape[1], x + w + pad)
    y1 = min(img.shape[0], y + h + pad)
    return img[y0:y1, x0:x1]


def lines_binary_mask(img: np.ndarray) -> np.ndarray:
    """
    Возвращает бинарную маску линий (0 – линии, 255 – фон).
    Морфология подбирается относительно размеров, чтобы не «выпиливать» стены.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # двоим фон; нам важны тёмные штрихи
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - th  # стены белые на чёрном

    H, W = inv.shape
    kh = max(10, W // 120)   # длина горизонтального штриха
    kv = max(10, H // 120)   # длина вертикального штриха

    horiz = cv2.erode(inv, cv2.getStructuringElement(cv2.MORPH_RECT, (kh, 1)), iterations=1)
    horiz = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (kh, 1)), iterations=1)

    vert = cv2.erode(inv, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kv)), iterations=1)
    vert = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kv)), iterations=1)

    lines = cv2.max(horiz, vert)

    # закрыть мелкие пробелы
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    # приводим к стандарту: фон 255, линии 0
    return 255 - lines


def prepare_for_parse(data: bytes, filename: Optional[str]) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Полный препроцесс:
      - читаем (PDF/JPG/PNG)
      - чистим синие печати
      - выпрямляем небольшой наклон
      - кадрируем по контенту
      - получаем бинарную маску линий (0 – линии, 255 – фон)
    Возвращает (mask, (w, h))
    """
    img = read_image_or_pdf(data, filename)
    img = _suppress_blue_stamps(img)
    img = _rotate_small_angle(img)
    img = _content_crop(img)
    mask = lines_binary_mask(img)
    h, w = mask.shape[:2]
    return mask, (w, h)
