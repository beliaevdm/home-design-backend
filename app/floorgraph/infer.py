# app/floorgraph/infer.py
from __future__ import annotations
import os
from typing import Optional, Dict, Any

import numpy as np
import torch
from huggingface_hub import snapshot_download

from app.preprocess import read_image_or_pdf, auto_crop_plan  # используем твой препроцесс

# --------- конфиг через env ----------
HF_FLOOR_REPO = os.getenv("FLOOR_MODEL_REPO", "").strip()
HF_FLOOR_FILENAME = os.getenv("FLOOR_MODEL_FILE", "").strip()   # например: "floormodel.pt" или "best.ckpt"

# кэш путей
_CKPT_PATH: Optional[str] = None
_MODEL: Optional[torch.nn.Module] = None
_AVAILABLE = False


def model_available() -> bool:
    """Есть ли доступная модель (веса скачаны и загружены)."""
    return _AVAILABLE


def ensure_model() -> bool:
    """
    Скачиваем веса с HF (если задано через env) и пытаемся загрузить модель.
    На этом шаге мы только подготавливаем окружение.
    Реальная привязка к конкретной архитектуре будет на следующем шаге.
    """
    global _CKPT_PATH, _MODEL, _AVAILABLE

    if _AVAILABLE:
        return True

    if not HF_FLOOR_REPO:
        # Модель не задана — работаем в fallback-режиме
        _AVAILABLE = False
        return False

    try:
        repo_dir = snapshot_download(repo_id=HF_FLOOR_REPO, local_files_only=False)
    except Exception as e:
        print(f"[floorgraph] download failed: {e}")
        _AVAILABLE = False
        return False

    # находим файл весов
    candidate = None
    if HF_FLOOR_FILENAME:
        cand = os.path.join(repo_dir, HF_FLOOR_FILENAME)
        if os.path.exists(cand):
            candidate = cand
    else:
        # пробуем автоматически угадать по распространённым именам
        for name in ["model.pt", "best.pt", "checkpoint.pt", "best.ckpt", "model.safetensors"]:
            cand = os.path.join(repo_dir, name)
            if os.path.exists(cand):
                candidate = cand
                break

    if candidate is None:
        print("[floorgraph] no weights file found in repo; will use fallback.")
        _AVAILABLE = False
        return False

    _CKPT_PATH = candidate

    # Загрузка конкретной архитектуры будет на шаге 3.
    # Пока просто помечаем, что веса скачены; инференс вернёт None (fallback).
    _MODEL = None
    _AVAILABLE = True
    print(f"[floorgraph] weights ready at: {_CKPT_PATH}")
    return True


def infer_floorplan(
    img_bgr: np.ndarray,
    raise_if_unavailable: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Единая точка входа для AI-распознавания.
    Сейчас — заглушка: если модель не подключена, вернёт None.
    На шаге 3 сюда добавим реальный инференс (FloorTrans / аналог) и маппинг в наш JSON-формат.

    Возвращаемый формат целим такой:
    {
      "image_size": {"w_px": int, "h_px": int},
      "scale": {"mm_per_px": float|None},
      "walls": [{"p1":[x,y], "p2":[x,y]}, ...],
      "rooms": [{"id": int, "contour": [[x,y], ...]}, ...]
    }
    """
    ok = ensure_model()
    if not ok or _MODEL is None:
        if raise_if_unavailable:
            raise RuntimeError("AI model is not loaded yet")
        return None

    # ---------- здесь будет реальный инференс ----------
    # Пример (будет заменён):
    # with torch.no_grad():
    #     pred = _MODEL(in_tensor) ...
    #     walls, rooms = postprocess(pred)
    #     ...
    # return {"image_size": ..., "walls": walls, "rooms": rooms, "scale": {...}}
    # ---------------------------------------------------

    return None
