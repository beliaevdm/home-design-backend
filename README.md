# Home Design Backend

FastAPI-сервис для парсинга планов квартир из изображений/PDF, калибровки масштаба, нормализации и экспорта в SVG.

## Локальный запуск

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```
