# Home Design Backend

FastAPI сервис: парсинг планов квартир из PDF/JPG/PNG, калибровка масштаба, нормализация, SVG, измерения стен и площадей.

## Запуск

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
