# Рекомендательная система фильмов (Streamlit + FastAPI)

Интерактивное приложение для рекомендаций фильмов с интерфейсом на `Streamlit` и backend API на `FastAPI`. UI обращается к API для получения разных типов рекомендаций: топ популярных, по жанрам, похожие по контенту, коллаборативные (NMF) и персональные рекомендации на основе ваших оценок.

## Возможности
- Топ популярных фильмов на основе взвешенного рейтинга IMDb.
- Рекомендации по жанру и по похожести контента.
- Коллаборативные рекомендации (матричная факторизация NMF) для известных пользователей.
- Персональные рекомендации на основе ваших оценок, сохраняемых в `personal_ratings_detailed.csv`.

## Структура
- `app.py` — UI на Streamlit (см. `app.py:8` для `API_BASE_URL`).
- `api.py` — сервис FastAPI с эндпоинтами (см. `api.py:154-167`).
- `models/` — артефакты и модели (исключены из Git). 
- `data/` — данные (исключены из Git).
- `notebook.ipynb` — формирование артефактов и эксперименты.
- `.gitignore` — исключает большие/чувствительные файлы и артефакты.

## Требования
- Python 3.10+
- Пакеты: `streamlit`, `fastapi`, `uvicorn`, `pandas`, `numpy`, `scikit-learn`, `requests`, при необходимости `jupyter` для ноутбука.

Установка:
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows PowerShell
pip install streamlit fastapi uvicorn pandas numpy scikit-learn requests
# Для работы с ноутбуком:
pip install jupyter
```

## Подготовка артефактов
Положите артефакты в каталог `models/` (создайте при необходимости). Используются файлы:
- `movies.pkl`, `ratings.pkl`, `movie_stats.pkl`
- `tfidf_vectorizer.pkl`, `content_similarity.npz` или `content_similarity.pkl`
- `user_movie_matrix.pkl`, `collaborative_model.pkl`
- `user_mappings.pkl`, `movie_mappings.pkl`

Артефакты можно сгенерировать, выполнив `notebook.ipynb`.

Важно: если артефакты сериализованы из `scikit-learn`, библиотека должна быть установлена для корректной десериализации.

## Запуск API
Есть два варианта:

1) Запуск через команду `uvicorn` на порту, ожидаемом UI (`8000`):
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```
UI по умолчанию смотрит на `http://localhost:8000` (см. `app.py:8`).

2) Запуск `api.py` напрямую:
```bash
python api.py
```
В этом режиме сервер поднимется на `http://localhost:5001` (см. `api.py:625-629`). Чтобы UI работал, либо измените `API_BASE_URL` в `app.py:8` на `http://localhost:5001`, либо запускайте API на `8000`.

Документация Swagger доступна по адресу `http://localhost:8000/docs` (или `http://localhost:5001/docs` при запуске на 5001).

## Запуск UI (Streamlit)
```bash
streamlit run app.py
```
Приложение проверит доступность API (см. `check_api_connection` в `app.py:44-50`). Если API недоступен, проверьте порт и `API_BASE_URL`.

## Персональные оценки
- Первые оценки можно задать во вкладке «Персональные» — создаётся файл `personal_ratings_detailed.csv` в корне проекта.
- После добавления ≥3 оценок UI вызовет переобучение персонального профиля через `POST /api/retrain`.

## Эндпоинты API
- `GET /` — информация о сервисе и список эндпоинтов (`api.py:154-167`).
- `GET /api/recommendations/top-movies?n=10` — топ популярных (`api.py:170-192`).
- `GET /api/recommendations/genre/{genre}?n=10` — по жанру (`api.py:194-219`).
- `GET /api/recommendations/similar/{movie_title}?n=10` — похожие по контенту (`api.py:220-265`).
- `GET /api/recommendations/collaborative/{user_id}?n=10` — коллаборативные (`api.py:266-339`).
- `GET /api/recommendations/personal?n=10` — персональные (`api.py:342-436`).
- `GET /api/user/{user_id}/ratings` — оценки пользователя (`api.py:439-471`).
- `POST /api/retrain` — переобучение персонального профиля (`api.py:564-572`).

## Типичные проблемы
- UI пишет «API недоступен»: запустите API на `8000` или обновите `API_BASE_URL` в `app.py:8`.
- Ошибка при загрузке артефактов (`pickle`/`ModuleNotFoundError`): установите совместимую версию `scikit-learn`/заново сформируйте артефакты.
- Пустые рекомендации: убедитесь, что `models/` заполнен файлами и в `ratings.pkl`/`movies.pkl` есть данные.
