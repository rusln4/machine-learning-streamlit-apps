# Распознавание цифр (Streamlit + FastAPI)

Интерактивное приложение для распознавания рукописных цифр на основе модели MNIST. UI написан на Streamlit, backend — FastAPI. Интерфейс отправляет изображение цифры на API, получает предсказание класса и вероятности по всем 10 классам.

## Структура
- `streamlit_app.py` — UI на Streamlit (`digit_recognition/streamlit_app.py`)
- `api/main.py` — FastAPI эндпоинт `/predict` (`digit_recognition/api/main.py`)
- `models/` — папка для файла модели `mnist_cnn.keras` (не хранится в Git)
- `requirements.txt` — зависимости подпроекта
- `.streamlit/config.toml` — тема оформления
- `launch.txt` — удобные команды запуска

## Требования
- Python 3.10+
- macOS ARM: `tensorflow-macos` + `tensorflow-metal`
- Иначе: обычный `tensorflow`

## Установка
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows PowerShell
pip install -r digit_recognition/requirements.txt
```

## Подготовка модели
Поместите обученную модель MNIST в:
```
digit_recognition/models/mnist_cnn.keras
```
Backend сам найдёт модель по этому пути. Если файла нет, API вернёт `503` с сообщением «Модель не найдена».

## Запуск
- Backend (порт 8001):
```bash
uvicorn digit_recognition.api.main:app --reload --port 8001
```
- UI:
```bash
streamlit run digit_recognition/streamlit_app.py
```

По умолчанию UI обращается к `http://127.0.0.1:8001/predict`. При необходимости можно переопределить:
```bash
API_URL=http://localhost:8001/predict streamlit run digit_recognition/streamlit_app.py
```

## Эндпоинт API
- `POST /predict`
  - Формы: `file` — PNG/JPG изображения цифры
  - Ответ: `{ prediction: int, probabilities: number[10] }`

Пример:
```bash
curl -F file=@digit.png http://127.0.0.1:8001/predict
```

## Как рисовать в UI
- Поле рисования 280×280 на чёрном фоне
- Белая линия, толщина регулируется слайдером
- Кнопки: «Удалить», «Распознать»
- После распознавания отображаются класс и столбчатая диаграмма вероятностей

## Типичные проблемы
- «API недоступен» в UI: убедитесь, что backend запущен на порту 8001 или задайте `API_URL`.
- «Модель не найдена»: положите `mnist_cnn.keras` в `digit_recognition/models/`.
- macOS M1/M2: ставьте `tensorflow-macos` и `tensorflow-metal` (см. `requirements.txt`).

## Разработка
- UI: `digit_recognition/streamlit_app.py`
- Backend: `digit_recognition/api/main.py`

Модель загружается лениво при первом запросе. Предобработка изображений приводит картинку к размеру 28×28, нормализует до `[0, 1]` и добавляет каналы/батч.