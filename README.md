# PriceTag Recognition

Система распознавания ценников для Lenta Tech.

## Быстрый старт

```bash
# Установка зависимостей
pip install -r requirements.txt

# Скачивание моделей
python scripts/download_models.py

# Подготовка датасета (если есть видео)
python scripts/prepare_dataset.py

# Обучение YOLO
python training/train_yolo.py

# Запуск веб-сервера
python app/main.py