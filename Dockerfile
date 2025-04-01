# Базовый образ с Python 3.10 (slim)
FROM python:3.10-slim-bullseye

WORKDIR /app

# Устанавливаем системные зависимости (включая git и gcc для некоторых библиотек)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Обновляем pip
RUN pip install --upgrade pip

# Устанавливаем рабочую директорию

# Копируем файлы (включая скрипты и requirements.txt)
COPY requirements.txt  /app/

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

EXPOSE 5000

# По умолчанию переходим в интерактивный Bash
CMD ["python", "app.py"]
