# Используем базовый образ Python
FROM python:3.11.8-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл требований в контейнер
COPY requirements.txt .

# Устанавливаем необходимые зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект в контейнер
COPY . .

# Определяем команду для запуска вашего скрипта
CMD ["python", "train_model.py"]