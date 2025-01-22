#!/bin/bash

# Создаем наборы данных
python3 data_creation.py

# Предобрабатываем данные
python3 model_preprocessing.py

# Обучаем модель
python3 model_preparation.py

# Тестируем модель
python3 model_testing.py