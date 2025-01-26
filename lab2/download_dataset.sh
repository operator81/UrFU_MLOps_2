#!/bin/bash

# URL до файла
URL="https://github.com/operator81/UrFU_MLOps_2/raw/6cb0eb83b54249c6fdf365d01106800288cba5c1/lab2/cars.csv"

wget "$URL" -O cars.csv

# Проверка на успешность загрузки
if [ $? -eq 0 ]; then
    echo "Файл успешно загружен: cars.csv"
else
    echo "Произошла ошибка при загрузке файла."
fi

















# python3 -m venv myenv
# source myenv/bin/activate
# pip install -r requirements.txt

# wget https://github.com/operator81/UrFU_MLOps_2/blob/6cb0eb83b54249c6fdf365d01106800288cba5c1/lab2/cars.csv

