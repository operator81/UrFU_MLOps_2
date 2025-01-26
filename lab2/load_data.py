import pandas as pd

# Загрузка датасета из CSV-файла
file_path = 'cars.csv' 
data = pd.read_csv(file_path)

# Вывод первых строк датасета для проверки
print(data.head())
