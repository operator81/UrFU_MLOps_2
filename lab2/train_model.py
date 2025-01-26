import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Загрузка данных
file_path = 'cars.csv'  
data = pd.read_csv(file_path)

# Предположим, что цена является целевой переменной и все остальные - предикторами
X = data.drop('Price(euro)', axis=1)
y = data['Price(euro)']

# Преобразуем категориальные переменные в численные
X = pd.get_dummies(X, drop_first=True)

# Разделяем данные на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем модель
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Обучаем модель
model.fit(X_train, y_train)

print("Модель обучена успешно.")