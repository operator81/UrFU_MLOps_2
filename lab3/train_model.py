import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import joblib

# Загрузка датасета Iris
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Разделение на обучающую и тестовую выборки
X = data.iloc[:, :-1]  # Все колонки, кроме целевой
y = data['target']      # Целевая переменная

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Сохранение датасетов для тренировки и тестирования
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Обучение модели
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Сохранение обученной модели
joblib.dump(model, 'logistic_regression_model.pkl')

print("Обработка данных завершена и модель сохранена.")