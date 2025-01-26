import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    scaler = StandardScaler()
    
    # Предобрабатываем только температурные данные
    normalized_temp = scaler.fit_transform(data[['Temperature']])
    data['Temperature'] = normalized_temp
    
    return data

if __name__ == '__main__':
    # Предобрабатываем тренировочные данные
    train_data = preprocess_data('train/temperature_train.csv')
    train_data.to_csv('train/temperature_train_scaled.csv', index=False)

    # Предобрабатываем тестовые данные
    test_data = preprocess_data('test/temperature_test.csv')
    test_data.to_csv('test/temperature_test_scaled.csv', index=False)