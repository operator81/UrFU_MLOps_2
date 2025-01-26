import numpy as np
import pandas as pd
import os

def create_temperature_data(num_samples=1000, noise_level=0.1, anomaly_freq=0.1):
    time = np.arange(num_samples)
    # Создаем базовую синусоидальную модель температуры
    temperature = 20 + 10 * np.sin(0.01 * time)  # Основная форма
    noise = noise_level * np.random.randn(num_samples)  # Шум
    temperature += noise

    # Вставляем аномалии
    anomalies = np.random.choice(num_samples, size=int(num_samples * anomaly_freq), replace=False)
    temperature[anomalies] += np.random.uniform(10, 20, size=anomalies.shape)  # Случайные аномалии

    return pd.DataFrame({'Time': time, 'Temperature': temperature})

def save_data(data, folder, filename):
    os.makedirs(folder, exist_ok=True)
    data.to_csv(os.path.join(folder, filename), index=False)

if __name__ == '__main__':
    # Создаем тренировочный набор данных
    train_data = create_temperature_data(num_samples=800)
    save_data(train_data, 'train', 'temperature_train.csv')

    # Создаем тестовый набор данных
    test_data = create_temperature_data(num_samples=200)
    save_data(test_data, 'test', 'temperature_test.csv')