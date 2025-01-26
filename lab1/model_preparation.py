import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

def train_model(train_data_path):
    data = pd.read_csv(train_data_path)
    
    X = data[['Time']]  # Признак
    y = data['Temperature']  # Целевая переменная

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model

if __name__ == '__main__':
    model = train_model('train/temperature_train_scaled.csv')
    joblib.dump(model, 'model.joblib')