import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

def test_model(test_data_path, model_path):
    data = pd.read_csv(test_data_path)
    model = joblib.load(model_path)

    X_test = data[['Time']]
    y_test = data['Temperature']

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    print(f"Mean Squared Error: {mse:.2f}")

if __name__ == '__main__':
    test_model('test/temperature_test_scaled.csv', 'model.joblib')