from joblib import load
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def evaluation_pipeline(x_test_path, y_test_path, model_path):
    X = pd.read_csv(x_test_path)
    y_target = pd.read_csv(y_test_path)
    
    numerical_columns = X.select_dtypes(exclude='object')

    scaler = load('Scaling/scaler.pkl')
    numerical_scaled_data = scaler.transform(numerical_columns)
    numerical_scaled_data = pd.DataFrame(numerical_scaled_data, columns=numerical_columns.columns)

    Features = pd.concat([numerical_scaled_data], axis=1)

    model = load(model_path)
    y_pred = model.predict(Features)
    test_score = accuracy_score(y_target, y_pred) * 100

    return y_pred, test_score
