import pandas as pd
from sklearn.metrics import accuracy_score
from joblib import load

def evaluation_pipeline(x_test_path, y_test_path, model_paths):
    # Read data
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)

    # Load scaler
    scaler = load('scaling/scaler.pkl')

    # Scale features
    X_test_scaled = scaler.transform(X_test)

    # Load models and make predictions
    predictions = []
    scores = []
    for model_path in model_paths:
        model = load(model_path)
        y_pred = model.predict(X_test_scaled)
        predictions.append(y_pred)
        score = accuracy_score(y_test, y_pred) * 100
        scores.append(score)

    return predictions, scores
