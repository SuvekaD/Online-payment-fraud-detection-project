import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score as ras
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from joblib import dump, load

def evaluation_metrics(x_test_path, y_test_path, model_path, scaler_path, encoder_path):
    X = pd.read_csv(x_test_path)
    y_target = pd.read_csv(y_test_path)

    scaler = load('scaling/scaler.pkl')
    numerical_columns = X.select_dtypes(exclude='object')
    categorical_columns = X.select_dtypes(include='object')

     encoder = load('encoder/df_new.pkl')
    categorical_encoded = encoder.transform(categorical_columns)

    numerical_scaled = scaler.transform(numerical_columns)
    Features = pd.concat([categorical_encoded, numerical_scaled], axis=1)

    log_reg = load(model_path)
    y_pred = pd.DataFrame(log_reg.predict(Features))
    test_score = accuracy_score(y_target, y_pred) * 100

    return y_pred, test_score
