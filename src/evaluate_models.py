import joblib
import pandas as pd
import json
from sklearn.metrics import accuracy_score # Example metric

def evaluate_models():
    # Load models
    model_RF = joblib.load('models/model_RF.pkl')
    model_LR = joblib.load('models/model_LR.pkl')

    # Load evaluation data
    eval_data = pd.read_csv('data/features.csv')
    X_eval = eval_data.drop('target', axis=1)
    y_eval = eval_data['target']

    # Evaluate models
    predictions_RF = model_RF.predict(X_eval)
    predictions_LR = model_LR.predict(X_eval)
    
    accuracy_RF = accuracy_score(y_eval, predictions_RF)
    accuracy_LR = accuracy_score(y_eval, predictions_LR)

    # Save metrics
    metrics = {
        'model_RF_accuracy': accuracy_RF,
        'model_LR_accuracy': accuracy_LR
    }
    with open('reports/metrics_report.json', 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    evaluate_models()
