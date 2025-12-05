import pandas as pd
import json
import yaml
from joblib import load
from sklearn.metrics import accuracy_score
import mlflow
import os

def main(params_path):
    with open(params_path) as f:
        params = yaml.safe_load(f)

    preprocess = params['preprocess']
    model_path = params['train']['model_output']
    test = pd.read_csv(preprocess['output_test'])
    X_test = test.drop(columns=['target'])
    y_test = test['target']

    model = load(model_path)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    metrics = { "test_accuracy": acc }

    # write test metrics to its own file
    os.makedirs(os.path.dirname(params['evaluate']['metrics_output']), exist_ok=True)
    with open(params['evaluate']['metrics_output'], "w") as f:
        json.dump(metrics, f)

    # Also log to MLflow
    mlflow.set_experiment('dvc_mlflow_demo')
    with mlflow.start_run() as run:
        mlflow.log_metric('test_accuracy', float(acc))

    print(f"Evaluation metrics saved to {params['evaluate']['metrics_output']}")
    print(f"MLflow eval run id: {run.info.run_id}")