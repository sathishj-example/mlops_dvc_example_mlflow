import pandas as pd
import os
import yaml
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import json
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def main(params_path):
    with open(params_path) as f:
        params = yaml.safe_load(f)

    t = params['train']
    preprocess = params['preprocess']
    os.makedirs(os.path.dirname(t['model_output']), exist_ok=True)
    os.makedirs('metrics', exist_ok=True)

    train = pd.read_csv(preprocess['output_train'])
    X = train.drop(columns=['target'])
    y = train['target']

    # Ensure MLflow writes to local mlruns by default; can be overridden with MLFLOW_TRACKING_URI
    mlflow.set_experiment('dvc_mlflow_demo')
    with mlflow.start_run() as run:
        # Log params
        mlflow.log_param('n_estimators', t['n_estimators'])
        mlflow.log_param('max_depth', t['max_depth'])

        model = RandomForestClassifier(n_estimators=t['n_estimators'],
                                       max_depth=t['max_depth'],
                                       random_state=params['seed'])
        model.fit(X, y)
        dump(model, t['model_output'])

        # quick train accuracy
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        metrics = { "train_accuracy": acc }
        with open(t['metrics_output'], "w") as f:
            json.dump(metrics, f)

        # Log metrics and model to MLflow
        mlflow.log_metric('train_accuracy', float(acc))
        mlflow.sklearn.log_model(model, artifact_path='model')

    print(f"Trained model -> {t['model_output']}")
    print(f"Metrics -> {t['metrics_output']}")
    print(f"MLflow run id: {run.info.run_id}")


if __name__ == "__main__":
    import sys
    params_path = sys.argv[1]
    main(params_path)
