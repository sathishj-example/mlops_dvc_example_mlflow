# MLOps DVC + MLflow Demo

This project demonstrates how to use **DVC (Data Version Control)** together with **MLflow** for experiment tracking and artifact versioning.
This fixed version avoids DVC output duplication by using separate metric files for train and evaluate stages.

## What this demo shows

- Reproducible pipeline managed by DVC (data + models tracked as artifacts).
- Logging run parameters, metrics and model artifacts to MLflow (local `mlruns/` folder).
- How to reproduce experiments and rollback with `git` + `dvc`.

## Steps to run

1. Create venv and install dependencies:

```
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

2. Initialize Git and DVC:

```
git init
dvc init
dvc remote add -d localremote ./dvc_remote
git add . && git commit -m "Initial commit"
```

3. Run pipeline (this will also log to MLflow):

```
dvc repro
```

4. View DVC metrics:

```
dvc metrics show
cat metrics/train_metrics.json
cat metrics/test_metrics.json
```

5. View MLflow UI (in a new terminal, with venv activated):

```
mlflow ui --backend-store-uri mlruns
# then open http://127.0.0.1:5000 in your browser
```

6. Push DVC-tracked artifacts to local remote:

```
dvc push
```

## Notes

- MLflow stores runs and artifacts in the `mlruns/` directory by default in this demo.
- DVC handles large files (data, models) and their versions; MLflow stores experiment metadata and model artifacts.
